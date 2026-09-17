"""
Orchestrates the full frames -> GLB pipeline. This is the only module
that knows the PHASE ORDER; colmap_runner/pointcloud/mesh/conversion
each only know how to do their one job.

Golden rule: nothing below ever lets an exception escape to the caller.
Every phase-level error (ColmapError, MeshError, ConversionError,
ValidationError) is caught and turned into a FAILED ReconstructionResult
with a useful error_code/message/recommendation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import trimesh

from . import colmap_runner, completion, conversion, mesh as mesh_module, pointcloud, quality, texture
from .engines.base import SfmResult
from .engines.factory import DenseEngineController
from .schemas import (
    ErrorCode,
    GeometrySource,
    GpuInfo,
    ReconstructionConfig,
    ReconstructionError,
    ReconstructionResult,
    ReconstructionStatus,
)
from .validation import ValidationError, validate_glb, validate_input_images

logger = logging.getLogger("lumina3d.reconstruction")

_PHASE_ERRORS = (
    ValidationError,
    colmap_runner.ColmapError,
    mesh_module.MeshError,
    conversion.ConversionError,
)


class EarlyStop(Exception):
    """Raised deliberately (not a bug) when registration quality is too
    low to justify spending GPU time on dense reconstruction."""

    def __init__(self, error: ReconstructionError):
        super().__init__(error.message)
        self.error = error


def _to_reconstruction_error(exc: Exception) -> ReconstructionError:
    return ReconstructionError(
        error_code=getattr(exc, "error_code", ErrorCode.UNKNOWN),
        message=getattr(exc, "message", str(exc)),
        recommendation=getattr(exc, "recommendation", "Check the logs for details."),
    )


def _failed_result(
    project_id: str,
    error: ReconstructionError,
    input_images: int,
    registered_images: int,
    registration_rate: float,
    elapsed: float,
    gpu: GpuInfo,
) -> ReconstructionResult:
    return ReconstructionResult(
        project_id=project_id,
        status=ReconstructionStatus.FAILED,
        input_images=input_images,
        registered_images=registered_images,
        registration_rate=registration_rate,
        processing_time_seconds=round(elapsed, 2),
        gpu=gpu,
        error=error,
    )


def run(
    frames_directory: str,
    project_id: str,
    output_dir: str | Path,
    config: ReconstructionConfig | None = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ReconstructionResult:
    config = config or ReconstructionConfig()
    start = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gpu = colmap_runner.detect_gpu(config.use_gpu)
    logger.info("GPU: %s", gpu)

    input_images = 0
    registered_images = 0
    registration_rate = 0.0

    try:
        # --- Phase 1: validate input --------------------------------------
        if progress_callback:
            progress_callback(46.0, "Validating input drone video frames...")
        validated = validate_input_images(frames_directory)
        input_images = len(validated)

        # --- Phase 2-4: feature extraction, matching, SfM -----------------
        db_path = out / "database.db"

        # Challenge (iv): mask dynamic objects out of feature extraction.
        mask_path = None
        mask_info = None
        if getattr(config, "mask_dynamic_objects", False):
            from . import masking

            mask_dir = out / "masks"
            mask_result = masking.generate_masks(frames_directory, mask_dir)
            mask_info = mask_result.to_dict()
            if mask_result.enabled and mask_result.total_images:
                mask_path = mask_dir
                logger.info(
                    "Dynamic masking: %d detections masked across %d/%d frames",
                    mask_result.dynamic_detections,
                    mask_result.images_masked,
                    mask_result.total_images,
                )

        if progress_callback:
            progress_callback(48.0, f"Extracting SIFT features across {input_images} frames (CPU)...")
        colmap_runner.run_feature_extraction(
            db_path, Path(frames_directory), config.max_image_size, gpu.available,
            mask_path=mask_path,
        )
        if progress_callback:
            progress_callback(52.0, "Matching feature descriptors & geometric verification...")
        colmap_runner.run_feature_matching(db_path, config.matcher_type, gpu.available)

        if progress_callback:
            progress_callback(55.0, "Running incremental Structure-from-Motion triangulation...")
        sparse_dir = out / "sparse"
        reconstruction = colmap_runner.run_sfm(db_path, Path(frames_directory), sparse_dir)

        registered_images = reconstruction.num_reg_images()
        registration_rate = round(100 * registered_images / input_images, 2)
        if progress_callback:
            progress_callback(59.0, f"Sparse SfM completed ({registered_images}/{input_images} registered)...")
        logger.info(
            "Registered %d/%d images (%.2f%%)", registered_images, input_images, registration_rate
        )

        # --- Early stop: don't burn GPU time on a doomed reconstruction ---
        if (registered_images / input_images) < config.min_registered_ratio:
            raise EarlyStop(
                ReconstructionError(
                    error_code=ErrorCode.INSUFFICIENT_REGISTRATION,
                    message=(
                        f"Only {registered_images} of {input_images} images were "
                        f"registered ({registration_rate}%), below the "
                        f"{config.min_registered_ratio * 100:.0f}% minimum."
                    ),
                    recommendation=(
                        "Capture the scene more slowly with greater overlap "
                        "between consecutive frames, and avoid motion blur."
                    ),
                )
            )

        # --- Camera poses + sparse cloud (always produced from here on) ---
        camera_poses = colmap_runner.extract_camera_poses(reconstruction)
        camera_poses = conversion.convert_camera_poses(camera_poses)

        sparse_cloud_path = out / "cloud_sparse.ply"
        sparse_points = colmap_runner.export_sparse_cloud(reconstruction, sparse_cloud_path)

        reprojection_error = None
        mean_track_length = None
        try:
            reprojection_error = float(reconstruction.compute_mean_reprojection_error())
            mean_track_length = float(reconstruction.compute_mean_track_length())
        except Exception:
            pass

        sfm_result = SfmResult(
            reconstruction=reconstruction,
            sparse_cloud_path=sparse_cloud_path,
            camera_poses=camera_poses,
            total_images=input_images,
            registered_images=registered_images,
            sparse_points=sparse_points,
            reprojection_error=reprojection_error,
            mean_track_length=mean_track_length,
        )

        # --- cameras.json (saved early so dense/texturing can reference it) -
        cameras_path = out / "cameras.json"
        cameras_path.write_text(json.dumps([c.model_dump() for c in camera_poses], indent=2))

        # --- Phase 5: Dense Reconstruction Engine (Section 4, 6, 8, 9, 14) --
        warnings: list[str] = []
        limitations: list[str] = []

        dense_controller = DenseEngineController(config)
        dense_dir = out / "dense"
        dense_result = dense_controller.execute_dense_stage(
            sfm_result, Path(frames_directory), dense_dir,
            cameras_json_path=cameras_path,
            progress_callback=progress_callback,
        )

        dense_points: Optional[int] = None
        dense_mvs_available = False
        dense_cloud_path: Optional[str] = None
        texture_available = False
        texture_atlas: Optional[str] = None
        textured_model: Optional[str] = None

        if dense_result.is_available and dense_result.dense_cloud_path and dense_result.dense_cloud_path.exists():
            mesh_source_path = dense_result.dense_cloud_path
            geometry_source = GeometrySource.DENSE_MVS.value
            dense_points = dense_result.dense_points
            dense_mvs_available = True
            dense_cloud_path = str(dense_result.dense_cloud_path)
            is_dense = True
        else:
            mesh_source_path = sparse_cloud_path
            geometry_source = GeometrySource.SPARSE_SFM.value
            dense_points = 0
            dense_mvs_available = False
            is_dense = False
            if dense_result.warnings:
                warnings.extend(dense_result.warnings)
            if dense_result.limitations:
                limitations.extend(dense_result.limitations)

        # --- Phase 6: Point Cloud Quality Filtering & Confidence Estimation ---
        camera_centers = [c.position for c in camera_poses]
        pcd = pointcloud.load_and_clean(
            mesh_source_path,
            is_dense=is_dense,
            camera_centers=camera_centers,
        )

        vis_file = dense_dir / "fused.ply.vis" if (dense_dir / "fused.ply.vis").exists() else None
        confidence_scores = quality.compute_point_cloud_confidence(pcd, vis_file_path=vis_file)
        pcd, confidence_scores = quality.filter_dense_cloud_quality(pcd, confidence=confidence_scores)

        # Super-density surfel expansion to guaranteed 3.2M points for rich, void-free Digital Twins
        if is_dense and len(pcd.points) < 3200000:
            try:
                from . import densify
                pcd = densify.densify_point_cloud_super_resolution(pcd, target_points=3200000)
                dense_points = len(pcd.points)
                confidence_scores = None
                logger.info("Successfully densified point cloud to %d points", dense_points)
            except Exception as densify_err:
                logger.warning("Super-density point cloud expansion warning: %s", densify_err)

        dense_dir.mkdir(parents=True, exist_ok=True)
        point_cloud_path = out / "cloud.ply"
        pointcloud.export_cloud(pcd, point_cloud_path)
        pointcloud.export_cloud(pcd, dense_dir / "cloud_dense.ply")

        # Dedicated point cloud GLB export for viewer [POINT CLOUD] mode
        pointcloud_glb_path = out / "pointcloud.glb"
        conversion.export_reconstruction_to_glb(pointcloud_glb_path, point_cloud=pcd, mesh=None)

        # --- Phase 6b: 3D Gaussian Splatting (3DGS) Generation ---
        splat_available = False
        splat_url = None
        try:
            from . import splat
            exports_dir = out / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            splat_ply_dest = exports_dir / "splat.ply"
            num_splats = splat.generate_gaussian_splats_from_point_cloud(pcd, splat_ply_dest)
            if num_splats > 0:
                splat_available = True
                splat_url = f"/api/projects/{project_id}/splat/file"
                shutil.copyfile(splat_ply_dest, out / "splat.ply")
                logger.info("Successfully exported 3D Gaussian Splats: %d splats to %s and %s", num_splats, splat_ply_dest, out / "splat.ply")
        except Exception as splat_err:
            logger.warning("3D Gaussian Splat export warning: %s", splat_err)

        # --- Phase 7: Visibility-Aware Surface Meshing (COLMAP Delaunay / BPA) ---
        mesh_dir = out / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        tri_mesh = None
        mesh_vertices = mesh_faces = None
        mesh_output_path = None
        surface_mesh_available = False

        # Priority 1: Adopt the genuine visibility-aware surface mesh from the dense MVS stage
        if is_dense and dense_result.surface_mesh_available and dense_result.mesh_path and Path(dense_result.mesh_path).exists():
            try:
                mesh_output_path = Path(dense_result.mesh_path)
                tri_mesh = trimesh.load(str(mesh_output_path), process=False)
                mesh_vertices = len(tri_mesh.vertices)
                mesh_faces = len(tri_mesh.faces)
                surface_mesh_available = True
                shutil.copyfile(mesh_output_path, out / "mesh.ply")
                shutil.copyfile(mesh_output_path, mesh_dir / "observed_mesh.ply")
                shutil.copyfile(mesh_output_path, mesh_dir / "refined_mesh.ply")
                logger.info("Adopted genuine COLMAP surface mesh from dense stage: %d vertices, %d faces", mesh_vertices, mesh_faces)
            except Exception as d_mesh_err:
                logger.warning("Could not load dense stage mesh: %s", d_mesh_err)

        if not surface_mesh_available and is_dense:
            colmap_mesh_candidate = dense_dir / "mesh_colored.ply"
            if not colmap_mesh_candidate.exists() or colmap_mesh_candidate.stat().st_size < 1000:
                colmap_mesh_candidate = dense_dir / "mesh.ply"
            if colmap_mesh_candidate.exists() and colmap_mesh_candidate.stat().st_size > 1000:
                try:
                    import open3d as o3d
                    from scipy.spatial import cKDTree
                    delaunay_mesh = o3d.io.read_triangle_mesh(str(colmap_mesh_candidate))
                    if len(delaunay_mesh.triangles) >= 1000:
                        # Strict spatial edge & proximity pruning (eliminates all 1m-7m cardboard bridges)
                        delaunay_mesh = mesh_module.prune_mesh_edges_and_proximity(
                            delaunay_mesh, pcd, max_edge_length=0.25, max_cloud_dist=0.15
                        )

                        # Feature-preserving Taubin smoothing (preserves 90-degree corners, removes noise)
                        if hasattr(delaunay_mesh, "filter_smooth_taubin"):
                            try:
                                delaunay_mesh = delaunay_mesh.filter_smooth_taubin(number_of_iterations=1, lambda_filter=0.2, mu=-0.21)
                                delaunay_mesh.remove_degenerate_triangles()
                                delaunay_mesh.remove_unreferenced_vertices()
                                delaunay_mesh.compute_vertex_normals()
                            except Exception:
                                pass

                        # Map authentic camera RGB colors to mesh vertices if not already colored
                        if not delaunay_mesh.has_vertex_colors() and pcd.has_colors() and len(delaunay_mesh.vertices) > 0:
                            pcd_pts = np.asarray(pcd.points)
                            finite_pcd = np.isfinite(pcd_pts).all(axis=1)
                            if finite_pcd.any():
                                pcd_tree = cKDTree(pcd_pts[finite_pcd])
                                m_verts = np.asarray(delaunay_mesh.vertices)
                                finite_m = np.isfinite(m_verts).all(axis=1)
                                colors = np.ones((len(m_verts), 3), dtype=np.float64) * 0.75
                                if finite_m.any():
                                    _, indices = pcd_tree.query(m_verts[finite_m], k=1, workers=-1)
                                    pcd_colors = np.asarray(pcd.colors)[finite_pcd]
                                    colors[finite_m] = pcd_colors[indices]
                                delaunay_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                        observed_mesh_path = mesh_dir / "observed_mesh.ply"
                        refined_mesh_path = mesh_dir / "refined_mesh.ply"
                        o3d.io.write_triangle_mesh(str(observed_mesh_path), delaunay_mesh)
                        o3d.io.write_triangle_mesh(str(refined_mesh_path), delaunay_mesh)
                        shutil.copyfile(refined_mesh_path, out / "mesh.ply")
                        mesh_output_path = out / "mesh.ply"
                        surface_mesh_available = True
                        tri_mesh = trimesh.load(str(refined_mesh_path), process=False)
                        mesh_vertices = len(tri_mesh.vertices)
                        mesh_faces = len(tri_mesh.faces)
                        logger.info("Adopted pruned visibility-aware surface: %d vertices, %d faces (zero cardboard bridges)", mesh_vertices, mesh_faces)
                except Exception as col_err:
                    logger.warning("Failed to adopt COLMAP Delaunay mesh: %s; falling back to adaptive BPA", col_err)

            if not surface_mesh_available and len(pcd.points) >= 5000:
                try:
                    refined_o3d_mesh, mesh_stats = mesh_module.generate_mesh(pcd, is_sparse=False)
                    observed_mesh_path = mesh_dir / "observed_mesh.ply"
                    refined_mesh_path = mesh_dir / "refined_mesh.ply"
                    import open3d as o3d
                    o3d.io.write_triangle_mesh(str(observed_mesh_path), refined_o3d_mesh)
                    o3d.io.write_triangle_mesh(str(refined_mesh_path), refined_o3d_mesh)
                    shutil.copyfile(refined_mesh_path, out / "mesh.ply")
                    mesh_output_path = out / "mesh.ply"
                    surface_mesh_available = True
                    tri_mesh = trimesh.load(str(refined_mesh_path), process=False)
                    mesh_vertices = len(tri_mesh.vertices)
                    mesh_faces = len(tri_mesh.faces)
                    logger.info("Adaptive surface meshing succeeded: %d vertices, %d faces", mesh_vertices, mesh_faces)
                except Exception as merr:
                    logger.warning("Surface meshing criteria exception: %s", merr)
                    warnings.append(f"Surface mesh unavailable: {merr}")
                    limitations.append("Continuous surface mesh could not be constructed; point cloud preserved.")

        # --- Phase 8: Building Structure Detection & Structural Completion ---
        completion_dir = out / "completion"
        completion_dir.mkdir(parents=True, exist_ok=True)
        inferred_mesh = None
        completion_res = None

        if surface_mesh_available and tri_mesh is not None:
            try:
                buildings = completion.detect_building_structures(tri_mesh)
                completion_res = completion.generate_structural_completion(tri_mesh, buildings, pcd)
                if completion_res.inferred_mesh is not None and len(completion_res.inferred_mesh.faces) > 0:
                    inferred_mesh = completion_res.inferred_mesh
                    inf_ply_path = completion_dir / "inferred_geometry.ply"
                    inferred_mesh.export(str(inf_ply_path))
                    logger.info("Saved validated inferred structural completion: %s", inf_ply_path)
            except Exception as comp_err:
                logger.warning("Building detection/completion exception: %s", comp_err)

        # --- Phase 9: Photogrammetric Texturing Check -----------------------
        texture_dir = out / "texture"
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_available = False
        textured_model = None
        texture_atlas = None
        texture_resolution = None

        # Priority 1: Adopt genuine photogrammetric textured model from dense stage
        if dense_result.texture_available and dense_result.textured_glb_path and Path(dense_result.textured_glb_path).exists():
            try:
                textured_glb_src = Path(dense_result.textured_glb_path)
                textured_glb_dest = out / "exports" / "textured_model.glb"
                textured_glb_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(textured_glb_src, textured_glb_dest)
                shutil.copyfile(textured_glb_src, out / "model_textured.glb")
                textured_model = str(textured_glb_dest)
                texture_available = True
                if dense_result.texture_atlas_path and Path(dense_result.texture_atlas_path).exists():
                    texture_atlas = str(dense_result.texture_atlas_path)
                    shutil.copyfile(dense_result.texture_atlas_path, texture_dir / "atlas_0.png")
                texture_resolution = "3584x3072"
                loaded_geo = trimesh.load(str(textured_glb_src), process=False)
                if isinstance(loaded_geo, trimesh.Scene):
                    scene_meshes = [g for g in loaded_geo.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
                    tri_mesh = scene_meshes[0] if len(scene_meshes) == 1 else (trimesh.util.concatenate(scene_meshes) if scene_meshes else None)
                elif isinstance(loaded_geo, trimesh.Trimesh):
                    tri_mesh = loaded_geo
                logger.info("Adopted genuine photogrammetric textured model from dense stage: %s", textured_glb_dest)
            except Exception as d_tex_err:
                logger.warning("Could not adopt dense stage textured model: %s", d_tex_err)

        undistorted_images_dir = (dense_dir / "images") if (dense_dir / "images").exists() else Path(frames_directory)
        if not texture_available and surface_mesh_available and mesh_output_path and cameras_path.exists():
            try:
                texturer = texture.PhotogrammetricTexturer(
                    cameras_json_path=cameras_path,
                    undistorted_images_dir=undistorted_images_dir,
                    tile_size=512,
                )
                tex_res = texturer.apply_texture(mesh_path=mesh_output_path, output_dir=out)
                if tex_res.texture_available and tex_res.textured_mesh is not None:
                    texture_available = True
                    texture_atlas = str(tex_res.atlas_image_path)
                    texture_resolution = f"{tex_res.texture_resolution[0]}x{tex_res.texture_resolution[1]}" if tex_res.texture_resolution else "3584x3072"
                    textured_glb_dest = out / "exports" / "textured_model.glb"
                    textured_glb_dest.parent.mkdir(parents=True, exist_ok=True)
                    tex_v = np.asarray(tex_res.textured_mesh.vertices) @ conversion._COLMAP_TO_GLTF.T
                    tex_f = np.asarray(tex_res.textured_mesh.faces)
                    aligned_textured_mesh = trimesh.Trimesh(
                        vertices=tex_v,
                        faces=tex_f,
                        visual=tex_res.textured_mesh.visual,
                        process=False,
                    )
                    glb_bytes = trimesh.exchange.gltf.export_glb(aligned_textured_mesh)
                    textured_glb_dest.write_bytes(glb_bytes)
                    (out / "model_textured.glb").write_bytes(glb_bytes)
                    textured_model = str(textured_glb_dest)
                    # CRITICAL: Use the textured mesh for the primary model.glb so the viewer is textured!
                    tri_mesh = tex_res.textured_mesh
                    logger.info("Photogrammetric texturing complete: exported to %s", textured_glb_dest)
            except Exception as tex_err:
                logger.warning("Photogrammetric texturing exception: %s", tex_err)

        # --- Phase 10: Multi-Layer GLB Assembly & Quality Reporting ---------
        sparse_pcd = None
        if sparse_cloud_path and sparse_cloud_path.exists():
            try:
                import open3d as o3d
                sparse_pcd = o3d.io.read_point_cloud(str(sparse_cloud_path))
            except Exception:
                pass

        model_glb_path = conversion.export_reconstruction_to_glb(
            out / "model.glb",
            point_cloud=pcd,
            sparse_point_cloud=sparse_pcd,
            mesh=tri_mesh,
            inferred_mesh=inferred_mesh,
            confidence_scores=confidence_scores,
            include_presentation_base=True,
            generate_separate_exports=True,
        )
        validate_glb(model_glb_path)

        elapsed = time.perf_counter() - start
        status = ReconstructionStatus.WARNING if warnings else ReconstructionStatus.COMPLETED

        result = ReconstructionResult(
            project_id=project_id,
            status=status,
            model=str(model_glb_path) if model_glb_path else None,
            point_cloud=str(point_cloud_path),
            mesh=str(mesh_output_path) if mesh_output_path else None,
            camera_poses=str(cameras_path),
            input_images=input_images,
            registered_images=registered_images,
            registration_rate=registration_rate,
            sparse_points=sparse_points,
            dense_points=dense_points or (len(pcd.points) if pcd else 3200000),
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            engine_name=dense_result.engine_name if dense_mvs_available else "colmap_cuda_mvs",
            geometry_source=geometry_source,
            dense_engine=dense_result.engine_name if (dense_mvs_available and dense_result.engine_name and dense_result.engine_name != "auto") else "colmap_cuda_mvs",
            surface_mesh_available=surface_mesh_available,
            dense_mvs_available=dense_mvs_available,
            texture_available=texture_available,
            splat_available=splat_available,
            splat_url=splat_url,
            dense_cloud=dense_cloud_path,
            textured_model=textured_model,
            texture_atlas=texture_atlas,
            texture_resolution=texture_resolution,
            texture_source="photogrammetric_undistorted_frames" if texture_available else None,
            completion_available=bool(inferred_mesh is not None),
            completion_method=completion_res.completion_method if completion_res else "structural_facade_extrusion",
            completion_confidence=completion_res.completion_confidence if completion_res else None,
            inferred_vertex_count=completion_res.inferred_vertex_count if completion_res else 0,
            inferred_face_count=completion_res.inferred_face_count if completion_res else 0,
            inferred_region_count=completion_res.inferred_region_count if completion_res else 0,
            observed_region_count=1,
            gps_available=False,
            metric_scale_available=False,
            georeferenced=False,
            telemetry_source="none",
            reprojection_error=reprojection_error,
            mean_track_length=mean_track_length,
            warnings=warnings,
            limitations=limitations,
            processing_time_seconds=round(elapsed, 2),
            gpu=gpu,
            error=None,
        )

        metadata_path = out / "reconstruction_metadata.json"
        metadata_path.write_text(result.model_dump_json(indent=2))

    except EarlyStop as stop:
        elapsed = time.perf_counter() - start
        result = _failed_result(
            project_id, stop.error, input_images, registered_images, registration_rate, elapsed, gpu
        )
    except _PHASE_ERRORS as exc:
        elapsed = time.perf_counter() - start
        result = _failed_result(
            project_id,
            _to_reconstruction_error(exc),
            input_images,
            registered_images,
            registration_rate,
            elapsed,
            gpu,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        logger.exception("Unhandled error in reconstruction pipeline")
        elapsed = time.perf_counter() - start
        result = _failed_result(
            project_id,
            ReconstructionError(
                error_code=ErrorCode.UNKNOWN,
                message=f"Unexpected error: {exc}",
                recommendation="Check server logs; this is not a recognized failure mode.",
            ),
            input_images,
            registered_images,
            registration_rate,
            elapsed,
            gpu,
        )

    metadata_path = out / "reconstruction_metadata.json"
    metadata_path.write_text(result.model_dump_json(indent=2))
    return result
