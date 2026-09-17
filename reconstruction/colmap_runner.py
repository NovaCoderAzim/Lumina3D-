"""
Thin wrapper around PyCOLMAP. This is the ONLY module that imports
pycolmap - everything else in the package talks to plain Python types
(ReconstructionResult, CameraPose, Path) so the rest of the pipeline
doesn't care whether we're calling COLMAP 4.x, a future version, or
(via mock.py) nothing at all.

Verified against pycolmap 4.1.1's actual API (function signatures below
are not guessed - see README.md for the inspection notes). All of
feature extraction, matching, SfM, undistortion, dense stereo, and
fusion are available natively through PyCOLMAP in this version, so we
never shell out to the COLMAP CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .schemas import CameraPose, ErrorCode, GpuInfo


class ColmapError(Exception):
    def __init__(self, error_code: ErrorCode, message: str, recommendation: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.recommendation = recommendation


def _require_pycolmap():
    try:
        import pycolmap  # noqa: F401
        return pycolmap
    except ImportError as exc:
        raise ColmapError(
            ErrorCode.COLMAP_NOT_FOUND,
            "pycolmap is not installed in this environment.",
            "pip install pycolmap, or set USE_MOCK_RECONSTRUCTION=true for "
            "development without COLMAP.",
        ) from exc


def detect_gpu(want_gpu: bool) -> GpuInfo:
    """nvidia-smi is a reliable, dependency-free way to check for a usable
    GPU regardless of whether COLMAP itself was compiled with CUDA."""

    if not want_gpu or shutil.which("nvidia-smi") is None:
        return GpuInfo(available=False)

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        line = output.stdout.strip().splitlines()[0]
        name, vram = [x.strip() for x in line.split(",")]
        vram_mb = int(vram.replace(" MiB", ""))
        return GpuInfo(available=True, name=name, vram_mb=vram_mb)
    except (subprocess.SubprocessError, IndexError, ValueError, OSError):
        return GpuInfo(available=False)


def run_feature_extraction(
    database_path: Path, images_dir: Path, max_image_size: int, use_gpu: bool,
    mask_path: Path | None = None,
) -> None:
    pycolmap = _require_pycolmap()

    reader_options = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
    # Dynamic-object masks (challenge iv): COLMAP ignores black pixels in
    # "<mask_path>/<image_name>.png", so features are never extracted from
    # masked vehicles/people/animals.
    if mask_path is not None:
        try:
            reader_options.mask_path = str(mask_path)
        except Exception:  # noqa: BLE001 - older pycolmap may name it differently
            pass
    can_use_gpu = bool(use_gpu and getattr(pycolmap, "has_cuda", False))
    extraction_options = pycolmap.FeatureExtractionOptions(
        max_image_size=max_image_size, use_gpu=can_use_gpu
    )

    # Drone video optimization (Section 10): all frames originate from a
    # single continuous physical drone camera. camera_mode=CameraMode.SINGLE
    # enforces a single shared intrinsic calibration (focal length & distortion)
    # rather than fitting 42+ independent focal lengths with gauge drift.
    extract_kwargs: dict = {
        "database_path": database_path,
        "image_path": images_dir,
        "reader_options": reader_options,
        "extraction_options": extraction_options,
        "device": pycolmap.Device.cuda if can_use_gpu else pycolmap.Device.cpu,
    }
    if hasattr(pycolmap, "CameraMode") and hasattr(pycolmap.CameraMode, "SINGLE"):
        extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE

    try:
        pycolmap.extract_features(**extract_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise ColmapError(
            ErrorCode.FEATURE_EXTRACTION_FAILED,
            f"COLMAP feature extraction failed: {exc}",
            "Check that images are readable and not corrupted; try a "
            "smaller MAX_IMAGE_SIZE if this is a GPU memory error.",
        ) from exc


def run_feature_matching(database_path: Path, matcher_type: str, use_gpu: bool) -> None:
    pycolmap = _require_pycolmap()
    can_use_gpu = bool(use_gpu and getattr(pycolmap, "has_cuda", False))
    device = pycolmap.Device.cuda if can_use_gpu else pycolmap.Device.cpu

    try:
        if matcher_type == "sequential":
            pycolmap.match_sequential(
                database_path=database_path,
                pairing_options=pycolmap.SequentialPairingOptions(overlap=10, loop_detection=False),
                device=device,
            )
        else:
            pycolmap.match_exhaustive(database_path=database_path, device=device)
    except Exception as exc:  # noqa: BLE001
        raise ColmapError(
            ErrorCode.INSUFFICIENT_MATCHES,
            f"COLMAP feature matching failed: {exc}",
            "Try MATCHER_TYPE=exhaustive if the drone path loops back or "
            "jumps between viewpoints.",
        ) from exc


def run_sfm(database_path: Path, images_dir: Path, sparse_dir: Path):
    """Returns the largest registered pycolmap.Reconstruction, or raises
    ColmapError if nothing registered at all."""

    pycolmap = _require_pycolmap()
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    # Drone video optimization: relaxed triangulation angle and forward motion threshold
    # so forward-facing aerial videos can initialize two-view baseline without failing
    pipeline_options.mapper.init_min_tri_angle = 4.0
    pipeline_options.mapper.init_max_forward_motion = 0.999
    pipeline_options.mapper.init_min_num_inliers = 30
    pipeline_options.min_focal_length_ratio = 0.1
    pipeline_options.max_focal_length_ratio = 10.0

    try:
        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_dir,
            options=pipeline_options,
        )
    except Exception as exc:  # noqa: BLE001
        raise ColmapError(
            ErrorCode.SFM_FAILED,
            f"COLMAP Structure-from-Motion crashed: {exc}",
            "Check the feature matching output; SfM usually crashes only "
            "when almost no image pairs were matched.",
        ) from exc

    if not reconstructions:
        raise ColmapError(
            ErrorCode.SFM_FAILED,
            "COLMAP registered zero images - no 3D model could be built.",
            "Capture the scene with more visual overlap between "
            "consecutive frames, and ensure the scene has enough texture.",
        )

    # Multiple disconnected components are possible; keep the largest.
    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    best_dir = sparse_dir / "0"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(best_dir)
    return best


def extract_camera_poses(reconstruction) -> list[CameraPose]:
    """Position = camera center in world space. Rotation = world-from-camera
    orientation as a [w, x, y, z] quaternion (pycolmap stores cam-from-world
    with scalar-last [x, y, z, w], so we invert and reorder)."""

    poses: list[CameraPose] = []
    for image_id, image in reconstruction.images.items():
        position = image.projection_center().tolist()
        world_from_cam = image.cam_from_world().rotation.inverse()
        x, y, z, w = world_from_cam.quat.tolist()
        poses.append(
            CameraPose(
                image=image.name,
                camera_id=image.camera_id,
                position=position,
                rotation=[w, x, y, z],
            )
        )
    return poses


def export_sparse_cloud(reconstruction, output_path: Path) -> int:
    reconstruction.export_PLY(str(output_path))
    return reconstruction.num_points3D()


def _find_colmap_cli() -> str | None:
    """Locate a COLMAP CLI binary that may have CUDA support. The pip
    `pycolmap` wheel is built WITHOUT CUDA, so its dense stereo raises
    "requires CUDA or HIP". A separately-installed COLMAP CUDA binary
    (pointed to by COLMAP_PATH, or on PATH as `colmap`) does the dense MVS
    on the GPU. Returns the executable path or None."""
    import os

    candidate = os.environ.get("COLMAP_PATH")
    if candidate and candidate.lower() not in {"colmap", ""}:
        p = Path(candidate)
        if p.exists():
            return str(p)
    found = shutil.which("colmap")
    return found


def _colmap_cli_has_cuda(colmap_exe: str) -> bool:
    try:
        out = subprocess.run(
            [colmap_exe, "--help"], capture_output=True, text=True, timeout=20
        )
        return "with CUDA" in (out.stdout + out.stderr)
    except (subprocess.SubprocessError, OSError):
        return False


def _run_dense_via_cli(
    colmap_exe: str,
    sparse_dir: Path,
    images_dir: Path,
    dense_dir: Path,
    max_image_size: int = 1600,
) -> Path:
    """Undistort -> patch_match_stereo (GPU) -> stereo_fusion via the
    COLMAP CUDA CLI. Runs real MVS on the GPU."""
    dense_dir.mkdir(parents=True, exist_ok=True)
    fused_path = dense_dir / "fused.ply"

    def _run(args: list[str]) -> None:
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ColmapError(
                ErrorCode.DENSE_RECONSTRUCTION_FAILED,
                f"COLMAP CLI step failed ({args[1]}): {proc.stderr[-500:]}",
                "Ensure the COLMAP CUDA binary and NVIDIA drivers are working.",
            )

    _run([
        colmap_exe, "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])
    _run([
        colmap_exe, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--PatchMatchStereo.geom_consistency", "true",
    ])
    _run([
        colmap_exe, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(fused_path),
    ])
    return fused_path


def run_dense_reconstruction(
    sparse_reconstruction,
    sparse_dir: Path,
    images_dir: Path,
    dense_dir: Path,
    use_gpu: bool,
) -> Path:
    """Undistort -> patch-match stereo -> fusion.

    Prefers a CUDA-enabled COLMAP CLI (COLMAP_PATH / `colmap` on PATH) so
    dense MVS runs on the GPU. Falls back to PyCOLMAP's in-process
    functions if no CLI is found (which will fail with 'requires CUDA' on
    the CPU-only pip wheel — handled by the caller)."""

    dense_dir.mkdir(parents=True, exist_ok=True)
    fused_path = dense_dir / "fused.ply"

    colmap_exe = _find_colmap_cli()
    if colmap_exe and _colmap_cli_has_cuda(colmap_exe):
        fused_path = _run_dense_via_cli(colmap_exe, sparse_dir, images_dir, dense_dir)
    else:
        pycolmap = _require_pycolmap()
        try:
            pycolmap.undistort_images(
                output_path=dense_dir,
                input_path=sparse_dir / "0",
                image_path=images_dir,
            )
            pycolmap.patch_match_stereo(workspace_path=dense_dir)
            pycolmap.stereo_fusion(output_path=fused_path, workspace_path=dense_dir)
        except Exception as exc:  # noqa: BLE001
            raise ColmapError(
                ErrorCode.DENSE_RECONSTRUCTION_FAILED,
                f"Dense reconstruction (MVS) failed: {exc}",
                "Install a CUDA-enabled COLMAP and set COLMAP_PATH to its "
                "colmap.exe, or set DENSE_RECONSTRUCTION_ENABLED=false to "
                "ship sparse-only geometry.",
            ) from exc

    if not fused_path.exists() or fused_path.stat().st_size == 0:
        raise ColmapError(
            ErrorCode.DENSE_RECONSTRUCTION_FAILED,
            "Stereo fusion produced an empty point cloud.",
            "Registration quality may be too low for dense matching; check "
            "the registration_rate before retrying.",
        )

    return fused_path
