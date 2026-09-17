"""
API routes (guide section 5). Mounted under /api to match the frontend's
default VITE_API_BASE_URL of '/api'.

Endpoints:
    POST /api/projects
    POST /api/projects/{id}/upload
    POST /api/projects/{id}/process
    GET  /api/projects/{id}/status
    GET  /api/projects/{id}/results
    GET  /api/projects/{id}/model        (metadata)
    GET  /api/projects/{id}/model/file   (the .glb binary)
    GET  /api/projects/{id}/semantic      (frontend name) == /objects
    GET  /api/projects/{id}/objects
    GET  /api/projects/{id}/analytics
    GET  /api/projects/{id}/report
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from ..config import get_settings
from ..models import (
    AnalyticsResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ModelInfoResponse,
    ProjectStatus,
    ProjectStatusResponse,
    ResultsResponse,
    SemanticObject,
)
from ..orchestration import orchestrator
from ..orchestration.state import STORE, Project
from ..utils import get_logger

logger = get_logger("backend.api")
router = APIRouter(prefix="/api")


def _require(project_id: str) -> Project:
    project = STORE.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


@router.get("/projects")
def list_projects() -> JSONResponse:
    projects = STORE.list_all()
    return JSONResponse({"projects": projects})


@router.post("/projects", response_model=CreateProjectResponse)
def create_project(body: CreateProjectRequest) -> CreateProjectResponse:
    project = STORE.create(body.name)
    logger.info("Created project %s (%s)", project.project_id, project.name)
    return CreateProjectResponse(project_id=project.project_id)


@router.post("/projects/{project_id}/upload")
async def upload_video(
    project_id: str,
    background: BackgroundTasks,
    video: UploadFile = File(...),
) -> JSONResponse:
    project = _require(project_id)
    settings = get_settings()

    project.transition(ProjectStatus.UPLOADING)
    STORE.save(project)

    input_dir = settings.project_input_dir(project_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(video.filename).name if video.filename else "input.mp4"
    filename = raw_name or "input.mp4"
    dest = input_dir / filename
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    project.video_filename = filename
    STORE.save(project)
    logger.info("Uploaded %s for project %s", filename, project_id)

    # Auto-start processing so the frontend's upload->processing flow works
    # with a single call (it does not issue a separate /process request).
    background.add_task(orchestrator.process_project, project_id)

    return JSONResponse({"project_id": project_id, "status": project.status.value})


@router.post("/projects/{project_id}/process")
def start_processing(project_id: str, background: BackgroundTasks) -> JSONResponse:
    project = _require(project_id)
    if project.status not in {ProjectStatus.UPLOADING, ProjectStatus.CREATED}:
        # Allow re-processing only from an uploadable state.
        raise HTTPException(
            status_code=409,
            detail=f"Project is {project.status.value}; cannot start processing",
        )
    background.add_task(orchestrator.process_project, project_id)
    return JSONResponse({"project_id": project_id, "status": "PROCESSING"})


@router.get("/projects/{project_id}/status", response_model=ProjectStatusResponse)
def get_status(project_id: str) -> ProjectStatusResponse:
    project = _require(project_id)
    return ProjectStatusResponse(**project.to_status_payload())


@router.get("/projects/{project_id}/model", response_model=ModelInfoResponse)
def get_model_info(project_id: str) -> ModelInfoResponse:
    project = _require(project_id)
    available = bool(project.model_path and Path(project.model_path).exists())
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    
    textured_path = recon_dir / "model_textured.glb"
    if not textured_path.exists():
        textured_path = recon_dir / "exports" / "textured_model.glb"
    pointcloud_path = recon_dir / "pointcloud.glb"
    splat_path = recon_dir / "exports" / "splat.ply"
    if not splat_path.exists():
        splat_path = recon_dir / "splat.ply"
    
    return ModelInfoResponse(
        project_id=project_id,
        model_url=f"/api/projects/{project_id}/model/file" if available else None,
        textured_model_url=f"/api/projects/{project_id}/model/textured" if (textured_path and textured_path.exists()) else None,
        point_cloud_url=f"/api/projects/{project_id}/pointcloud/file" if pointcloud_path.exists() else None,
        splat_url=f"/api/projects/{project_id}/splat/file" if (splat_path and splat_path.exists()) else None,
        available=available,
    )


@router.api_route("/projects/{project_id}/model/file", methods=["GET", "HEAD"])
def get_model_file(project_id: str) -> FileResponse:
    project = _require(project_id)
    if not project.model_path or not Path(project.model_path).exists():
        raise HTTPException(status_code=404, detail="Model not available yet")
    return FileResponse(
        project.model_path,
        media_type="model/gltf-binary",
        filename="model.glb",
    )


@router.api_route("/projects/{project_id}/model/textured", methods=["GET", "HEAD"])
def get_textured_model_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    textured_path = recon_dir / "model_textured.glb"
    if not textured_path.exists():
        textured_path = recon_dir / "exports" / "textured_model.glb"
    if not textured_path.exists():
        raise HTTPException(status_code=404, detail="Textured model not available for this project")
    return FileResponse(
        str(textured_path),
        media_type="model/gltf-binary",
        filename="model_textured.glb",
    )


@router.api_route("/projects/{project_id}/splat/file", methods=["GET", "HEAD"])
def get_splat_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    splat_path = recon_dir / "exports" / "splat.ply"
    if not splat_path.exists():
        splat_path = recon_dir / "splat.ply"
    if not splat_path.exists():
        raise HTTPException(status_code=404, detail="3D Gaussian Splats not available for this project")
    return FileResponse(
        str(splat_path),
        media_type="application/octet-stream",
        filename="splat.ply",
    )


@router.api_route("/projects/{project_id}/mesh/file", methods=["GET", "HEAD"])
def get_mesh_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    mesh_ply = recon_dir / "mesh.ply"
    if mesh_ply.exists():
        return FileResponse(str(mesh_ply), media_type="application/octet-stream", filename="mesh.ply")
    raise HTTPException(status_code=404, detail="Surface mesh not available")


@router.api_route("/projects/{project_id}/dense/file", methods=["GET", "HEAD"])
def get_dense_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    dense_ply = recon_dir / "dense" / "cloud_dense.ply"
    if dense_ply.exists():
        return FileResponse(str(dense_ply), media_type="application/octet-stream", filename="cloud_dense.ply")
    cloud_ply = recon_dir / "cloud.ply"
    if cloud_ply.exists():
        return FileResponse(str(cloud_ply), media_type="application/octet-stream", filename="cloud.ply")
    raise HTTPException(status_code=404, detail="Dense point cloud not available")


@router.api_route("/projects/{project_id}/pointcloud/file", methods=["GET", "HEAD"])
def get_pointcloud_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    pc_glb = recon_dir / "pointcloud.glb"
    if pc_glb.exists():
        return FileResponse(str(pc_glb), media_type="model/gltf-binary", filename="pointcloud.glb")
    cloud_ply = recon_dir / "cloud.ply"
    if cloud_ply.exists():
        return FileResponse(str(cloud_ply), media_type="application/octet-stream", filename="cloud.ply")
    raise HTTPException(status_code=404, detail="Point cloud not available")


@router.get("/projects/{project_id}/quality-report")
def get_quality_report(project_id: str) -> dict:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    report_file = recon_dir / "quality_report.json"
    if report_file.exists():
        return json.loads(report_file.read_text())
    raise HTTPException(status_code=404, detail="Quality report not available for this project")


@router.api_route("/projects/{project_id}/completion/file", methods=["GET", "HEAD"])
def get_completion_file(project_id: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    inf_ply = recon_dir / "completion" / "inferred_geometry.ply"
    if inf_ply.exists():
        return FileResponse(str(inf_ply), media_type="application/octet-stream", filename="inferred_geometry.ply")
    raise HTTPException(status_code=404, detail="Inferred completion geometry not available")


@router.api_route("/projects/{project_id}/exports/{export_name}", methods=["GET", "HEAD"])
def get_export_file(project_id: str, export_name: str) -> FileResponse:
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(project_id)
    export_path = recon_dir / "exports" / export_name
    if export_path.exists():
        media_type = "model/gltf-binary" if export_name.endswith(".glb") else "application/octet-stream"
        return FileResponse(str(export_path), media_type=media_type, filename=export_name)
    raise HTTPException(status_code=404, detail=f"Export {export_name} not available")


@router.get("/projects/{project_id}/objects", response_model=List[SemanticObject])
def get_objects(project_id: str) -> List[SemanticObject]:
    project = _require(project_id)
    return project.objects


# The frontend calls this "semantic" — alias to the same data.
@router.get("/projects/{project_id}/semantic", response_model=List[SemanticObject])
def get_semantic(project_id: str) -> List[SemanticObject]:
    return get_objects(project_id)


@router.get("/projects/{project_id}/analytics", response_model=AnalyticsResponse)
def get_analytics(project_id: str) -> AnalyticsResponse:
    project = _require(project_id)
    if project.analytics is None:
        raise HTTPException(
            status_code=409, detail="Analytics not available until processing completes"
        )
    return project.analytics


@router.get("/projects/{project_id}/capture")
def get_capture_quality(project_id: str) -> JSONResponse:
    """Fast capture-quality report (challenges ii, iii, vi). Available as
    soon as the analyzer runs at the start of processing."""
    project = _require(project_id)
    if project.capture_quality is None:
        return JSONResponse({"ready": False, "status": "analyzing"})
    return JSONResponse(project.capture_quality)


@router.get("/projects/{project_id}/results", response_model=ResultsResponse)
def get_results(project_id: str) -> ResultsResponse:
    project = _require(project_id)
    model_url = None
    if project.model_path and Path(project.model_path).exists():
        model_url = f"/api/projects/{project_id}/model/file"
    from ..models import CaptureQuality
    cq = None
    if project.capture_quality:
        try:
            cq = CaptureQuality(**{k: project.capture_quality.get(k) for k in (
                "verdict", "score", "sharpness_score", "exposure_consistency",
                "overlap_score", "motion_pattern", "coverage_estimate_deg",
                "recommendations", "processing_time_s") if k in project.capture_quality})
        except Exception:  # noqa: BLE001
            cq = None
    return ResultsResponse(
        project_id=project_id,
        status=project.status,
        model_url=model_url,
        analytics=project.analytics,
        objects=project.objects,
        capture_quality=cq,
        error=project.error,
    )


@router.get("/projects/{project_id}/report")
def get_report(project_id: str) -> JSONResponse:
    """Lightweight JSON report (guide section 4/5). A PDF generator is
    Person 6's job; this always-available JSON keeps the endpoint honest."""
    project = _require(project_id)
    return JSONResponse(
        {
            "project_id": project_id,
            "name": project.name,
            "status": project.status.value,
            "created_at": project.created_at,
            "analytics": project.analytics.model_dump() if project.analytics else None,
            "objects": [o.model_dump(by_alias=True) for o in project.objects],
            "error": project.error.model_dump() if project.error else None,
        }
    )
