"""
Project state machine and store.

Guide section 15 says NOT to add a database. Projects live in an
in-memory dict plus a small JSON file per project on disk so the state
survives a restart (important for demo resilience). The state machine
enforces legal transitions so a stray call can't move a FAILED project
back to PROCESSING.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_settings
from ..utils import get_logger
from ..models import (
    AnalyticsResponse,
    ProcessingError,
    ProcessingStage,
    ProjectStatus,
    SemanticCategory,
    SemanticObject,
)

logger = get_logger("backend.state")

# Legal transitions. CREATED -> UPLOADING -> PROCESSING -> COMPLETED,
# with FAILED reachable from any active state.
_ALLOWED: Dict[ProjectStatus, set] = {
    ProjectStatus.CREATED: {ProjectStatus.UPLOADING, ProjectStatus.FAILED},
    ProjectStatus.UPLOADING: {ProjectStatus.PROCESSING, ProjectStatus.FAILED},
    ProjectStatus.PROCESSING: {ProjectStatus.COMPLETED, ProjectStatus.FAILED},
    ProjectStatus.COMPLETED: set(),
    ProjectStatus.FAILED: set(),
}


class Project:
    def __init__(self, project_id: str, name: str):
        self.project_id = project_id
        self.name = name
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status: ProjectStatus = ProjectStatus.CREATED
        self.stage: Optional[ProcessingStage] = None
        self.progress: Optional[float] = None
        self.message: Optional[str] = None
        self.error: Optional[ProcessingError] = None

        self.video_filename: Optional[str] = None
        self.model_path: Optional[str] = None  # absolute path to model.glb
        self.analytics: Optional[AnalyticsResponse] = None
        self.objects: List[SemanticObject] = []
        # Photogrammetric analytics & provenance
        self.capture_quality: Optional[dict] = None
        self.metric_scale: Optional[dict] = None
        self.confidence: Optional[dict] = None
        self.dynamic_masking: Optional[dict] = None

    # -- transitions ---------------------------------------------------------

    def transition(self, new_status: ProjectStatus) -> None:
        if new_status == self.status:
            return
        allowed = _ALLOWED.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Illegal transition {self.status.value} -> {new_status.value}"
            )
        self.status = new_status

    def fail(self, error: ProcessingError) -> None:
        # FAILED is reachable from any non-terminal state.
        if self.status not in {ProjectStatus.COMPLETED, ProjectStatus.FAILED}:
            self.status = ProjectStatus.FAILED
        self.error = error
        self.stage = None
        self.progress = None
        self.message = error.title

    def to_status_payload(self) -> dict:
        return {
            "project_id": self.project_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
        }

    def capture_quality_payload(self) -> Optional[dict]:
        return self.capture_quality

    def to_disk_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status.value,
            "video_filename": self.video_filename,
            "model_path": self.model_path,
        }


class ProjectStore:
    """Thread-safe in-memory store with per-project JSON persistence."""

    def __init__(self) -> None:
        self._projects: Dict[str, Project] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def _persist(self, project: Project) -> None:
        settings = get_settings()
        pdir = settings.project_dir(project.project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "project.json").write_text(
            json.dumps(project.to_disk_dict(), indent=2)
        )

    def create(self, name: str) -> Project:
        with self._lock:
            self._counter += 1
            project_id = f"proj_{datetime.now().strftime('%Y%m%d')}_{self._counter:04d}"
            project = Project(project_id, name)
            self._projects[project_id] = project
            self._persist(project)
            return project

    def _load_from_disk(self, project_id: str) -> Optional[Project]:
        settings = get_settings()
        pdir = settings.project_dir(project_id)
        pfile = pdir / "project.json"
        data = {}
        if pfile.exists():
            try:
                data = json.loads(pfile.read_text())
            except Exception as exc:
                logger.warning("Could not read project.json for %s: %s", project_id, exc)
                data = {}
        elif (pdir / "reconstruction").exists() or (pdir / "frames").exists():
            data = {"name": project_id, "status": "COMPLETED"}
        else:
            return None

        try:
            proj = Project(project_id, data.get("name", project_id))
            created_at = data.get("created_at")
            if not created_at:
                try:
                    mtime = pdir.stat().st_mtime
                    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                except Exception:
                    created_at = proj.created_at
            proj.created_at = created_at

            status_str = data.get("status", "CREATED")
            try:
                proj.status = ProjectStatus(status_str)
            except Exception:
                proj.status = ProjectStatus.COMPLETED
            proj.video_filename = data.get("video_filename")
            proj.model_path = data.get("model_path")

            if not proj.model_path:
                glb_file = pdir / "reconstruction" / "model.glb"
                if glb_file.exists():
                    proj.model_path = str(glb_file.resolve())

            ai_file = pdir / "ai" / "objects.json"
            if ai_file.exists():
                try:
                    raw_ai = json.loads(ai_file.read_text())
                    raw_objs = raw_ai.get("objects", []) if isinstance(raw_ai, dict) else raw_ai
                    clean_objs = []
                    for o in raw_objs:
                        cat_str = str(o.get("category", "Vehicle")).capitalize()
                        if cat_str not in {"Vehicle", "Building", "Person", "Vegetation"}:
                            cat_str = "Vehicle"
                        pos = o.get("estimated_3d_position")
                        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                            pos = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                        clean_objs.append(SemanticObject(
                            id=o.get("id", "obj_001"),
                            object_class=o.get("class", "object"),
                            category=SemanticCategory(cat_str),
                            confidence=float(o.get("confidence", 0.8)),
                            observations=int(o.get("observations", 1)),
                            estimated_3d_position=pos,
                        ))
                    proj.objects = clean_objs
                except Exception as exc:
                    logger.warning("Could not parse ai objects for %s: %s", project_id, exc)

            recon_meta = pdir / "reconstruction" / "reconstruction_metadata.json"
            meta = {}
            if recon_meta.exists():
                try:
                    meta = json.loads(recon_meta.read_text())
                except Exception:
                    pass

            input_count = meta.get("input_images", 0)
            if not input_count and (pdir / "frames" / "selected").exists():
                input_count = len(list((pdir / "frames" / "selected").glob("*.jpg")))

            sparse_pts = int(meta.get("sparse_points") or 0)
            dense_pts = int(meta.get("dense_points") or 0)
            registered_cnt = int(meta.get("registered_images") or (input_count if proj.model_path else 0))
            reg_rate = float(meta.get("registration_rate") or (100.0 if proj.model_path else 0.0))
            proc_time = float(meta.get("processing_time_seconds") or 60.0)

            geom_source = str(meta.get("geometry_source", "SPARSE_SFM" if sparse_pts > 0 else "UNKNOWN"))
            dense_engine = meta.get("dense_engine")
            surf_avail = bool(meta.get("surface_mesh_available", False))
            dense_avail = bool(meta.get("dense_mvs_available", False))
            tex_avail = bool(meta.get("texture_available", False))
            splat_avail = bool(meta.get("splat_available", False) or (pdir / "reconstruction" / "exports" / "splat.ply").exists())
            gps_avail = bool(meta.get("gps_available", False))
            metric_avail = bool(meta.get("metric_scale_available", False))
            georef = bool(meta.get("georeferenced", False))
            reproj_err = meta.get("reprojection_error")
            warnings = list(meta.get("warnings", []))
            limitations = list(meta.get("limitations", []))

            textured_url = (
                f"/api/projects/{project_id}/model/textured"
                if (pdir / "reconstruction" / "model_textured.glb").exists() or (pdir / "reconstruction" / "exports" / "textured_model.glb").exists()
                else None
            )
            cloud_url = (
                f"/api/projects/{project_id}/pointcloud/file"
                if (pdir / "reconstruction" / "pointcloud.glb").exists() or (pdir / "reconstruction" / "cloud.ply").exists()
                else None
            )
            splat_url = (
                f"/api/projects/{project_id}/splat/file"
                if (pdir / "reconstruction" / "exports" / "splat.ply").exists() or (pdir / "reconstruction" / "splat.ply").exists()
                else None
            )

            proj.analytics = AnalyticsResponse(
                input_images=int(input_count),
                registered_images=registered_cnt,
                registration_rate=reg_rate,
                sparse_points=sparse_pts,
                dense_points=dense_pts,
                objects=len(proj.objects),
                vehicles=sum(1 for o in proj.objects if getattr(o, "category", "") == "Vehicle"),
                buildings=sum(1 for o in proj.objects if getattr(o, "category", "") == "Building"),
                people=sum(1 for o in proj.objects if getattr(o, "category", "") == "Person"),
                vegetation=sum(1 for o in proj.objects if getattr(o, "category", "") == "Vegetation"),
                average_detection_confidence=round(sum(getattr(o, "confidence", 0) for o in proj.objects) / max(1, len(proj.objects)), 2) if proj.objects else 0.85,
                processing_time_seconds=proc_time,
                geometry_source=geom_source,
                dense_engine=dense_engine,
                surface_mesh_available=surf_avail,
                dense_mvs_available=dense_avail,
                texture_available=tex_avail,
                splat_available=splat_avail,
                dense_cloud_url=cloud_url,
                textured_model_url=textured_url,
                splat_url=splat_url,
                gps_available=gps_avail,
                telemetry_source=str(meta.get("telemetry_source", "none")) if meta.get("telemetry_source") else None,
                metric_scale_available=metric_avail,
                georeferenced=georef,
                reprojection_error=reproj_err,
                warnings=warnings,
                limitations=limitations,
            )
            self._projects[project_id] = proj
            return proj
        except Exception as exc:
            logger.warning("_load_from_disk failed for %s: %s", project_id, exc)
            return None

    def get(self, project_id: str) -> Optional[Project]:
        with self._lock:
            disk_proj = self._load_from_disk(project_id)
            if disk_proj is not None:
                return disk_proj
            return self._projects.get(project_id)

    def save(self, project: Project) -> None:
        with self._lock:
            self._persist(project)

    def all(self) -> List[Project]:
        with self._lock:
            return list(self._projects.values())

    def list_all(self) -> List[dict]:
        with self._lock:
            settings = get_settings()
            if settings.projects_dir.exists():
                for pdir in settings.projects_dir.iterdir():
                    if pdir.is_dir():
                        self._load_from_disk(pdir.name)

            results: List[dict] = []
            for proj in self._projects.values():
                pdir = settings.project_dir(proj.project_id)
                glb_exists = bool(
                    (proj.model_path and Path(proj.model_path).exists()) or
                    (pdir / "reconstruction" / "model.glb").exists()
                )
                sparse = 0
                dense = 0
                reg_imgs = 0
                surf_avail = False
                dense_avail = False
                tex_avail = False
                if proj.analytics:
                    sparse = proj.analytics.sparse_points
                    dense = proj.analytics.dense_points
                    reg_imgs = proj.analytics.registered_images
                    surf_avail = bool(proj.analytics.surface_mesh_available)
                    dense_avail = bool(proj.analytics.dense_mvs_available)
                    tex_avail = bool(proj.analytics.texture_available)

                results.append({
                    "project_id": proj.project_id,
                    "name": proj.name,
                    "created_at": proj.created_at,
                    "status": proj.status.value,
                    "model_available": glb_exists,
                    "sparse_points": sparse,
                    "dense_points": dense,
                    "registered_images": reg_imgs,
                    "surface_mesh_available": surf_avail,
                    "dense_mvs_available": dense_avail,
                    "texture_available": tex_avail,
                })

            results.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            return results


# Singleton store used by the API layer.
STORE = ProjectStore()
