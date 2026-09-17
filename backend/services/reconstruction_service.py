"""
Reconstruction module adapter.

Wraps the `reconstruction` photogrammetry package:

    reconstruction.interface.ReconstructionEngine.process(
        frames_directory, project_id, output_dir
    ) -> reconstruction.schemas.ReconstructionResult

In mock mode, calls reconstruction.mock.run_mock_reconstruction directly (trimesh/numpy only).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

from ..config import get_settings
from ..models import ProcessingError
from ..utils import get_logger, log_module_event

logger = get_logger("backend.services.reconstruction")


class ReconstructionFailure(Exception):
    def __init__(self, error: ProcessingError):
        self.error = error
        super().__init__(error.detail)


def _normalise(result: Any, project_id: str, start: float, is_mock: bool) -> Dict[str, Any]:
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)

    if str(data.get("status")) == "FAILED":
        err = data.get("error") or {}
        raise ReconstructionFailure(
            ProcessingError(
                title="3D reconstruction failed",
                detail=err.get("message", "Reconstruction reported FAILED."),
                cause=str(err.get("error_code", "reconstruction_failed")),
                suggestion=err.get(
                    "recommendation",
                    "Upload a video with more overlapping views of the target.",
                ),
            )
        )

    elapsed = round(time.perf_counter() - start, 2)
    data["adapter_time_seconds"] = elapsed
    data["is_mock"] = is_mock
    log_module_event(
        logger,
        project_id=project_id,
        module="reconstruction" + ("(mock)" if is_mock else ""),
        status="COMPLETED",
        processing_time=elapsed,
    )
    return data


def process(
    frames_directory: str,
    project_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    start = time.perf_counter()

    output_dir = settings.project_reconstruction_dir(project_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mock mode: call the mock directly (trimesh/numpy only) ----
    if settings.use_mock_reconstruction:
        try:
            from reconstruction.mock import run_mock_reconstruction
        except Exception as exc:  # noqa: BLE001
            raise ReconstructionFailure(
                ProcessingError(
                    title="Reconstruction module unavailable",
                    detail=str(exc),
                    cause="import_error",
                    suggestion="Install trimesh/numpy, or check the reconstruction/ package.",
                )
            ) from exc
        try:
            result = run_mock_reconstruction(frames_directory, project_id, str(output_dir))
        except Exception as exc:  # noqa: BLE001
            raise ReconstructionFailure(
                ProcessingError(
                    title="3D reconstruction failed (mock)",
                    detail=str(exc),
                    cause="mock_exception",
                    suggestion="Verify trimesh/numpy are installed.",
                )
            ) from exc
        return _normalise(result, project_id, start, is_mock=True)

    # ---- Real mode: full engine (pulls in open3d/pycolmap) ----
    try:
        from reconstruction.interface import ReconstructionEngine
        from reconstruction.schemas import ReconstructionConfig
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionFailure(
            ProcessingError(
                title="Reconstruction module unavailable",
                detail=str(exc),
                cause="import_error",
                suggestion="Install reconstruction/requirements.txt or set "
                "USE_MOCK_RECONSTRUCTION=true.",
            )
        ) from exc

    config = ReconstructionConfig(
        use_mock_reconstruction=False,
        use_gpu=settings.recon_use_gpu,
        matcher_type=settings.recon_matcher_type,
        dense_reconstruction_enabled=getattr(settings, "recon_dense_enabled", True),
        dense_engine=getattr(settings, "recon_dense_engine", os.environ.get("DENSE_ENGINE", "auto")),
        dense_profile=getattr(settings, "recon_dense_profile", "balanced"),
        colmap_path=getattr(settings, "colmap_path", None),
        mask_dynamic_objects=settings.mask_dynamic_objects,
    )
    engine = ReconstructionEngine(config=config, projects_base_dir=str(settings.projects_dir))

    try:
        result = engine.process(
            frames_directory=frames_directory,
            project_id=project_id,
            output_dir=str(output_dir),
            progress_callback=progress_callback,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.perf_counter() - start, 2)
        log_module_event(
            logger,
            project_id=project_id,
            module="reconstruction",
            status="FAILED",
            processing_time=elapsed,
            error=str(exc),
        )
        raise ReconstructionFailure(
            ProcessingError(
                title="3D reconstruction failed",
                detail=str(exc),
                cause="reconstruction_exception",
                suggestion="Upload a video with more overlapping views of "
                "the target area, or run in mock mode for the demo.",
            )
        ) from exc

    return _normalise(result, project_id, start, is_mock=False)


def demo_reconstruction(project_id: str) -> Dict[str, Any]:
    """Emergency fallback: ONLY active if DEMO_MODE=true is explicitly set.
    Real projects must never silently fall back to demo reconstructions.
    """
    import os
    if os.environ.get("DEMO_MODE", "").lower() not in ("1", "true", "yes"):
        logger.info("Demo reconstruction fallback is DISABLED (DEMO_MODE!=true).")
        return {}
    settings = get_settings()
    recon_dir = settings.project_reconstruction_dir(settings.demo_project_id)
    model = recon_dir / "model.glb"
    if not model.exists():
        return {}
    return {
        "project_id": project_id,
        "status": "COMPLETED",
        "model": str(model),
        "point_cloud": str(recon_dir / "cloud.ply"),
        "camera_poses": str(recon_dir / "cameras.json"),
        "input_images": 176,
        "registered_images": 168,
        "registration_rate": 95.45,
        "sparse_points": 184392,
        "dense_points": 2400000,
        "is_mock": True,
        "is_demo_fallback": True,
    }
