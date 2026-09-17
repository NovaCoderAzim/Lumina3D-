"""
Central configuration for the Lumina3D backend.

All tunables come from environment variables (optionally loaded from a
.env file). The mock flags are the heart of the integration strategy
(guide section 7): the whole pipeline must run end-to-end with every
real module mocked, so the frontend and API can be demonstrated on a
machine with no GPU, no COLMAP, and no model weights.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Load a .env file if python-dotenv is available; never a hard dependency.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Repository root = parent of the backend/ package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    """Runtime settings. Read once and cached via get_settings()."""

    def __init__(self) -> None:
        # ---- Mock flags (guide section 7) ----
        self.use_mock_vision: bool = _env_bool("USE_MOCK_VISION", True)
        self.use_mock_reconstruction: bool = _env_bool("USE_MOCK_RECONSTRUCTION", True)
        self.use_mock_ai: bool = _env_bool("USE_MOCK_AI", True)
        self.use_mock_analytics: bool = _env_bool("USE_MOCK_ANALYTICS", True)

        # ---- Reconstruction tuning (used only in real mode) ----
        # NOTE: the pip `pycolmap` wheel has no GPU SIFT (it needs an
        # OpenGL/CUDA build), so feature extraction must run on CPU. Default
        # USE_GPU to false so real reconstruction works out of the box.
        self.recon_use_gpu: bool = _env_bool("USE_GPU", False)
        self.recon_matcher_type: str = os.environ.get("MATCHER_TYPE", "exhaustive")
        self.recon_dense_enabled: bool = _env_bool("DENSE_RECONSTRUCTION_ENABLED", False)
        self.recon_dense_engine: str = os.environ.get("DENSE_ENGINE", "auto")
        self.recon_dense_profile: str = os.environ.get("DENSE_PROFILE", "balanced")
        self.mask_dynamic_objects: bool = _env_bool("MASK_DYNAMIC_OBJECTS", False)

        # Dense MVS uses a CUDA-enabled COLMAP CLI. Resolve COLMAP_PATH to an
        # absolute path (relative to the repo root) and re-export it so the
        # reconstruction module's CLI finder locates it regardless of CWD.
        colmap_path = os.environ.get("COLMAP_PATH", "")
        if colmap_path and colmap_path.lower() != "colmap":
            p = Path(colmap_path)
            if not p.is_absolute():
                p = REPO_ROOT / colmap_path
            if p.exists():
                os.environ["COLMAP_PATH"] = str(p)
                self.colmap_path = str(p)
            else:
                self.colmap_path = colmap_path
        else:
            self.colmap_path = colmap_path or "colmap"

        # ---- Paths ----
        self.repo_root: Path = REPO_ROOT
        self.data_dir: Path = REPO_ROOT / os.environ.get("DATA_DIR", "data")
        self.projects_dir: Path = self.data_dir / "projects"
        self.uploads_dirname: str = "input"
        self.frames_dirname: str = "frames"

        # A guaranteed-good demo dataset (guide section 13). If a real
        # reconstruction produces nothing, the orchestrator can fall back
        # to this so the app never shows an empty screen in a demo.
        self.demo_project_id: str = os.environ.get("DEMO_PROJECT_ID", "demo_001")

        # ---- Server ----
        self.host: str = os.environ.get("HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("PORT", "8000"))
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.environ.get(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if o.strip()
        ]

        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO")

    # -- convenience helpers -------------------------------------------------

    def project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def project_input_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / self.uploads_dirname

    def project_frames_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / self.frames_dirname

    def project_reconstruction_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "reconstruction"

    def as_dict(self) -> dict:
        return {
            "use_mock_vision": self.use_mock_vision,
            "use_mock_reconstruction": self.use_mock_reconstruction,
            "use_mock_ai": self.use_mock_ai,
            "use_mock_analytics": self.use_mock_analytics,
            "data_dir": str(self.data_dir),
            "demo_project_id": self.demo_project_id,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
