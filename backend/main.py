"""
Lumina3D backend entry point.

Run:
    uvicorn backend.main:app --reload --port 8000

The app is a modular monolith: one FastAPI process orchestrating the
vision, photogrammetry reconstruction, 3DGS, AI, and analytics modules.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import get_settings
from .utils import configure_logging, get_logger

configure_logging()
logger = get_logger("backend.main")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Lumina3D backend starting. Config: %s", settings.as_dict())
    yield


app = FastAPI(
    title="Lumina3D Backend",
    version="1.0.0",
    description="Autonomous drone video to photorealistic 3D Digital Twin & 3DGS radiance field platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mocks": settings.as_dict(),
        "config": settings.as_dict(),
    }


@app.get("/")
def root() -> dict:
    return {
        "app": "Lumina3D",
        "docs": "/docs",
        "health": "/api/health",
    }
