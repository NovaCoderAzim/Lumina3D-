"""
DRISHTI-3D backend entry point (Person 1 — Tech Lead / Integration).

Run:
    uvicorn backend.main:app --reload --port 8000

The app is a modular monolith: one FastAPI process orchestrating the
vision / reconstruction / ai / analytics modules behind clean interfaces
(guide sections 2, 3, 8). Every module can run mocked so the whole thing
works on a laptop with no GPU (guide section 7).
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
    logger.info("DRISHTI-3D backend starting. Mock config: %s", settings.as_dict())
    yield


app = FastAPI(
    title="DRISHTI-3D Backend",
    version="0.1.0",
    description="Single-pass drone video -> 3D digital twin. Person 1 integration layer.",
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
    return {"status": "ok", "mocks": settings.as_dict()}


@app.get("/")
def root() -> dict:
    return {
        "app": "DRISHTI-3D",
        "docs": "/docs",
        "health": "/api/health",
    }
