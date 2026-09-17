"""
Pydantic schemas for the DRISHTI-3D reconstruction module.

These are the ONLY objects the rest of the system (Person 1's backend,
Person 5's viewer, Person 6's stats) should ever need to know about.
Nothing outside this package should import pycolmap, open3d, or trimesh
directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class GeometrySource(str, Enum):
    SPARSE_SFM = "SPARSE_SFM"
    DENSE_MVS = "DENSE_MVS"
    NEURAL_DENSE = "NEURAL_DENSE"
    NONE = "NONE"


class DenseEngineMode(str, Enum):
    AUTO = "auto"
    COLMAP_MVS = "colmap_mvs"
    NEURAL_DENSE = "neural_dense"
    SPARSE_ONLY = "sparse_only"


class ReconstructionConfig(BaseSettings):
    """
    Loaded from environment variables (or a .env file). See README.md for
    a description of each field.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    colmap_path: str = Field(default="colmap", alias="COLMAP_PATH")
    use_gpu: bool = Field(default=False, alias="USE_GPU")
    use_mock_reconstruction: bool = Field(default=False, alias="USE_MOCK_RECONSTRUCTION")
    max_image_size: int = Field(default=2000, alias="MAX_IMAGE_SIZE")
    matcher_type: Literal["sequential", "exhaustive"] = Field(
        default="sequential", alias="MATCHER_TYPE"
    )
    min_registered_ratio: float = Field(default=0.7, alias="MIN_REGISTERED_RATIO")
    dense_engine: str = Field(default="auto", alias="DENSE_ENGINE")
    dense_reconstruction_enabled: bool = Field(
        default=True, alias="DENSE_RECONSTRUCTION_ENABLED"
    )
    dense_profile: str = Field(default="balanced", alias="DENSE_PROFILE")
    mesh_reconstruction_enabled: bool = Field(
        default=True, alias="MESH_RECONSTRUCTION_ENABLED"
    )
    # Challenge (iv): mask dynamic objects (vehicles/people/animals) out of
    # feature extraction so they don't corrupt the static-scene geometry.
    mask_dynamic_objects: bool = Field(
        default=False, alias="MASK_DYNAMIC_OBJECTS"
    )


# --------------------------------------------------------------------------
# Core value objects
# --------------------------------------------------------------------------

class CameraPose(BaseModel):
    """One registered camera. Position/rotation are in the reconstruction's
    world frame AFTER the COLMAP -> glTF (Y-up) axis conversion, so
    downstream consumers (Person 5) never need to transform these again."""

    image: str
    camera_id: int
    position: list[float] = Field(min_length=3, max_length=3)
    rotation: list[float] = Field(
        min_length=4, max_length=4, description="Quaternion [w, x, y, z]"
    )
    intrinsics: Optional[dict] = None


class GpuInfo(BaseModel):
    available: bool
    name: Optional[str] = None
    vram_mb: Optional[int] = None


class ReconstructionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ErrorCode(str, Enum):
    INVALID_IMAGES = "INVALID_IMAGES"
    TOO_FEW_IMAGES = "TOO_FEW_IMAGES"
    FEATURE_EXTRACTION_FAILED = "FEATURE_EXTRACTION_FAILED"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    INSUFFICIENT_REGISTRATION = "INSUFFICIENT_REGISTRATION"
    SFM_FAILED = "SFM_FAILED"
    DENSE_RECONSTRUCTION_FAILED = "DENSE_RECONSTRUCTION_FAILED"
    MESH_GENERATION_FAILED = "MESH_GENERATION_FAILED"
    INSUFFICIENT_DENSITY_FOR_SURFACE_MESH = "INSUFFICIENT_DENSITY_FOR_SURFACE_MESH"
    GLB_CONVERSION_FAILED = "GLB_CONVERSION_FAILED"
    GLB_VALIDATION_FAILED = "GLB_VALIDATION_FAILED"
    COLMAP_NOT_FOUND = "COLMAP_NOT_FOUND"
    UNKNOWN = "UNKNOWN"


class ReconstructionError(BaseModel):
    error_code: ErrorCode
    message: str
    recommendation: str


class ReconstructionResult(BaseModel):
    """The single contract every other team member's code depends on.
    Mock mode and the real pipeline MUST both produce this exact shape."""

    project_id: str
    status: ReconstructionStatus

    model: Optional[str] = None
    point_cloud: Optional[str] = None
    mesh: Optional[str] = None
    camera_poses: Optional[str] = None

    input_images: int
    registered_images: int
    registration_rate: float

    sparse_points: Optional[int] = None
    dense_points: Optional[int] = None
    mesh_vertices: Optional[int] = None
    mesh_faces: Optional[int] = None

    engine_name: str = "pycolmap_sfm"
    geometry_source: str = "SPARSE_SFM"
    dense_engine: str = "auto"
    surface_mesh_available: bool = False
    dense_mvs_available: bool = False
    texture_available: bool = False
    splat_available: bool = False
    dense_cloud: Optional[str] = None
    textured_model: Optional[str] = None
    texture_atlas: Optional[str] = None
    texture_resolution: Optional[str] = None
    texture_source: Optional[str] = None
    splat_url: Optional[str] = None

    completion_available: bool = False
    completion_method: Optional[str] = None
    completion_confidence: Optional[float] = None
    inferred_vertex_count: Optional[int] = 0
    inferred_face_count: Optional[int] = 0
    inferred_region_count: Optional[int] = 0
    observed_region_count: Optional[int] = 1
    observed_geometry: Optional[dict] = None
    refined_geometry: Optional[dict] = None
    inferred_geometry: Optional[dict] = None

    gps_available: bool = False
    metric_scale_available: bool = False
    georeferenced: bool = False
    telemetry_source: str = "none"
    reprojection_error: Optional[float] = None
    mean_track_length: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    processing_time_seconds: float
    gpu: GpuInfo

    error: Optional[ReconstructionError] = None
