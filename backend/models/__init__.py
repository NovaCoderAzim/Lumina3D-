"""
Pydantic schemas — the API contract between the backend and the React
frontend (Person 5). Field names and enum values here are deliberately
aligned with src/types/index.ts so the frontend consumes responses
without any remapping.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# State machine enums (aligned with frontend src/types/index.ts)
# ---------------------------------------------------------------------------

class ProjectStatus(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingStage(str, Enum):
    VIDEO_ANALYSIS = "VIDEO_ANALYSIS"
    FRAME_SELECTION = "FRAME_SELECTION"
    RECONSTRUCTION = "RECONSTRUCTION"
    SEMANTIC_ANALYSIS = "SEMANTIC_ANALYSIS"
    FINALIZATION = "FINALIZATION"


class SemanticCategory(str, Enum):
    VEHICLE = "Vehicle"
    BUILDING = "Building"
    PERSON = "Person"
    VEGETATION = "Vegetation"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class CreateProjectResponse(BaseModel):
    project_id: str


class ProcessingError(BaseModel):
    """Human-readable error surfaced to the UI (guide section 11).
    The UI must never simply crash — it shows title/detail/suggestion."""

    title: str
    detail: str
    cause: Optional[str] = None
    suggestion: Optional[str] = None


class ProjectStatusResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    stage: Optional[ProcessingStage] = None
    progress: Optional[float] = None  # 0..100; usually stage-based (Section 18)
    message: Optional[str] = None
    error: Optional[ProcessingError] = None


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class SemanticObject(BaseModel):
    """Matches the frontend SemanticObject. `estimated_3d_position` stays
    null whenever a 3D association is not reliably known — never fabricated."""

    id: str
    object_class: str = Field(..., alias="class")
    category: SemanticCategory
    confidence: float  # 0..1
    observations: int
    estimated_3d_position: Optional[Vec3] = None

    model_config = {"populate_by_name": True}


class MetricScale(BaseModel):
    """Real-world scale + georeference recovered from GPS (challenge viii)."""
    metres_per_unit: float
    rms_error_m: float
    is_synthetic_gps: bool
    anchor_lat: Optional[float] = None
    anchor_lon: Optional[float] = None
    anchor_alt: Optional[float] = None


class ConfidenceSummary(BaseModel):
    """Per-region reconstruction trust (challenges i, vii)."""
    overall_confidence: float
    coverage_score: float
    high_conf_fraction: float
    medium_conf_fraction: float
    low_conf_fraction: float
    weak_region_count: int
    notes: str = ""


class DynamicMaskingSummary(BaseModel):
    """Moving objects removed from reconstruction (challenge iv)."""
    dynamic_detections: int
    images_masked: int
    total_images: int
    mean_masked_fraction: float


class CaptureQuality(BaseModel):
    """Fast pre-reconstruction go/no-go (challenges ii, iii, vi)."""
    verdict: str
    score: float
    sharpness_score: float
    exposure_consistency: float
    overlap_score: float
    motion_pattern: str
    coverage_estimate_deg: float
    recommendations: List[str] = Field(default_factory=list)
    processing_time_s: float = 0.0


class AnalyticsResponse(BaseModel):
    input_images: int
    registered_images: int
    registration_rate: float  # percent, e.g. 95.45
    sparse_points: int
    dense_points: int
    objects: int
    vehicles: int
    buildings: int
    people: int
    vegetation: int
    average_detection_confidence: float  # 0..1
    processing_time_seconds: Optional[float] = None
    # ---- SIH 26158 differentiators ----
    metric_scale: Optional[MetricScale] = None
    confidence: Optional[ConfidenceSummary] = None
    dynamic_masking: Optional[DynamicMaskingSummary] = None
    geometry_source: str = "SPARSE_SFM"
    dense_engine: Optional[str] = None
    surface_mesh_available: bool = False
    dense_mvs_available: bool = False
    texture_available: bool = False
    splat_available: bool = False
    dense_cloud_url: Optional[str] = None
    textured_model_url: Optional[str] = None
    splat_url: Optional[str] = None
    gps_available: bool = False
    telemetry_source: Optional[str] = None
    metric_scale_available: bool = False
    georeferenced: bool = False
    reprojection_error: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    """Returned by GET /model when the client asks for metadata (not the
    binary). The frontend also builds a direct download URL to /model/file."""

    project_id: str
    model_url: Optional[str] = None
    textured_model_url: Optional[str] = None
    point_cloud_url: Optional[str] = None
    splat_url: Optional[str] = None
    format: str = "glb"
    available: bool = False


class ResultsResponse(BaseModel):
    """Full aggregated result (guide section 4)."""

    project_id: str
    status: ProjectStatus
    model_url: Optional[str] = None
    analytics: Optional[AnalyticsResponse] = None
    objects: List[SemanticObject] = Field(default_factory=list)
    capture_quality: Optional[CaptureQuality] = None
    error: Optional[ProcessingError] = None
