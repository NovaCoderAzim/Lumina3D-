// ---- Project status schema (Section 18) ----
export type ProjectStatus =
  | 'CREATED'
  | 'UPLOADING'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'FAILED';

export type ProcessingStage =
  | 'VIDEO_ANALYSIS'
  | 'FRAME_SELECTION'
  | 'RECONSTRUCTION'
  | 'SEMANTIC_ANALYSIS'
  | 'FINALIZATION';

export type StageState = 'pending' | 'processing' | 'completed' | 'failed';

export interface ProjectStatusResponse {
  project_id: string;
  status: ProjectStatus;
  stage: ProcessingStage | null;
  progress: number | null; // usually null — stage-based, not %-based (Section 18)
  message?: string | null;
  error?: {
    title: string;
    detail: string;
    cause?: string;
    suggestion?: string;
  };
}

export const STAGE_ORDER: ProcessingStage[] = [
  'VIDEO_ANALYSIS',
  'FRAME_SELECTION',
  'RECONSTRUCTION',
  'SEMANTIC_ANALYSIS',
  'FINALIZATION',
];

export const STAGE_LABEL: Record<ProcessingStage, string> = {
  VIDEO_ANALYSIS: 'Video uploaded',
  FRAME_SELECTION: 'Frame extraction & quality analysis',
  RECONSTRUCTION: '3D reconstruction',
  SEMANTIC_ANALYSIS: 'Semantic analysis',
  FINALIZATION: 'Digital twin generation',
};

// ---- Semantic objects (Section 13) ----
export type SemanticCategory = 'Vehicle' | 'Building' | 'Person' | 'Vegetation';

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface SemanticObject {
  id: string;
  class: string;
  category: SemanticCategory;
  confidence: number; // 0..1
  observations: number;
  estimated_3d_position: Vec3 | null; // null → "3D Association: Unavailable"
}

// ---- Analytics schema (Section 19) ----
export interface MetricScale {
  metres_per_unit: number;
  rms_error_m: number;
  is_synthetic_gps: boolean;
  anchor_lat: number | null;
  anchor_lon: number | null;
  anchor_alt: number | null;
}

export interface ConfidenceSummary {
  overall_confidence: number;
  coverage_score: number;
  high_conf_fraction: number;
  medium_conf_fraction: number;
  low_conf_fraction: number;
  weak_region_count: number;
  notes: string;
}

export interface DynamicMaskingSummary {
  dynamic_detections: number;
  images_masked: number;
  total_images: number;
  mean_masked_fraction: number;
}

export interface CaptureQuality {
  verdict: 'GO' | 'MARGINAL' | 'NO_GO' | string;
  score: number;
  sharpness_score: number;
  exposure_consistency: number;
  overlap_score: number;
  motion_pattern: string;
  coverage_estimate_deg: number;
  recommendations: string[];
  processing_time_s?: number;
}

export interface AnalyticsResponse {
  input_images: number;
  registered_images: number;
  registration_rate: number; // percent, e.g. 95.45
  sparse_points: number;
  dense_points: number;
  objects: number;
  vehicles: number;
  buildings: number;
  people: number;
  vegetation: number;
  average_detection_confidence: number; // 0..1
  processing_time_seconds?: number;
  // Lumina3D photogrammetric metrics & scientific provenance
  metric_scale?: MetricScale | null;
  confidence?: ConfidenceSummary | null;
  dynamic_masking?: DynamicMaskingSummary | null;
  geometry_source?: string;
  dense_engine?: string;
  surface_mesh_available?: boolean;
  dense_mvs_available?: boolean;
  texture_available?: boolean;
  splat_available?: boolean;
  dense_cloud_url?: string | null;
  textured_model_url?: string | null;
  splat_url?: string | null;
  gps_available?: boolean;
  telemetry_source?: string;
  metric_scale_available?: boolean;
  georeferenced?: boolean;
  reprojection_error?: number | null;
  warnings?: string[];
  limitations?: string[];
}

export interface Project {
  project_id: string;
  name: string;
  created_at: string;
  model_url: string | null;
}

export interface ProjectSummary {
  project_id: string;
  name: string;
  created_at: string;
  status: ProjectStatus;
  model_available: boolean;
  sparse_points: number;
  dense_points: number;
  registered_images: number;
  surface_mesh_available: boolean;
  dense_mvs_available: boolean;
  texture_available: boolean;
}

export const CATEGORY_COLOR: Record<SemanticCategory, string> = {
  Vehicle: 'var(--cat-vehicle)',
  Building: 'var(--cat-building)',
  Person: 'var(--cat-person)',
  Vegetation: 'var(--cat-vegetation)',
};

export const CATEGORY_LAYER_LABEL: Record<SemanticCategory, string> = {
  Vehicle: 'Vehicles',
  Building: 'Buildings',
  Person: 'People',
  Vegetation: 'Vegetation',
};
