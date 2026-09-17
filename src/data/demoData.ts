import type { AnalyticsResponse, SemanticObject, Project } from '../types';

export const DEMO_PROJECT: Project = {
  project_id: 'demo_001',
  name: 'DEMO-001',
  created_at: new Date().toISOString(),
  model_url: null, // demo mode renders the procedural fallback scene instead of a .glb
};

export const DEMO_ANALYTICS: AnalyticsResponse = {
  input_images: 176,
  registered_images: 168,
  registration_rate: 95.45,
  sparse_points: 184392,
  dense_points: 2400000,
  objects: 93,
  vehicles: 12,
  buildings: 4,
  people: 7,
  vegetation: 16,
  average_detection_confidence: 0.87,
  processing_time_seconds: 642,
  metric_scale: {
    metres_per_unit: 2.418,
    rms_error_m: 0.62,
    is_synthetic_gps: false,
    anchor_lat: 27.1751,
    anchor_lon: 78.0421,
    anchor_alt: 168.0,
  },
  confidence: {
    overall_confidence: 0.88,
    coverage_score: 0.79,
    high_conf_fraction: 0.71,
    medium_conf_fraction: 0.2,
    low_conf_fraction: 0.09,
    weak_region_count: 3,
    notes: '71% of the model is high-confidence; 3 weak regions flagged as unreliable rather than hallucinated.',
  },
  dynamic_masking: {
    dynamic_detections: 42,
    images_masked: 61,
    total_images: 176,
    mean_masked_fraction: 0.048,
  },
};

// Positions correspond to the procedural DemoScene layout so that clicking
// a mesh and clicking its sidebar entry point at the same object.
export const DEMO_OBJECTS: SemanticObject[] = [
  { id: 'veh-01', class: 'Car', category: 'Vehicle', confidence: 0.94, observations: 18, estimated_3d_position: { x: -3.2, y: 0, z: 4.1 } },
  { id: 'veh-02', class: 'Car', category: 'Vehicle', confidence: 0.89, observations: 14, estimated_3d_position: { x: -1.6, y: 0, z: 4.1 } },
  { id: 'veh-03', class: 'Truck', category: 'Vehicle', confidence: 0.81, observations: 9, estimated_3d_position: { x: 4.4, y: 0, z: -3.0 } },
  { id: 'bld-01', class: 'Warehouse', category: 'Building', confidence: 0.97, observations: 42, estimated_3d_position: { x: 0, y: 0, z: -4 } },
  { id: 'bld-02', class: 'Office Block', category: 'Building', confidence: 0.93, observations: 37, estimated_3d_position: { x: 6, y: 0, z: -1 } },
  { id: 'veg-01', class: 'Tree Cluster', category: 'Vegetation', confidence: 0.88, observations: 21, estimated_3d_position: { x: 5.5, y: 0, z: 4.5 } },
  { id: 'veg-02', class: 'Tree Cluster', category: 'Vegetation', confidence: 0.85, observations: 16, estimated_3d_position: { x: -5.5, y: 0, z: -1.5 } },
  { id: 'per-01', class: 'Pedestrian', category: 'Person', confidence: 0.76, observations: 5, estimated_3d_position: null },
];
