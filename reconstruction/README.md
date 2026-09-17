# Lumina3D — 3D Reconstruction Engine

Photogrammetric reconstruction and 3D Gaussian Splatting engine for **Lumina3D**. Transforms drone video frames and aerial image sets into high-fidelity 3D digital twins, dense point clouds, textured meshes, and real-time radiance fields.

## Features

- **Structure from Motion (SfM)**: PyCOLMAP / COLMAP incremental SfM with SIFT feature extraction and robust geometric verification.
- **Dense MVS Reconstruction**: High-density point cloud generation with OpenMVS / COLMAP patch-match stereo.
- **Mesh Generation & Texturing**: Screened Poisson Surface Reconstruction, Ball-Pivoting Algorithm (BPA), and multi-view UV texture mapping.
- **3D Gaussian Splatting (3DGS)**: Real-time neural radiance field rendering with `.ply` export.
- **Metric Georeferencing**: Closed-form Umeyama similarity alignment against flight GPS / telemetry tracks without requiring ground control points (GCPs).
- **Dynamic Masking**: Automated YOLO-based semantic exclusion of moving vehicles, humans, and animals during feature matching.

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from reconstruction.interface import ReconstructionEngine
from reconstruction.schemas import ReconstructionConfig

engine = ReconstructionEngine(ReconstructionConfig())
result = engine.process(
    frames_directory="data/projects/proj_001/frames/selected",
    project_id="proj_001",
)
print(result.status, result.model)
```

## Mock Mode (Offline Development)

```bash
export USE_MOCK_RECONSTRUCTION=true
```

Generates schema-identical models and camera trajectories for testing viewers and API workflows without requiring a GPU.

## Configuration

Environment variables (see `schemas.ReconstructionConfig`):

| Variable | Default | Meaning |
|---|---|---|
| `USE_GPU` | `true` | Auto-verified via CUDA; gracefully falls back to CPU if unavailable |
| `USE_MOCK_RECONSTRUCTION` | `false` | Skip reconstruction, return synthetic test assets |
| `MAX_IMAGE_SIZE` | `2000` | Resolution cap for SIFT feature extraction |
| `MATCHER_TYPE` | `sequential` | `sequential` for linear drone sweeps, `exhaustive` for orbits |
| `MIN_REGISTERED_RATIO` | `0.7` | Minimum registration threshold required before dense reconstruction |
| `DENSE_RECONSTRUCTION_ENABLED` | `true` | Toggle dense point cloud generation |
| `MESH_RECONSTRUCTION_ENABLED` | `true` | Toggle Poisson / BPA surface meshing |

## Pipeline Architecture

```
frames/selected/
      │
      ▼
validate_input_images          (validation.py)
      │
      ▼
feature extraction              (PyCOLMAP SIFT)
      │
      ▼
feature matching                (sequential / spatial matcher)
      │
      ▼
Structure from Motion           (incremental SfM mapping)
      │
      ▼
registration rate check --------> below MIN_REGISTERED_RATIO -> early stop
      │
      ▼
camera poses + sparse cloud export
      │
      ▼
dense reconstruction (CUDA)    (patch-match stereo / OpenMVS dense)
      │
      ▼
point cloud filtering          (statistical outlier removal & normals)
      │
      ▼
surface mesh generation        (Poisson / Ball Pivoting)
      │
      ▼
texture baking & GLB export    (UV unwrap & GLB conversion)
      │
      ▼
3D Gaussian Splat generation   (splat.ply export)
      │
      ▼
reconstruction_metadata.json + ReconstructionResult
```

## Coordinate System Conventions

- **COLMAP / Computer Vision**: Right-handed, X-right, Y-down, Z-forward.
- **glTF / Three.js**: Right-handed, X-right, Y-up, Z-backward.
- The pipeline applies a standard 180° rotation about X `(x, y, z) -> (x, -y, -z)` during GLB conversion so output assets load seamlessly in Three.js and web viewers.

## Output Directory Layout

```
data/projects/{project_id}/reconstruction/
├── database.db
├── sparse/0/                 # COLMAP binary reconstruction
├── dense/                    # undistorted images + depth maps
├── cloud_sparse.ply          # sparse SfM point cloud
├── cloud.ply                 # filtered point cloud
├── mesh.ply                  # reconstructed surface mesh
├── model.glb                 # web-ready digital twin model
├── model_textured.glb        # high-resolution textured asset
├── splat.ply                 # 3D Gaussian Splatting radiance field
├── cameras.json              # camera poses in glTF coordinates
└── reconstruction_metadata.json
```

## Running Tests

```bash
pytest reconstruction/test_reconstruction.py -v
```
