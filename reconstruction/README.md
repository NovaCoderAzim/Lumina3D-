# reconstruction/

**Person 3's module for DRISHTI-3D.** Turns selected drone frames into a
browser-ready `model.glb`. This is the only package in the project that
knows COLMAP exists.

## Quickstart

```bash
pip install -r reconstruction/requirements.txt

# COLMAP itself is NOT a pip package - PyCOLMAP bundles a working COLMAP,
# so no separate system install is required for the SfM/MVS steps used
# here. A CUDA-capable GPU is required for dense reconstruction
# (patch-match stereo); everything else runs on CPU, just slower.
```

```python
from reconstruction.interface import ReconstructionEngine
from reconstruction.schemas import ReconstructionConfig

engine = ReconstructionEngine(ReconstructionConfig())
result = engine.process(
    frames_directory="data/projects/demo_001/frames/selected",
    project_id="demo_001",
)
print(result.status, result.model)
```

This is the **entire** public API. Nothing else in this package should be
imported from outside `reconstruction/`.

## Mock mode (for Person 5 / Person 1, no COLMAP needed)

```bash
export USE_MOCK_RECONSTRUCTION=true
```

Returns a real, loadable demo `model.glb` (a simple colored icosphere) with
the exact same `ReconstructionResult` schema the real pipeline produces, in
milliseconds. Use this to build the viewer and API layer in parallel while
the real pipeline is still being tuned.

## Configuration

All via environment variables (see `schemas.ReconstructionConfig`):

| Variable | Default | Meaning |
|---|---|---|
| `COLMAP_PATH` | `colmap` | Reserved for a future CLI fallback; unused while PyCOLMAP covers every phase |
| `USE_GPU` | `true` | Auto-verified via `nvidia-smi`; silently falls back to CPU/sparse-only if no GPU |
| `USE_MOCK_RECONSTRUCTION` | `false` | Skip COLMAP entirely, return a demo GLB |
| `MAX_IMAGE_SIZE` | `2000` | Downscale cap fed to SIFT extraction |
| `MATCHER_TYPE` | `sequential` | `exhaustive` for looping/non-linear flight paths |
| `MIN_REGISTERED_RATIO` | `0.7` | Below this, stop before dense reconstruction |
| `DENSE_RECONSTRUCTION_ENABLED` | `true` | Kill switch for time-pressured demos |
| `MESH_RECONSTRUCTION_ENABLED` | `true` | Kill switch; without it you get poses + point cloud only |

## Pipeline

```
frames/selected/
      |
      v
validate_input_images          (validation.py)
      |
      v
feature extraction              (colmap_runner.py, PyCOLMAP SIFT)
      |
      v
feature matching                 (colmap_runner.py, sequential/exhaustive)
      |
      v
Structure from Motion             (colmap_runner.py, incremental_mapping)
      |
      v
registration rate check  --------> below MIN_REGISTERED_RATIO -> FAILED, stop here
      |
      v
camera poses + sparse cloud export
      |
      v
dense reconstruction (needs CUDA) (colmap_runner.py: undistort -> patch-match -> fusion)
      |                             (skipped with a WARNING if no GPU - falls back to sparse cloud)
      v
point cloud cleanup                (pointcloud.py, Open3D outlier removal)
      |
      v
mesh generation                    (mesh.py, Poisson -> Ball Pivoting fallback)
      |
      v
GLB conversion + axis fix          (conversion.py)
      |
      v
GLB validation                     (validation.py) --> never report COMPLETED if this fails
      |
      v
reconstruction_metadata.json + ReconstructionResult
```

## Coordinate system note

COLMAP/OpenCV convention: right-handed, X-right, **Y-down**, Z-forward.
glTF convention: right-handed, X-right, **Y-up**, Z-backward. The fixed
conversion is `(x, y, z) -> (x, -y, -z)` (180-degree rotation about X),
applied to both the mesh and every camera pose in `conversion.py`. Person
5's viewer receives everything already in glTF convention - no client-side
transform needed.

This does **not** solve absolute "which way is up" - SfM has no gravity
reference without IMU/GPS priors, so a reconstruction can still come out
tilted relative to a real-world horizon. Out of scope for the 48-hour
build; flagged here rather than silently ignored.

## Output structure

```
data/projects/{project_id}/reconstruction/
├── database.db
├── sparse/0/                 # COLMAP binary reconstruction
├── dense/                    # undistorted images + depth maps (if dense ran)
├── cloud_sparse.ply
├── cloud.ply                 # cleaned dense cloud (or sparse, if dense was skipped)
├── mesh.ply
├── model.glb                 # <- what Person 5 loads
├── cameras.json              # CameraPose[] in glTF-convention world space
└── reconstruction_metadata.json
```

## Testing

```bash
pytest reconstruction/test_reconstruction.py -v
```

Covers the three required datasets (too-few-images, poor-overlap,
schema-parity between mock and real). A true "good overlap" pass requires
a real photo set with actual parallax - point `tmp_frames` at a sample
dataset from Person 2 once one exists, or drop your own test images in and
extend `test_dataset_a_schema_contract_via_mock`.

## What's NOT implemented yet (next steps)

- COLMAP CLI fallback for environments where PyCOLMAP's dense functions
  aren't available (current build targets PyCOLMAP >=4.1, which covers
  undistortion, patch-match stereo, and fusion natively).
- Full UV texturing (currently: per-vertex color baked from the dense
  cloud - see `mesh.py`/`conversion.py` docstrings for why this was the
  right tradeoff for 48 hours).
- Progress callback plumbing for Person 1 to show live pipeline stage in
  the UI (`incremental_mapping` accepts callbacks - not yet wired up).
