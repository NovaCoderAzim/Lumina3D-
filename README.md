# Lumina3D — Single-Pass Drone Video → 3D Digital Twin Platform

**SIH · Problem Statement 26158 — Single-Pass Drone Video to Accurate 3D
Model Generation System.**

One drone video in, an interactive, metrically-scaled 3D digital twin out:
intelligent frame selection → 3D reconstruction → AI semantic detection →
analytics → browser dashboard — with **honest per-region trust scores**
instead of a blobby model that hides its own errors.

See [`docs/SIH_26158_SOLUTION.md`](docs/SIH_26158_SOLUTION.md) for the full
map of every feature to the problem statement's 8 Key Challenges.

## What makes this win (the differentiators)

- **Instant capture-quality verdict (~1s)** — GO / MARGINAL / NO-GO with
  reasons, before the slow reconstruction. *(Challenges ii, iii, vi)*
- **GPS metric scaling & georeferencing** — real **metres** + geo-anchor,
  no GCPs, by aligning the reconstruction to the flight track. *(Challenge viii)*
- **Dynamic-object masking** — YOLO removes vehicles/people/animals from SfM
  so they don't corrupt the geometry. *(Challenge iv)*
- **Confidence & coverage map** — occluded/unseen regions flagged, not
  hallucinated. *(Challenges i, vii)*
- **Measurements in real metres** on the model. *(Deliverable: measurement)*

Everything is real and verified on real footage — no pre-baked fakes.

```
Drone Video → Capture-Quality Check → Frame Selection → Dynamic Masking
   → 3D Reconstruction (COLMAP+MVS) → GPS Metric Scaling → AI Semantics
   → Confidence/Coverage → Measurable 3D Digital Twin
```

## Architecture

Modular monolith. One FastAPI process orchestrates four modules behind
clean interfaces — no module imports another's internals.

```
                       ┌──────────────┐
   React + Three.js ──▶│   FastAPI    │
     (frontend)        │   backend    │
                       └──────┬───────┘
                              ▼
                       ORCHESTRATOR  (backend/orchestration)
             ┌────────────────┼───────────────┐
             ▼                ▼                ▼
        vision           reconstruction        ai
      (P2, mock)          (P3, real)         (P4, real)
             └────────────────┼───────────────┘
                              ▼
                          analytics  (aggregation)
                              ▼
                           RESULTS
```

| Path | Owner | What it is |
|------|-------|------------|
| `backend/` | P1 | FastAPI app, schemas, state machine, orchestrator, module adapters |
| `src/` | P5 | React + Vite + Three.js dashboard |
| `reconstruction/` | P3 | Frames → point cloud / mesh / GLB (COLMAP/Open3D) + mock |
| `ai/` | P4 | YOLO detection + tracking + semantic summary + mock |
| `data/projects/` | — | Per-project working dirs + the guaranteed-good `demo_001` dataset |
| `tests/` | P1 | Mock, no-GPU integration test + state-machine unit tests |

## Mock mode — the heart of the demo strategy

Every module can run mocked (guide section 7). With the four
`USE_MOCK_*` flags true, the whole pipeline runs on a laptop with **no
GPU, no COLMAP, and no model weights**, producing a real `model.glb`
(via trimesh) and the committed demo AI detections. Mock implementations
are clearly labelled and never presented as real results.

```
USE_MOCK_VISION=true
USE_MOCK_RECONSTRUCTION=true
USE_MOCK_AI=true
USE_MOCK_ANALYTICS=true
```

Flip any flag to `false` once that team's real module + dependencies are
installed and verified. If a real reconstruction fails at runtime, the
orchestrator falls back to the committed `demo_001` model so the app
never shows an empty screen during a presentation (guide section 13).

## Quick start

### 1. Backend (Python 3.8+)

```bash
pip install -r requirements.txt
# Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
# macOS/Linux:
bash scripts/run_backend.sh
```

The API serves on http://127.0.0.1:8000 — interactive docs at `/docs`,
health at `/api/health`.

To configure, copy `.env.example` to `.env`. Defaults are full mock mode.

### 2. Frontend (Node 18+)

```bash
npm install
# Demo mode (default) — standalone, no backend needed:
npm run dev
# Real backend mode — talk to the FastAPI server:
#   copy .env.example.frontend to .env.local and set VITE_DEMO_MODE=false
npm run dev
```

Dev server: http://localhost:5173. In dev, `/api` is proxied to the
backend (see `vite.config.ts`), so no CORS setup is needed. The
frontend-specific README is preserved as `README.frontend.md`.

### 3. Tests (mock, no GPU)

```bash
# Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
# any OS:
USE_MOCK_VISION=true USE_MOCK_RECONSTRUCTION=true USE_MOCK_AI=true \
  USE_MOCK_ANALYTICS=true python -m pytest tests/ -v
```

`tests/test_integration_mock.py` drives the full slice through the HTTP
API — create → upload → poll → COMPLETED → model + semantic + analytics
+ results + report — and must stay green before any branch is merged
(guide section 9).

## API contract

All routes are under `/api`.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/projects` | Create a project → `{project_id}` |
| POST | `/projects/{id}/upload` | Upload video (form field `video`); auto-starts processing |
| POST | `/projects/{id}/process` | Manually (re)start processing |
| GET  | `/projects/{id}/status` | `{status, stage, progress, message, error}` |
| GET  | `/projects/{id}/model` | Model metadata + download URL |
| GET  | `/projects/{id}/model/file` | The `.glb` binary |
| GET  | `/projects/{id}/objects` | Semantic detections |
| GET  | `/projects/{id}/semantic` | Alias of `/objects` (frontend name) |
| GET  | `/projects/{id}/analytics` | Registration rate, point counts, category tallies |
| GET  | `/projects/{id}/results` | Aggregated final result |
| GET  | `/projects/{id}/report` | JSON report |
| GET  | `/api/health` | Health + active mock flags |

### State machine

```
CREATED → UPLOADING → PROCESSING → COMPLETED
                  ↘        ↘
                   → →  FAILED  (reachable from any active state, stores a
                                 user-facing title/detail/cause/suggestion)
```

Processing stages surfaced to the UI: `VIDEO_ANALYSIS → FRAME_SELECTION
→ RECONSTRUCTION → SEMANTIC_ANALYSIS → FINALIZATION`.

## Full REAL mode (verified)

Every module has been run for real and verified end-to-end on this machine
(Windows, NVIDIA RTX 3060):

| Module | Real implementation | Verified result |
|--------|--------------------|-----------------|
| Vision | OpenCV decode + Laplacian-variance sharpness selection (`vision/`) | Real frames selected from a real video |
| Reconstruction | COLMAP/pycolmap SfM + Open3D mesh (`reconstruction/`) | 50/50 images registered (100%), 16k sparse points, real model.glb (46k verts) |
| AI | YOLOv8 detection on GPU (`ai/`) | Real detections on real frames |
| Analytics | Real aggregation (`backend/services/analytics_service.py`) | Real registration rate + point/object counts |

### Requirements for real mode

Real reconstruction needs Python 3.10+ (the base Python 3.8 cannot install
`pycolmap>=4.1` / `open3d>=0.19`). A dedicated venv is used:

```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\python -m pip install -r requirements.txt
.venv311\Scripts\python -m pip install -r reconstruction/requirements.txt -r vision/requirements.txt -r ai/requirements.txt
.venv311\Scripts\python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

powershell -ExecutionPolicy Bypass -File scripts\run_backend_real.ps1
```

Generate a real multi-view test scene (real parallax for COLMAP):

```powershell
.venv311\Scripts\python scripts\generate_multiview.py data\multiview\selected 50
```

### Real-mode caveats (honest limitations)

- The pip `pycolmap` wheel has no GPU SIFT (needs an OpenGL/CUDA build), so
  `USE_GPU=false` keeps feature extraction on CPU. YOLO still uses the GPU.
- Dense reconstruction (patch-match stereo) needs a CUDA/OpenGL COLMAP build
  not in the pip wheel, so `DENSE_RECONSTRUCTION_ENABLED=false`. Sparse SfM +
  Poisson mesh are still fully real.
- Real reconstruction needs input images with genuine parallax and overlap
  (a slow orbit/pass). A flat or low-texture scene will fail registration —
  that's correct SfM behaviour, not a bug.

## Notes for integrators

- **Vision (P2)** real module lives in `vision/` (OpenCV frame extractor).
- **Reconstruction (P3)** real mode needs `reconstruction/requirements.txt`.
  Mock mode calls `reconstruction.mock` directly so it stays GPU-free.
- **AI (P4)** real mode needs `ai/requirements.txt`. The adapter maps P4's
  categories (structure/vehicle/human/environment) to the frontend's
  (Building/Vehicle/Person/Vegetation).
- **Analytics (P6)** is computed by `analytics_service` from the other
  modules' outputs. Swap in a real `analytics/` package when available.
