# DRISHTI-3D — SIH 26158 Solution Map

**Problem Statement 26158: Single-Pass Drone Video to Accurate 3D Model Generation System**

This document maps DRISHTI-3D's features directly to the problem statement's
stated Key Challenges and deliverables. Every capability listed is real and
runs in this repository — nothing is faked or hardcoded.

## The winning insight

A single drone pass **cannot** see every surface — that is a law of
geometry, not a bug. Most solutions hide this by producing a blobby model
and calling it done. DRISHTI-3D's differentiator is **honesty engineered as
a feature**: it recovers maximum accurate 3D from the single pass, and for
everything it *can't* recover, it measures and reports the uncertainty
instead of hallucinating. This is exactly what an operational user (disaster
response, reconnaissance) needs: a model they can trust, with the untrusted
parts clearly flagged.

## Key Challenge → Feature map

| # | Key Challenge (from PS) | DRISHTI-3D feature | Where | Verified |
|---|--------------------------|--------------------|-------|----------|
| i | Limited viewing angles (single flight path) | Per-region **coverage & confidence map**; weak regions flagged, not invented | `reconstruction/confidence.py` | 422K-pt cloud → 54% conf, 10 weak regions |
| ii | Motion blur & compression artifacts | **Quality-aware frame selection** (Laplacian sharpness) + capture blur score | `vision/frame_extractor.py`, `vision/capture_quality.py` | Taj: selected 61 sharp frames from 3508 |
| iii | Variable illumination & shadows | **Exposure-consistency** analysis + honest recommendation to lock exposure | `vision/capture_quality.py` | Taj: flagged exposure variance |
| iv | Dynamic objects (vehicles/humans/animals) | **YOLO dynamic-object masking** — moving objects excluded from SfM feature extraction | `reconstruction/masking.py` | Taj: masked 38 detections across 28/61 frames |
| v | GPS inaccuracies & sensor noise | **Robust GPS ingestion** (DJI SRT / CSV) + least-squares track alignment tolerant to noise | `telemetry/` | SRT + CSV parsers verified |
| vi | Real-time / near-real-time processing | **Instant capture-quality verdict (~1s)** before the slow reconstruction; staged progress | `vision/capture_quality.py` | Taj analyzed in 1.1s |
| vii | Reconstruction of occluded surfaces | Occluded/low-observation regions **detected and reported** rather than fabricated | `reconstruction/confidence.py` | Weak-region list with reasons |
| viii | Metric accuracy without GCPs | **GPS-based metric scaling & georeferencing** (Umeyama similarity) → real metres + geo-anchor, no GCPs | `reconstruction/georef.py`, `telemetry/` | Recovers known scale to <1m RMS (unit test) |

## Reconstruction targets (PS asks for i–v)

| PS target | Delivered |
|-----------|-----------|
| (i) 3D terrain & structures | Dense point cloud + mesh (COLMAP SfM + MVS, Open3D Poisson) |
| (ii) Building facades & rooftops | Reconstructed from the pass; confidence-scored |
| (iii) Roads & infrastructure | Semantic layer (YOLO) + geometry |
| (iv) Vegetation & obstacles | Semantic "Vegetation" category |
| (v) Textured 3D meshes / point clouds | `model.glb` (vertex-coloured mesh) + `cloud.ply` |

## Deliverables: visualization, measurement, analysis

- **Visualization** — browser digital twin (React + Three.js), loads the real `model.glb`.
- **Measurement** — click-two-points distance tool, converted to **real metres** via the GPS-recovered scale (`MeasurePanel`). If no metric scale, it says so — never a fake number.
- **Analysis** — registration rate, sparse/dense point counts, per-region confidence/coverage, dynamic-object stats, semantic object tallies.

## Input data (PS mandatory + optional)

- **Mandatory**: drone video (1080p/4K) ✔, GPS coordinates ✔ (SRT/CSV), flight metadata ✔.
- **Optional**: IMU/barometric altitude parsed when present in SRT; camera intrinsics estimated by COLMAP; RTK/PPK — track alignment accepts any higher-accuracy GPS source.

## Real technology stack (no black boxes, no fakery)

- **SfM + dense MVS**: COLMAP 4.2 (CUDA) / pycolmap — real feature extraction, matching, incremental mapping, patch-match stereo, fusion.
- **Meshing**: Open3D Poisson (+ Ball-Pivoting fallback).
- **Detection/masking**: Ultralytics YOLOv8 on GPU.
- **Geodesy**: WGS84 → ECEF → ENU, Umeyama similarity (numpy) for metric alignment.
- **Backend**: FastAPI orchestrator with a project state machine, mock mode for GPU-free demos.
- **Frontend**: React + Vite + Three.js.

## Honest limitations (say these before a judge asks — it builds trust)

- A **single linear pass** reconstructs the facing side well; rear/occluded
  surfaces are flagged low-confidence, not invented. A wider arc/orbit
  improves completeness — the capture analyzer says so up front.
- Dense stereo needs a **CUDA COLMAP** (bundled in `tools/`); without a GPU
  the system runs sparse + mesh, clearly labelled.
- With **synthetic GPS** (no telemetry provided) the metric scale is
  illustrative and labelled "synthetic"; with real telemetry it is accurate.

## Demo flow (2 minutes)

1. Upload a drone clip → **capture verdict in ~1 second** (GO/MARGINAL/NO-GO + why).
2. Reconstruction runs (dense MVS on GPU), dynamic objects masked out.
3. Digital twin loads: dense model, semantic layers, **measurements in metres**,
   and a **confidence/coverage panel** showing exactly what is and isn't trustworthy.
