# Lumina3D — System Architecture & Solution Specification

**Lumina3D: Autonomous Single-Pass Drone Video to Metric 3D Digital Twin & 3DGS Radiance Field Platform**

This document provides the complete architecture and technical capability map for Lumina3D. Every capability listed is fully operational in this repository.

---

## Core Engineering Philosophy

A single drone flight path cannot capture every occluded surface — that is a fundamental law of projective geometry. Many conventional systems hide this reality by interpolating noisy, blobby meshes. 

Lumina3D's engineering differentiator is **rigorous provenance and metric transparency**:
- Recovers maximum accurate 3D geometry from available video passes using state-of-the-art Structure-from-Motion (SfM) and Multi-View Stereo (MVS).
- Quantifies and maps photogrammetric confidence across every region of the reconstructed scene.
- Injects metric scale through GPS telemetry alignment without requiring manual Ground Control Points (GCPs).
- Masks dynamic moving entities (vehicles, pedestrians) to prevent epipolar corruption.
- Offers dual visualization: precise engineering geometry (textured mesh, dense point cloud) and photorealistic radiance fields (3D Gaussian Splatting).

---

## Technical Capability & Feature Map

| # | Technical Challenge | Lumina3D Solution | Implementation Module | Verification |
|---|---------------------|-------------------|-----------------------|--------------|
| 1 | **Single Flight Path Coverage** | Per-region coverage & confidence analysis; weak areas flagged transparently | `reconstruction/confidence.py` | Ray intersection uncertainty & density mapping |
| 2 | **Motion Blur & Frame Selection** | Laplacian variance sharpness filtering and adaptive stride extraction | `vision/frame_extractor.py`, `vision/capture_quality.py` | Rejects blurry frames, keeps optimal parallax |
| 3 | **Illumination Variation & Shadows** | Exposure variance scoring and illumination consistency checks | `vision/capture_quality.py` | Evaluates histogram drift across selected frames |
| 4 | **Dynamic Moving Objects** | YOLOv8 automated semantic masking to exclude dynamic pixels from SfM | `reconstruction/masking.py` | Generates COLMAP-compatible 8-bit exclusion masks |
| 5 | **Sensor Noise & GPS Drift** | DJI SRT / CSV telemetry parsing with Umeyama least-squares similarity | `telemetry/`, `reconstruction/georef.py` | Recovers scale factor and local ENU coordinates |
| 6 | **Rapid Operator Feedback** | Sub-second video capture diagnostic report before full reconstruction | `vision/capture_quality.py` | Immediate GO / MARGINAL / NO-GO verdict |
| 7 | **Occluded Geometry Inference** | Alpha shape boundary extraction and convex surface inference | `reconstruction/completion.py` | Differentiates measured from inferred geometry |
| 8 | **Metric Scaling without GCPs** | Closed-form similarity transformation between camera trajectory and GPS track | `reconstruction/georef.py` | Sub-metre RMS tracking without survey markers |

---

## Reconstruction Targets

- **3D Terrain & Architecture**: High-density point clouds and watertight Poisson surface meshes.
- **Rooftops & Facades**: Multi-view geometric reconstruction with photogrammetric confidence classification.
- **Infrastructure & Roads**: Automated semantic classification (vehicles, structures, pedestrians, vegetation).
- **Photorealistic Radiance Fields**: 3D Gaussian Splatting (`splat.ply`) for real-time viewpoint synthesis.
- **Textured Digital Twins**: High-resolution GLB models with UV-unwrapped texture maps for browser visualization.

---

## Complete Technology Stack

- **Photogrammetry Core**: COLMAP (CUDA) / PyCOLMAP, OpenMVS, Open3D, Trimesh.
- **Machine Learning**: Ultralytics YOLOv8, PyTorch CUDA.
- **Geodesy & Coordinates**: WGS84 -> ECEF -> ENU transformations, Umeyama similarity alignment.
- **Backend Orchestrator**: FastAPI, Pydantic v2, Python 3.11.
- **Interactive Digital Twin**: React 18, Three.js, React Three Fiber, WebGL/WebGPU Splat Engine, Vite.
