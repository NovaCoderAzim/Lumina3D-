# Lumina3D

> **Autonomous Aerial Drone Video to 3D Digital Twin & Real-Time 3D Gaussian Splatting Platform**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/3D_Engine-Three.js_/_R3F-black.svg?logo=three.js&logoColor=white)](https://threejs.org/)
[![COLMAP](https://img.shields.io/badge/Photogrammetry-COLMAP_CUDA-red.svg)](https://colmap.github.io/)
[![YOLOv8](https://img.shields.io/badge/AI_Perception-YOLOv8-blue.svg)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

**Lumina3D** is a spatial computing and photogrammetry platform designed to turn uncalibrated aerial drone footage into high-precision, interactive **3D Digital Twins** and real-time **3D Gaussian Splat (3DGS)** radiance fields.

Equipped with GPU-accelerated computer vision, automated keyframe selection, dynamic moving object masking, GPS telemetry metric scaling, and multi-class semantic intelligence, Lumina3D bridges the gap between raw video capture and engineering-grade spatial inspection.

---

## 🚀 Key Features

- **Autonomous Video-to-3D Pipeline**:
  - Ingest raw `.mp4` or `.mov` aerial drone footage.
  - Sub-second video quality assessment (Laplacian sharpness, illumination drift, ORB overlap scoring).
  - Adaptive keyframe decimation maximizing parallax while discarding motion blur.
- **High-Fidelity Photogrammetry & Surface Reconstruction**:
  - Incremental Structure-from-Motion (SfM) using SIFT feature extraction.
  - Multi-View Stereo (MVS) dense depth fusion generating millions of surface points.
  - Watertight Screened Poisson Surface Reconstruction and Ball-Pivoting Algorithm (BPA).
  - UV texture projection and texture atlas baking into web-optimized `.glb` assets.
- **Real-Time 3D Gaussian Splatting (3DGS)**:
  - Generation and export of `.ply` neural radiance fields.
  - Interactive WebGL/WebGPU radiance rendering in browser without native plugins.
- **Metric Scaling & Georeferencing**:
  - Ingestion of DJI `.srt` sidecars and CSV flight telemetry logs.
  - Closed-form Umeyama similarity alignment recovering true real-world metre scale without requiring manual Ground Control Points (GCPs).
- **AI Perception & Dynamic Object Masking**:
  - Automated YOLOv8 dynamic entity removal (moving cars, pedestrians, animals) before feature extraction to eliminate epipolar errors.
  - Semantic object tracking and 3D centroid projection for interactive asset inspection.
- **Interactive Web Digital Twin**:
  - Five distinct viewing modes: **Textured Mesh**, **3DGS Radiance Field**, **Dense MVS Point Cloud**, **Hybrid Wireframe**, and **Inferred Surface Geometry**.
  - Interactive spatial measurement suite (point-to-point distance, elevation, volume).
  - Ground leveling, spatial alignment, and dark/light high-contrast interface themes.

---

## 🏗️ Architecture

```
drone-video-to-3d-reconstruction/
├── backend/                  # FastAPI orchestration server & REST API
│   ├── api/                  # Project lifecycle, upload, model & report endpoints
│   ├── orchestration/        # Pipeline stage coordinator & state machine
│   └── services/             # Adapters for vision, reconstruction, and analytics
├── reconstruction/           # Photogrammetry & 3DGS core engine
│   ├── colmap_runner.py      # SIFT extraction, matching, SfM & dense MVS
│   ├── splat.py              # 3D Gaussian Splatting radiance field exporter
│   ├── georef.py             # Umeyama telemetry alignment & metric scaling
│   ├── masking.py            # Dynamic moving-object exclusion masks
│   ├── texture.py            # UV unwrapping & photographic texture baking
│   └── mesh.py               # Poisson & Ball-Pivoting surface reconstruction
├── vision/                   # Video processing & frame quality analyzer
│   ├── capture_quality.py    # Sub-second sharpness, exposure, and motion diagnostics
│   └── frame_extractor.py    # Adaptive keyframe decimation
├── telemetry/                # Flight log & GPS processing
│   └── parsers.py            # DJI SRT subtitle parser, CSV reader, ENU geodesy
├── ai/                       # Semantic intelligence & 2D-to-3D projection
│   ├── detector.py           # YOLOv8 object detection wrapper
│   └── tracker.py            # Multi-frame object association
├── src/                      # React 18 / Three.js interactive digital twin UI
│   ├── components/viewer/    # Canvas, 3DGS SplatViewer, GLBModel, Scene controls
│   └── components/screens/   # Landing, Upload, Processing monitor, Digital Twin
└── scripts/                  # Cross-platform startup and test runners
```

---

## ⚡ Quick Start

### Prerequisites
- **Node.js**: v18+ and `npm`
- **Python**: 3.10+ (Python 3.11 recommended for full CUDA support)
- **GPU (Optional)**: NVIDIA GPU with CUDA for accelerated dense MVS and YOLO inference (CPU mock modes available for offline testing).

### 1. Frontend Setup
```bash
npm install
npm run dev
```
The frontend starts on `http://localhost:5173`.

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation is accessible at `http://127.0.0.1:8000/docs`.

### 3. One-Click Launch (Windows PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
```

---

## 🧪 Testing

```bash
# Run unit tests and contract validation in mock mode
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1

# Or run pytest directly
pytest tests/ -v
pytest reconstruction/test_reconstruction.py -v

# Frontend type safety & build verification
npm run build
```

---

## 📄 Documentation

- [Lumina3D Project Report](LUMINA_3D_PROJECT_REPORT.md) — Comprehensive technical report and mathematical specifications.
- [System Architecture & Capabilities](docs/LUMINA_3D_SYSTEM_ARCHITECTURE.md) — Detailed feature map and photogrammetric pipeline breakdown.
- [Frontend Guide](README.frontend.md) — UI design system and Three.js viewer architecture.
- [Reconstruction Engine Guide](reconstruction/README.md) — Deep dive into SfM, MVS, Poisson meshing, and 3DGS export.
- [AI Perception Guide](ai/README.md) — YOLOv8 object detection and tracking documentation.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
