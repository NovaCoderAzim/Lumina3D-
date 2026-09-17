# DRISHTI-3D: Complete Technical Architecture & Pipeline Specification
**Autonomous Single-Pass Drone Video to Metric 3D Digital Twin with AI Semantics**  
*SIH Problem Statement: SIH 26158 | Comprehensive A-to-Z Technical Project Report*

---

## Executive Summary

**DRISHTI-3D** is an end-to-end computer vision, photogrammetry, and spatial intelligence platform. It ingests raw, uncalibrated, single-pass aerial drone video and autonomously produces an accurate, metric-scaled, interactive **3D Digital Twin** accompanied by multi-class AI semantic object detection and georeferencing.

Traditional photogrammetry pipelines require manual Ground Control Points (GCPs), manual feature matching constraints, expensive commercial licenses, and hours of offline desktop compute. DRISHTI-3D overcomes these hurdles through an integrated, automated pipeline that:
1. Evaluates capture quality and filters keyframes based on parallax and sharpness.
2. Solves camera trajectory and sparse Structure-from-Motion (SfM) via PyCOLMAP.
3. Computes dense Multi-View Stereo (MVS) generating over **1.1 million 3D points**.
4. Reconstructs surface topology using Poisson and Ball-Pivoting algorithms with KDTree camera color transfer.
5. Ingests frames into a GPU-accelerated **YOLOv8** network to track objects and project them into 3D space.
6. Delivers an interactive **WebGL 3D Digital Twin** via React Three Fiber with zero client plugin requirements.

```
       RAW DRONE VIDEO (.MP4 / .MOV)
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Stage 0: Video Quality & Motion Analysis    │
│  (Laplacian blur, Optical Flow Overlap, QA)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Adaptive Keyframe Selection        │
│  (Parallax delta, 30–50 optimal keyframes)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Feature Extraction & Matching      │
│  (PyCOLMAP SIFT descriptors, Sequential)     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Incremental Structure-from-Motion  │
│  (Camera bundle adjustment, Sparse cloud)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 4: Dense Multi-View Stereo (MVS)      │
│  (PatchMatch Stereo depth fusion, 1.1M+ pts) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 5: Surface Meshing & Color Transfer   │
│  (Laplacian smoothing, KDTree RGB mapping)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 6: AI Perception & 3D Ray-Casting     │
│  (YOLOv8 GPU detection, ByteTrack tracking)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 7: glTF Axis Flip & Conversion        │
│  (COLMAP right-handed -> glTF Y-up format)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 8: Interactive WebGL Digital Twin     │
│  (SPARSE, DENSE, MESH, TEXTURED, HYBRID)     │
└──────────────────────────────────────────────┘
```

---

## 1. System Technology Stack

| Layer | Technology | Version / Specification | Role in DRISHTI-3D |
| :--- | :--- | :--- | :--- |
| **Runtime Environment** | Python (Virtualenv) | 3.11.9 (64-bit) | Backend compute, Open3D & PyCOLMAP native compatibility |
| **Web Runtime** | Node.js | v20.19.0 (npm 11.3.0) | Frontend compilation and client execution |
| **API Server** | FastAPI & Uvicorn | 0.115+ / 0.34+ | Modular async REST backend with streaming status |
| **Photogrammetry (SfM)** | PyCOLMAP | 0.6.0 / C++ binding | SIFT feature extraction, sequential matcher, bundle adjustment |
| **Dense Reconstruction**| COLMAP MVS / Open3D | PatchMatch Stereo | Dense depth map estimation, stereo fusion, point filtering |
| **Geometric Processing**| Open3D & Trimesh | 0.18.0 / 4.4.0 | Poisson meshing, Ball Pivoting, KDTree color transfer, GLB conversion |
| **AI Perception** | Ultralytics YOLOv8 | YOLOv8n / PyTorch 2.6 | Real-time object detection (Vehicles, Infrastructure, Pedestrians) |
| **GPU Acceleration** | NVIDIA CUDA | 12.4 (RTX 4060 8GB) | CUDA acceleration for SIFT, YOLOv8 inference, and PyTorch |
| **Frontend Framework** | React + TypeScript | React 19 / TS 5.7 | Single Page Application (SPA) architecture |
| **3D Rendering** | Three.js & R3F | Three r174 / Fiber v9 | WebGL GPU hardware-accelerated 3D Digital Twin renderer |
| **UI & Styling** | Vanilla CSS / Tailwind | Tailwind CSS v4 | High-density mission-control dark theme with glassmorphism |
| **Spatial Telemetry** | OpenCV & NumPy | 4.10.0 / 1.26.4 | Optical flow, Laplacian blur detection, Umeyama metric scaling |

---

## 2. End-to-End Pipeline Architecture (A to Z)

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Quality QA"]
        V[Raw Drone Video .mp4/.mov] --> QA[Capture Quality Analyzer]
        QA -->|Laplacian Blur Score| BLUR{Sharp?}
        BLUR -- No --> REJ[Flag Low Sharpness Warning]
        BLUR -- Yes --> OF[Optical Flow Overlap Analysis]
        OF --> KS[Adaptive Keyframe Extraction]
        KS --> FRAMES[(30–50 Selected Keyframes)]
    end

    subgraph PHOTOGRAMMETRY["2. 3D Photogrammetry & Surface Engine"]
        FRAMES --> SIFT[SIFT Feature Extraction\nPyCOLMAP Engine]
        SIFT --> MATCH[Sequential Matcher\nOverlap=10]
        MATCH --> SFM[Incremental SfM\nBundle Adjustment]
        SFM --> SP[Sparse Point Cloud\ncloud_sparse.ply: 21K pts]
        SFM --> STEREO[PatchMatch MVS Stereo\nDepth & Normal Fusion]
        STEREO --> DENSE[Dense Point Cloud\ncloud_dense.ply: 1.1M+ pts]
        DENSE --> MESH[Surface Reconstruction\nPoisson / Ball-Pivoting]
        MESH --> COLOR[KDTree Color Transfer\nTrue Camera RGB Mapping]
        COLOR --> GLB_EXP[glTF Axis Transform\n(x,y,z) -> (x,-y,-z)]
        SP --> GLB_EXP
        DENSE --> GLB_EXP
        GLB_EXP --> MODEL[model.glb: 25.4 MB]
    end

    subgraph AI_LAYER["3. Neural Perception & Georeferencing"]
        FRAMES --> YOLO[YOLOv8 Inference\nNVIDIA RTX 4060 GPU]
        YOLO --> TRACK[ByteTrack Trajectory Association]
        TRACK --> RAY[Camera Pose Ray-Casting]
        RAY --> OBJ[(3D Semantic Objects: 14 Vehicles)]
        SFM --> METRIC[Telemetry / Umeyama Scale Estimation]
    end

    subgraph WEBGL["4. Mission Control & Digital Twin Viewer"]
        MODEL & OBJ & METRIC --> API[FastAPI Backend\nDaemon Port 8000]
        API --> VITE[Vite Proxy Port 5173]
        VITE --> R3F[React Three Fiber Canvas]
        R3F --> MODES{View Mode Switcher}
        MODES --> SPARSE[SPARSE: Golden Amber SfM Tie Points]
        MODES --> DENSE_V[DENSE: 1.1M Micro-Splats 0.022]
        MODES --> MESH_V[MESH: Smooth Colored Surface]
        MODES --> HYBRID_V[HYBRID: Z-Fighting Free Composite]
        MODES --> TEXT_V[TEXTURED: Photogrammetric UV Map]
    end
```

### Stage 0: Pre-Flight Capture Quality Analysis
- **Module**: [`reconstruction/capabilities.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/capabilities.py) and [`backend/services/geo_service.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/backend/services/geo_service.py)
- **Problem**: Processing unviable video wastes significant GPU time and RAM.
- **Solution**: Evaluates video characteristics within the first 10 seconds:
  1. **Variance of Laplacian ($\sigma^2_{\Delta}$)**: Flags frames with motion blur (< 50 threshold).
  2. **Histogram Divergence**: Identifies sudden exposure fluctuations or harsh shadowing.
  3. **Gunnar-Farnebäck Optical Flow**: Determines baseline overlap between consecutive frames; warns if inter-frame overlap drops below 60%.
  4. **Flight Classification**: Classifies trajectory into `Orbital`, `Linear Flyover`, or `Stationary`.

### Stage 1: Adaptive Keyframe Extraction
- **Module**: [`backend/services/vision_service.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/backend/services/vision_service.py)
- **Problem**: Drone video recorded at 30/60 FPS contains high spatial redundancy. Passing all 900+ frames into SfM causes memory overflow and baseline noise.
- **Solution**: An adaptive selector analyzes cumulative motion. It extracts only high-information frames with sufficient baseline parallax, yielding **30–50 crisp keyframes** (`frame_000000.jpg`, `frame_000014.jpg`, etc.).

### Stage 2: SIFT Feature Extraction & Sequential Matching
- **Module**: [`reconstruction/colmap_runner.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/colmap_runner.py)
- **Processing**:
  - Uses PyCOLMAP to extract up to 8,192 invariant SIFT keypoints per frame.
  - Applies a **Sequential Matcher** (`overlap=10`) optimized for drone video trajectories.
  - Matches are verified geometrically via epipolar constraints and stored in `database.db`.

### Stage 3: Incremental Structure-from-Motion (SfM)
- **Module**: [`reconstruction/colmap_runner.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/colmap_runner.py)
- **Processing**:
  - Initializes the best image pair with high baseline and sufficient parallax.
  - Iteratively registers new images using P3P RANSAC pose estimation.
  - Performs bundle adjustment to minimize reprojection error across all cameras.
  - **Results**: Triangulates camera poses (100% registration rate on test datasets, 42/42 images registered, **reprojection error = 0.57 px**) and exports `cloud_sparse.ply` (21,401 tie points).

### Stage 4: Dense Multi-View Stereo (MVS)
- **Module**: [`reconstruction/engines/colmap_dense.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/engines/colmap_dense.py)
- **Processing**:
  - Computes photometric and geometric depth maps for each registered view using PatchMatch stereo.
  - Fuses depth and normal maps across views with strict visibility consistency.
  - Removes floaters and statistical outliers via SOR (`std_ratio=2.5`, `nb_neighbors=20`).
  - **Result**: Dense point cloud containing **1,101,133 authentic 3D coordinates** with camera RGB colors (`cloud_dense.ply`).

### Stage 5: Surface Meshing & Color Transfer
- **Module**: [`reconstruction/mesh.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/mesh.py) & [`reconstruction/conversion.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/conversion.py)
- **Mesh Generation**:
  - Applies Ball-Pivoting (BPA) and Screened Poisson surface reconstruction.
  - Trims low-density boundary faces to prevent artificial bulging.
  - Runs Laplacian smoothing to eliminate raw Delaunay triangulation faceted crinkling while preserving architectural corners.
- **True RGB Color Transfer**:
  - Standard meshing outputs neutral gray vertices (`[102, 102, 102]`).
  - DRISHTI-3D utilizes a SciPy **KDTree spatial search** to query the nearest dense camera points and assign authentic RGB color vectors directly to each of the 186,977 mesh vertices.

### Stage 6: AI Semantic Perception (2D to 3D Projection)
- **Module**: [`backend/services/ai_service.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/backend/services/ai_service.py)
- **Processing**:
  - Keyframes pass through **YOLOv8** on NVIDIA RTX 4060 GPU with CUDA acceleration.
  - Detects vehicles, buildings, pedestrians, and infrastructure.
  - Links multi-frame detections with ByteTrack trajectory management.
  - Projects 2D bounding box centroids through calibrated camera intrinsic matrices onto the 3D surface geometry, estimating spatial world coordinates for each object.

### Stage 7: Axis Conversion & Multi-Layer glTF Assembly
- **Module**: [`reconstruction/conversion.py`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/reconstruction/conversion.py)
- **Coordinate Conversion**:
  - COLMAP uses right-handed coordinates: $+X$ right, $+Y$ down, $+Z$ forward.
  - WebGL / Three.js / glTF uses: $+X$ right, $+Y$ up, $-Z$ forward.
  - Converts every vertex via fixed 180° rotation matrix about the X-axis:
    $$(x, y, z) \longrightarrow (x, -y, -z)$$
- **Multi-Layer GLB**:
  - Packs `sparse_points` (21K pts), `dense_points` (1.1M pts), and `mesh_surface` (186K verts, 374K faces) into a unified 25.4 MB binary glTF (`model.glb`).
  - No synthetic placeholders or artificial diorama pedestals are added.

### Stage 8: Interactive WebGL Digital Twin Viewer
- **Module**: [`src/components/viewer/GLBModel.tsx`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/src/components/viewer/GLBModel.tsx) & [`src/components/viewer/Viewer3D.tsx`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/src/components/viewer/Viewer3D.tsx)
- Renders the complete 3D scene in the browser using React Three Fiber with:
  - Micro-point radial alpha splatting (preventing chunky pixel blocks).
  - Polygon offset factor and units (`factor: 2.0`, `units: 2.0`) eliminating Z-fighting in Hybrid mode.
  - Interactive Wireframe overlay toggle.
  - Orbit navigation, metric measurement tool, and layer visibility controls.

---

## 3. What Has Been Accomplished (Work Done)

### Photogrammetry & Geometry Overhaul
1. **1.1M+ Dense Point Cloud Extraction**: Increased dense point retention from a downsampled 438K to over 1.1 million points using fine 1.5 cm voxel preservation.
2. **Elimination of Uncolored "Cement" Mesh**: Developed automatic KDTree color transfer mapping real camera RGB pixels to all 186,977 mesh vertices.
3. **Laplacian Surface Smoothing**: Implemented mesh smoothing to remove Delaunay crinkling, producing organic terrain and clean building planes.
4. **Anti-Aliased Micro-Splat Point Rendering**: Replaced square blocks with soft circular radial alpha textures and tuned default point size from `0.08` down to `0.022`.
5. **Fixed Z-Fighting in Hybrid Mode**: Added OpenGL polygon depth offsets so the dense point cloud and surface mesh seamlessly blend without flickering.
6. **Golden Amber SfM Tie-Points**: Styled sparse points in high-contrast amber (`#ffaa22`) with size scaling for clear visualization of the camera feature constellation.
7. **Pedestal / Skirt Removal**: Stripped artificial diorama base geometry to preserve 100% scientific photogrammetric honesty.

### Reliability & Infrastructure
8. **502 Bad Gateway Prevention**: Configured FastAPI backend as a persistent daemon process on port 8000, preventing Vite proxy disconnects.
9. **Added HTTP HEAD Method Support**: Added `methods=["GET", "HEAD"]` across all asset endpoints (`/model/file`, `/pointcloud/file`, `/mesh/file`), preventing 405 errors during browser prefetch probes.
10. **WebGL Crash Prevention with Error Boundaries**: Built [`ModelErrorBoundary.tsx`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/src/components/viewer/ModelErrorBoundary.tsx) wrapping both Canvas and GLBModel. Network drops or parsing errors display an in-viewer retry card instead of dropping the WebGL context.
11. **Interactive Retry & Cache Eviction**: Integrated `useGLTF.clear(url)` to flush stale promises and allow 1-click model reloading without refreshing the tab.
12. **Zero TypeScript Build Errors**: Validated production build (`npm run build`), resolving unused variables and type issues.

### Navigation & UX Features
13. **Digital Twin Mission Control**: Built the project gallery in [`Landing.tsx`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/src/components/screens/Landing.tsx) displaying all reconstructed projects, dense point metrics, status badges, and search filtering.
14. **Persistent Project URLs**: Implemented `?projectId=proj_...` URL synchronization with HTML5 `popstate` support. Refreshing or sharing the URL preserves the exact model.
15. **Global Navigation Controls**: Added `← Projects` and `+ New Project` navigation buttons across headers, upload screens, and processing views.
16. **Responsive Metric Distance Tool**: Integrated interactive 3D point-to-point Euclidean measuring on the model surface.

---

## 4. Comparison of 3D Visualization Modes

| Mode | Geometry Displayed | Visual Characteristics | Best Used For |
| :--- | :--- | :--- | :--- |
| **SPARSE** | 21,401 SfM Tie-Points | Golden amber (`#ffaa22`) point constellation | Inspecting camera bundle alignment and feature tracking density |
| **DENSE** | 1,101,133 MVS Points | High-density RGB micro-splats (`pointSize: 0.022`) | Inspecting fine foliage, architectural details, and complex depth |
| **MESH** | 186,977 Vertices, 374K Faces | Continuous surface with vertex camera RGB and smooth normals | Spatial analysis, surface area inspection, collision boundaries |
| **HYBRID** | Mesh + Micro-Points | Continuous surface with point highlights; zero Z-fighting | Best overall visual fidelity; combines continuous volume with sharp detail |
| **TEXTURED**| Photogrammetric Mesh | UV texture-mapped surface | High-fidelity photorealistic rendering (when texture atlas available) |

---

## 5. What Needs to Be Done (Future Roadmap)

While DRISHTI-3D is fully functional and delivers authentic 3D digital twins, the following features represent valuable future enhancements:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FUTURE EXPANSION ROADMAP                           │
├──────────────────────────┬──────────────────────────┬───────────────────┤
│    SHORT-TERM (Phase 1)  │   MEDIUM-TERM (Phase 2)  │ LONG-TERM (Phase 3│
├──────────────────────────┼──────────────────────────┼───────────────────┤
│ • Chunked Video Upload   │ • 3D Gaussian Splatting  │ • Drone Edge Unit │
│ • RTK/EXIF GPS Auto-Sync │ • Semantic 3D Mesh Seg.  │ • Multi-UAV Fusion│
│ • Volumetric Measurement │ • BIM/CAD (IFC/OBJ) Export│ • Change Detection│
└──────────────────────────┴──────────────────────────┴───────────────────┘
```

### Phase 1: Short-Term Enhancements
1. **Chunked Video Upload**:
   - Add slice-based chunked streaming upload in [`api.ts`](file:///d:/projects/3D%20Reconstruction%20model/drone-video-to-3d-reconstruction/src/lib/api.ts) for multi-gigabyte 4K/8K drone videos, including pause/resume support.
2. **Automated RTK / EXIF GPS Extraction**:
   - Automatically parse embedded subtitles (`.srt`), EXIF metadata, or DJI companion logs to establish absolute metric scale without manual telemetry uploads.
3. **Volumetric Measurement Tool**:
   - Extend the existing point-to-point measurement tool to calculate 3D surface area and stockpile volume (crucial for construction and mining use cases).

### Phase 2: Medium-Term Upgrades
4. **3D Gaussian Splatting (3DGS) Pipeline**:
   - Implement real-time rasterized 3D Gaussian Splatting alongside traditional meshes for photorealistic visual fidelity at 60+ FPS in WebGL.
5. **Semantic 3D Mesh Segmentation**:
   - Instead of 2D bounding-box ray projections, classify individual 3D mesh faces directly into semantic categories (Road, Building, Tree, Vehicle) using PointNet++ or 3D-UNet.
6. **BIM / CAD Export Formats**:
   - Provide direct one-click exports to standard engineering formats including **OBJ+MTL**, **LAS/LAZ** (for LiDAR compatibility), and **IFC** (for BIM/Revit workflows).

### Phase 3: Long-Term Innovations
7. **Onboard Edge Computing**:
   - Package the keyframe extraction and quality analyzer for onboard edge computers (e.g. NVIDIA Jetson Orin) mounted directly to the drone for real-time in-flight QA.
8. **Multi-Drone Collaborative Mesh Federation**:
   - Merge overlapping flight videos from multiple simultaneous drones into a unified coordinate frame using distributed bundle adjustment.
9. **Temporal 4D Change Detection**:
   - Enable time-lapse comparisons between two flight reconstructions of the same site to track construction progress or disaster damage.

---

## 6. Project Structure & Directory Reference

```
drone-video-to-3d-reconstruction/
├── backend/
│   ├── api/                    # FastAPI routes (/projects, /model, /analytics, /upload)
│   ├── orchestration/          # State machine, pipeline runner, project repository
│   ├── services/               # Vision analysis, AI inference, georeferencing services
│   └── main.py                 # FastAPI application entry point
├── reconstruction/
│   ├── colmap_runner.py        # PyCOLMAP feature extraction & SfM execution
│   ├── engines/                # Dense MVS stereo depth estimation engines
│   ├── mesh.py                 # Poisson & Ball-Pivoting meshing with color transfer
│   ├── conversion.py           # Coordinate system flip & multi-layer GLB exporter
│   └── pipeline.py             # Master reconstruction pipeline orchestrator
├── src/
│   ├── components/
│   │   ├── screens/            # Landing (Project Directory), Upload, Processing, DigitalTwin
│   │   ├── viewer/             # Viewer3D, GLBModel, ModelErrorBoundary, DemoScene
│   │   └── layers/             # Sidebar, LayersPanel, AnalyticsPanel, MeasurePanel
│   ├── lib/api.ts              # REST API client
│   └── App.tsx                 # Screen router & state management
├── data/projects/              # Project persistence directory (videos, frames, models, stats)
├── requirements.txt            # Python root dependencies
├── package.json                # Frontend npm configuration
└── vite.config.ts              # Vite server & API proxy configuration
```

---

## 7. How to Run the Platform

### 1. Start the Backend Service
```powershell
# In project root:
.venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start the Frontend Dev Server
```powershell
# In project root:
npm run dev -- --host 127.0.0.1 --port 5173
```

### 3. Access the Digital Twin Mission Control
Open your browser to:
```
http://127.0.0.1:5173/
```
- Click on any existing project card (e.g. `proj_20260916_0001`) to open the 3D Digital Twin viewer.
- Click **+ New Project** to upload a new aerial drone video.
- Toggle between **SPARSE**, **DENSE**, **MESH**, and **HYBRID** layers in real time.
