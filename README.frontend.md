# Lumina3D — Frontend & Interactive 3D Digital Twin Viewer

The user-facing web application for **Lumina3D**: an interactive 3D digital twin platform built with React, Three.js, React Three Fiber, Vite, and modern WebGL/WebGPU radiance rendering.

## Getting Started

```bash
npm install
npm run dev       # Launches on http://localhost:5173
npm run build     # Production build in dist/
npm run preview   # Preview production bundle
```

## Core Features

- **Workflow Flow**: Landing -> Project Creation & Video Upload -> Real-Time Processing Monitor -> Interactive 3D Digital Twin.
- **Multi-Modal 3D Viewer**:
  - **Textured Mesh**: Multi-view UV unwrapped photogrammetric model.
  - **3D Gaussian Splatting (3DGS)**: Real-time neural radiance field rendering powered by `@mkkellogg/gaussian-splats-3d`.
  - **Dense Point Cloud**: High-density OpenMVS point cloud with color-graded confidence and elevation overlays.
  - **Hybrid Mode**: Combined mesh wireframe and volumetric point clouds.
  - **Inferred Geometry**: Surface completion for unobserved occlusions.
- **Digital Twin Intelligence**:
  - Semantic object recognition overlays (Vehicles, Buildings, People, Vegetation).
  - Volumetric, distance, and height measurement tools.
  - Ground leveling, spatial alignment, and georeferenced coordinate inspection.
  - Dark / Light high-contrast theme toggling.
- **Direct Backend Integration**: Connects to the Lumina3D FastAPI backend via `/api` proxy or standalone API endpoint.

## Architecture

```
src/
  types/             TypeScript models for projects, geometry, and semantics
  lib/api.ts         Typed API client for backend communication
  data/demoData.ts   Offline fallback assets
  components/
    screens/         Landing, Upload, Processing, DigitalTwin
    viewer/          Viewer3D, GLBModel, SplatViewer, DenseCloudViewer, CameraFlightPath
    layers/          LayersPanel, AnalyticsPanel, MeasurePanel
    Header, Sidebar, StatusBar, ObjectPanel
```
