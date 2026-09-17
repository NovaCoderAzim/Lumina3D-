# DRISHTI-3D ΓÇö Frontend & Interactive 3D Digital Twin Viewer

Person 5's deliverable: the application judges actually see. Turns the
outputs of reconstruction (COLMAP), detection (YOLO), and the backend API
into a polished upload ΓåÆ process ΓåÆ inspect experience.

## Run it

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build in dist/
npm run preview   # serve the production build
```

## What's here

- **Landing ΓåÆ Upload ΓåÆ Processing ΓåÆ Digital Twin**, the full journey from
  Section 29 of the brief, driven by a small state machine in `src/App.tsx`.
- **Demo mode is on by default** (`DEMO_MODE = true` in `App.tsx`) since no
  live backend is wired into this environment. It uses the precomputed data
  in `src/data/demoData.ts` and a procedural fallback 3D scene
  (`src/components/viewer/DemoScene.tsx`) instead of a real `model.glb`, so
  the UI is fully explorable without a backend.
- **Real backend path**: `src/lib/api.ts` is a typed client for the contract
  in Section 17 (`POST /projects`, `.../upload`, `GET .../status`,
  `.../model`, `.../semantic`, `.../analytics`). Set `VITE_API_BASE_URL` and
  flip `DEMO_MODE` to `false` in `App.tsx` to switch over. `GLBModel.tsx`
  loads a real `model.glb`, computes its bounding box, and re-centers it ΓÇö
  the viewer never assumes the exported origin is correct.

## Structure

```
src/
  types/             project status, semantic object, and analytics models
  lib/api.ts         typed API client
  data/demoData.ts   precomputed demo-mode fallback data
  components/
    screens/         Landing, Upload, Processing, DigitalTwin
    viewer/          Viewer3D (canvas/lighting/controls), GLBModel, DemoScene
    layers/          LayersPanel, AnalyticsPanel
    Header, Sidebar, StatusBar, ObjectPanel
```

## Design

Dark technical/engineering interface ΓÇö near-black panels, hairline borders,
JetBrains Mono for chrome and data readouts, an amber instrument accent
(nods to old avionics/survey-equipment phosphor displays), and a
category-coded palette for semantic layers (vehicles blue, buildings
steel-gray, people coral, vegetation green). The 3D viewport is always the
dominant element on screen, per Section 5 of the brief.

## Notes

- Progress is shown as **stage, not percentage** unless the backend
  provides a real number (Section 18).
- If `estimated_3d_position` is `null`, the object panel always shows
  "3D Association: Unavailable" ΓÇö it never fabricates a coordinate
  (Section 12/13).
- "Registration Rate" / "Detection Confidence" terminology is used
  deliberately instead of "Accuracy" (Section 15).
