# DRISHTI-3D ΓÇö AI & Semantic Intelligence Module (Person 4)

Turns selected drone frames into a structured semantic digital twin:

```
Selected Frames ΓåÆ YOLO detection ΓåÆ confidence filtering ΓåÆ multi-frame
tracking ΓåÆ deduplication ΓåÆ semantic categorization ΓåÆ (optional)
reconstruction association ΓåÆ objects.json + semantic.json
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # tune as needed

python -c "
from interface import AIProcessor
p = AIProcessor()
result = p.process(
    frames_directory='data/projects/demo_001/frames/selected',
    project_id='demo_001',
)
print(result.status, result.summary)
"
```

No GPU / no model yet? Run the smoke tests, which exercise mock mode and
failure handling without needing torch/ultralytics installed:

```bash
python test_smoke.py
```

Or set `USE_MOCK_AI=true` to get schema-correct fake data for any project:

```python
from interface import AIProcessor
p = AIProcessor(use_mock=True)
result = p.process(frames_directory="unused", project_id="demo_001")
```

## Directory layout

```
ai/
Γö£ΓöÇΓöÇ interface.py    # AIProcessor ΓÇö the only class other people should import
Γö£ΓöÇΓöÇ detector.py      # YOLO wrapper: image -> filtered Detection list
Γö£ΓöÇΓöÇ tracker.py        # IoU + centroid multi-frame tracker -> deduped Track list
Γö£ΓöÇΓöÇ semantic.py        # class -> category config, and summary counts
Γö£ΓöÇΓöÇ association.py      # MVP 2D -> reconstruction association (best-effort)
Γö£ΓöÇΓöÇ schemas.py            # dataclasses = the JSON contract with Person 5 / 6
Γö£ΓöÇΓöÇ mock.py                 # USE_MOCK_AI=true path, same schema as real output
Γö£ΓöÇΓöÇ demo/objects.json         # sample data for mock mode
Γö£ΓöÇΓöÇ test_smoke.py               # fast sanity check, no GPU required
Γö£ΓöÇΓöÇ requirements.txt
ΓööΓöÇΓöÇ .env.example
```

## Configuration

| Env var                | Default        | Meaning                                   |
|-------------------------|----------------|--------------------------------------------|
| `DETECTION_MODEL`        | `yolov8n.pt`   | Ultralytics model name or local path        |
| `DETECTION_CONFIDENCE`    | `0.40`        | Minimum confidence kept after filtering      |
| `USE_GPU`                  | `true`       | Use CUDA if available, else CPU fallback      |
| `USE_MOCK_AI`                | `false`    | Skip the real model, return demo/objects.json  |

All four can also be passed directly to `AIProcessor(...)`.

## Public interface

```python
class AIProcessor:
    def process(
        self,
        frames_directory: str,
        project_id: str,
        reconstruction_metadata: dict | None = None,
    ) -> AIResult:
        ...
```

`reconstruction_metadata` is Person 3's plain dict/JSON output (camera
poses + reconstructed points) ΓÇö this module never imports another
person's internal classes. If it's `None` or incomplete, detection and
tracking still run fully; only the 3D association fields stay `null`.

## Output files

Written to `data/projects/{project_id}/ai/`:

- **`detections.json`** ΓÇö every raw filtered detection, per frame
- **`objects.json`** ΓÇö deduplicated tracked objects (the main contract with Person 5)
- **`semantic.json`** ΓÇö category counts (`buildings`, `vehicles`, `people`, `vegetation`, ...)

## Confidence ΓÇö three different numbers, never merged

1. **Detection confidence** ΓÇö how sure YOLO is about a class. Stored as-is, never called "accuracy".
2. **Reconstruction confidence** ΓÇö owned by Person 3, not this module.
3. **Association confidence** ΓÇö how sure the 2DΓåÆ3D heuristic is that a
   detection maps to a specific reconstructed point. Deliberately capped
   low (0.1ΓÇô0.6) since this is a coarse proxy, not ray-cast geometry.

## Failure handling

Every one of these degrades to a valid `COMPLETED` result with an empty
object list (or a `FAILED` status with a message for missing input), and
never crashes the caller:

- model / GPU unavailable
- invalid or unreadable image
- zero detections in a frame or across the whole run
- tracking finds nothing to link
- missing camera poses / reconstruction metadata

"No detections" is success, not failure ΓÇö see `schemas.AIResult`.

## Integration contracts

**From Person 3:** `reconstruction_metadata` ΓÇö camera poses, and
optionally point cloud/depth, as a plain dict. Fully optional; absence
never blocks detection.

**To Person 5:** `objects.json` (id, class, category, confidence,
observations, frames_seen, and ΓÇö where available ΓÇö bbox_history /
estimated_3d_position / semantic_region / association_confidence),
plus `semantic.json` for the toggleable category view.

**To Person 6:** `AIResult.statistics` ΓÇö frames_processed, detections,
unique_objects, average_detection_confidence, processing_time_seconds.

## One-sentence responsibility

> "I make the 3D digital twin understand what it's looking at."
