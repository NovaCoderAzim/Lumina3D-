# Lumina3D — AI & Semantic Intelligence Module

Turns selected drone frames into a structured semantic digital twin:

```
Selected Frames -> YOLO detection -> confidence filtering -> multi-frame
tracking -> deduplication -> semantic categorization -> (optional)
reconstruction association -> objects.json + semantic.json
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

Or set `USE_MOCK_AI=true` to get schema-correct data for testing:

```python
from interface import AIProcessor
p = AIProcessor(use_mock=True)
result = p.process(frames_directory="unused", project_id="demo_001")
```

## Directory layout

```
ai/
├── interface.py      # AIProcessor — main public interface
├── detector.py       # YOLO wrapper: image -> filtered Detection list
├── tracker.py        # IoU + centroid multi-frame tracker -> deduped Track list
├── semantic.py       # class -> category config, and summary counts
├── association.py    # 2D -> 3D reconstruction association (best-effort)
├── schemas.py        # dataclasses for JSON contracts
├── mock.py           # USE_MOCK_AI=true path, same schema as real output
├── demo/objects.json # sample data for mock mode
├── test_smoke.py     # fast sanity check, no GPU required
├── requirements.txt
└── .env.example
```

## Configuration

| Env var                | Default        | Meaning                                      |
|------------------------|----------------|----------------------------------------------|
| `DETECTION_MODEL`      | `yolov8n.pt`   | Ultralytics model name or local path         |
| `DETECTION_CONFIDENCE` | `0.40`         | Minimum confidence kept after filtering       |
| `USE_GPU`              | `true`         | Use CUDA if available, else CPU fallback     |
| `USE_MOCK_AI`          | `false`        | Skip the real model, return demo/objects.json |

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

`reconstruction_metadata` is the reconstruction engine's plain dict/JSON output (camera
poses + reconstructed points). If it is `None` or incomplete, detection and
tracking still run fully; only the 3D association fields stay `null`.

## Output files

Written to `data/projects/{project_id}/ai/`:

- **`detections.json`** — every raw filtered detection, per frame
- **`objects.json`** — deduplicated tracked objects for 3D overlay
- **`semantic.json`** — category counts (`buildings`, `vehicles`, `people`, `vegetation`, ...)

## Confidence Metrics

1. **Detection confidence** — how confident YOLO is about a detected class.
2. **Reconstruction confidence** — photogrammetric ray intersection uncertainty from the reconstruction engine.
3. **Association confidence** — confidence of 2D bounding boxes projected into 3D space.

## Failure handling

Degrades gracefully to a valid `COMPLETED` result with an empty
object list (or a `FAILED` status with a message for missing input), and
never crashes the caller:

- Model / GPU unavailable
- Invalid or unreadable image
- Zero detections in a frame or across the whole run
- Tracking finds nothing to link
- Missing camera poses / reconstruction metadata

"No detections" is success, not failure — see `schemas.AIResult`.
