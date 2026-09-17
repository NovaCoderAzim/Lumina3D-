"""
Two jobs only:

1. validate_input_images  - sanity-check the input frames
   BEFORE we spend any GPU time on them.
2. validate_glb           - make sure the file we're about to report as
   COMPLETED will actually load in a browser.

Keeping this separate from pipeline.py means both checks are trivially
unit-testable without touching COLMAP at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import trimesh

from .schemas import ErrorCode

MIN_IMAGES = 3
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_DIMENSION_PX = 200  # below this, COLMAP feature matching is unreliable


class ValidationError(Exception):
    """Raised for any failure in this module. Carries an ErrorCode so the
    pipeline can turn it straight into a ReconstructionError without
    re-classifying anything."""

    def __init__(self, error_code: ErrorCode, message: str, recommendation: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.recommendation = recommendation


@dataclass
class ValidatedImage:
    path: Path
    width: int
    height: int


def validate_input_images(frames_directory: str | Path) -> list[ValidatedImage]:
    frames_dir = Path(frames_directory)

    if not frames_dir.is_dir():
        raise ValidationError(
            ErrorCode.INVALID_IMAGES,
            f"Frames directory does not exist: {frames_dir}",
            "Confirm the frame-selection step ran and wrote to this path.",
        )

    candidates = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )

    if len(candidates) < MIN_IMAGES:
        raise ValidationError(
            ErrorCode.TOO_FEW_IMAGES,
            f"Only {len(candidates)} usable image(s) found in {frames_dir}; "
            f"need at least {MIN_IMAGES}.",
            "Select more frames from the source video, or lower the frame "
            "filtering threshold in the vision pipeline.",
        )

    validated: list[ValidatedImage] = []
    unreadable: list[str] = []

    for path in candidates:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:  # re-open: verify() invalidates the handle
                width, height = img.size
        except (UnidentifiedImageError, OSError):
            unreadable.append(path.name)
            continue

        if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
            unreadable.append(f"{path.name} ({width}x{height}, too small)")
            continue

        validated.append(ValidatedImage(path=path, width=width, height=height))

    if len(validated) < MIN_IMAGES:
        raise ValidationError(
            ErrorCode.INVALID_IMAGES,
            f"Only {len(validated)} of {len(candidates)} images were readable "
            f"and above the minimum resolution. Rejected: {unreadable[:10]}",
            "Re-export frames from the source video; check for corrupt or "
            "truncated JPEGs.",
        )

    return validated


def validate_glb(glb_path: str | Path) -> None:
    """Raises ValidationError unless the file exists, is non-empty, parses,
    and contains actual geometry. A reconstruction must NEVER be reported
    COMPLETED if this fails."""

    path = Path(glb_path)

    if not path.exists():
        raise ValidationError(
            ErrorCode.GLB_VALIDATION_FAILED,
            f"model.glb does not exist at {path}",
            "Check conversion.py logs for the export step that should have "
            "produced this file.",
        )

    if path.stat().st_size == 0:
        raise ValidationError(
            ErrorCode.GLB_VALIDATION_FAILED,
            f"model.glb at {path} is empty (0 bytes).",
            "The GLB export likely failed silently; check the mesh export step.",
        )

    try:
        import trimesh

        loaded = trimesh.load(path, file_type="glb")
    except Exception as exc:  # noqa: BLE001 - we want to convert ANY parse failure
        raise ValidationError(
            ErrorCode.GLB_VALIDATION_FAILED,
            f"model.glb at {path} could not be parsed: {exc}",
            "The exported GLB is corrupt; re-run GLB conversion.",
        ) from exc

    geometry = loaded.geometry.values() if hasattr(loaded, "geometry") else [loaded]
    total_vertices = sum(getattr(g, "vertices", []).__len__() for g in geometry)

    if total_vertices == 0:
        raise ValidationError(
            ErrorCode.GLB_VALIDATION_FAILED,
            f"model.glb at {path} parses but contains no geometry.",
            "Check the mesh generation step; the input point cloud or mesh "
            "may have been empty.",
        )
