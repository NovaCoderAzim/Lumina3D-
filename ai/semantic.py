"""
semantic.py

Central semantic category configuration for DRISHTI-3D. This is the single
source of truth mapping raw model class names -> semantic categories, and
categories -> the human-facing labels used in semantic.json / the demo UI.

Extend SEMANTIC_CATEGORIES if the detection model supports more classes.
Only report classes actually supported by the selected model/pipeline.
"""

from typing import Dict, List

# class_name -> which semantic bucket it belongs to
SEMANTIC_CATEGORIES: Dict[str, List[str]] = {
    "vehicle": ["car", "motorcycle", "bus", "truck"],
    "human": ["person"],
    "environment": ["tree"],
    "structure": ["building"],
}

# category -> plural label shown in semantic.json / demo summary
CATEGORY_LABELS: Dict[str, str] = {
    "structure": "buildings",
    "vehicle": "vehicles",
    "human": "people",
    "environment": "vegetation",
}

_CLASS_TO_CATEGORY: Dict[str, str] = {
    cls: category
    for category, classes in SEMANTIC_CATEGORIES.items()
    for cls in classes
}


def classify(class_name: str) -> str:
    """Map a raw detection class name to a semantic category.

    Falls back to "other" for any class not present in the configuration.
    Never raises ΓÇö an unmapped class must not crash the pipeline.
    """
    return _CLASS_TO_CATEGORY.get(class_name, "other")


def build_summary(tracked_objects) -> Dict[str, int]:
    """Build the semantic.json summary counts from a list of TrackedObject."""
    counts: Dict[str, int] = {}
    for obj in tracked_objects:
        label = CATEGORY_LABELS.get(obj.category, obj.category)
        counts[label] = counts.get(label, 0) + 1
    return counts
