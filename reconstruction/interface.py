"""
The ONLY module Person 1 (and anyone else outside this package) should
import from. Everything about COLMAP, PyCOLMAP, Open3D, and trimesh is
an implementation detail hidden behind ReconstructionEngine.process().
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import pipeline
from .mock import run_mock_reconstruction
from .schemas import ReconstructionConfig, ReconstructionResult


class ReconstructionEngine:
    def __init__(
        self,
        config: ReconstructionConfig | None = None,
        projects_base_dir: str | Path = "data/projects",
    ):
        self.config = config or ReconstructionConfig()
        self.projects_base_dir = Path(projects_base_dir)

    def process(
        self,
        frames_directory: str,
        project_id: str,
        output_dir: str | Path | None = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ReconstructionResult:
        """output_dir is derived from project_id, NOT from
        frames_directory's folder depth - guessing the project root by
        counting ".." from an arbitrary frames path is fragile (it
        silently breaks, or worse, collides with an unrelated directory,
        the moment a caller's layout doesn't match exactly). Pass
        output_dir explicitly if your layout differs from
        {projects_base_dir}/{project_id}/reconstruction."""

        resolved_output_dir = (
            Path(output_dir) if output_dir else self.projects_base_dir / project_id / "reconstruction"
        )

        if self.config.use_mock_reconstruction:
            return run_mock_reconstruction(frames_directory, project_id, resolved_output_dir)

        return pipeline.run(
            frames_directory=frames_directory,
            project_id=project_id,
            output_dir=resolved_output_dir,
            config=self.config,
            progress_callback=progress_callback,
        )
