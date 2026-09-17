"""
Engine factory for DRISHTI-3D reconstruction (Section 7).

Dispatches based on DENSE_ENGINE setting:
  AUTO -> genuine dense MVS if available, else authentic sparse-only
  COLMAP_MVS -> requires CUDA COLMAP CLI
  NEURAL_DENSE -> reserved for future neural depth models
  SPARSE_ONLY -> produces sparse SfM directly
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ..schemas import DenseEngineMode, ReconstructionConfig
from .base import DenseResult, SfmResult
from .colmap_dense import ColmapDenseEngine

logger = logging.getLogger("reconstruction.engines.factory")


class DenseEngineController:
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.colmap_dense = ColmapDenseEngine(config)

    def execute_dense_stage(
        self,
        sparse_result: SfmResult,
        images_dir: Path,
        dense_dir: Path,
        cameras_json_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DenseResult:
        mode = self.config.dense_engine.lower()

        if mode == DenseEngineMode.SPARSE_ONLY.value:
            logger.info("Dense engine mode is SPARSE_ONLY; skipping dense MVS.")
            return DenseResult(
                is_available=False,
                dense_cloud_path=None,
                dense_points=0,
                engine_name="sparse_only",
                warnings=[],
                limitations=["Configured for sparse-only reconstruction."],
            )

        if mode in (DenseEngineMode.COLMAP_MVS.value, DenseEngineMode.AUTO.value):
            if self.colmap_dense.is_cuda_available:
                logger.info("Running COLMAP CUDA MVS...")
                return self.colmap_dense.run_dense_mvs(
                    sparse_result, images_dir, dense_dir,
                    cameras_json_path=cameras_json_path,
                    progress_callback=progress_callback,
                )
            
            if mode == DenseEngineMode.COLMAP_MVS.value:
                return DenseResult(
                    is_available=False,
                    dense_cloud_path=None,
                    dense_points=0,
                    engine_name="colmap_cuda_mvs",
                    warnings=["COLMAP_MVS was explicitly requested, but no CUDA COLMAP binary was found."],
                    limitations=["Dense MVS unavailable without CUDA COLMAP CLI."],
                )

            # In AUTO mode with no CUDA COLMAP:
            msg = (
                "Dense MVS requires CUDA-enabled COLMAP. None found on host (set COLMAP_PATH). "
                "Preserving authentic sparse SfM point cloud."
            )
            logger.info(msg)
            return DenseResult(
                is_available=False,
                dense_cloud_path=None,
                dense_points=0,
                engine_name="sparse_fallback_auto",
                warnings=[msg],
                limitations=["Sparse feature points preserved; continuous dense surface was not observed."],
            )

        if mode == DenseEngineMode.NEURAL_DENSE.value:
            return DenseResult(
                is_available=False,
                dense_cloud_path=None,
                dense_points=0,
                engine_name="neural_dense",
                warnings=["Neural dense engine is not yet configured in this environment."],
                limitations=["Neural depth model not installed."],
            )

        # Unknown mode
        return DenseResult(
            is_available=False,
            dense_cloud_path=None,
            dense_points=0,
            engine_name="unknown",
            warnings=[f"Unknown dense engine mode '{mode}'; defaulting to sparse."],
            limitations=[],
        )
