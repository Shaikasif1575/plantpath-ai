"""Segmentation backend: the plug-in point for lesion/leaf segmentation.

No pretrained lesion-segmentation checkpoint compatible with RGB leaf
photographs has been identified as both technically and legally usable for this
project, so this module intentionally ships no weights and never fabricates a
mask.

Investigation record
--------------------
PlantSeg / PanSeg (Wolny et al., 2020, eLife 9:e57613; MIT licensed; source
https://github.com/kreshuklab/panseg, formerly https://github.com/hci-unihd/plant-seg)
does ship pretrained checkpoints, downloaded on demand into ~/.plantseg_models.
They are 3D residual U-Nets (pytorch-3dunet) trained on confocal and light-sheet
microscopy volumes of Arabidopsis thaliana ovules and lateral roots, and they
segment individual cell boundaries in 3D volumes.

Why those checkpoints cannot be used here:
  * Modality mismatch: volumetric microscopy, not 2D RGB leaf photographs.
  * Target mismatch: cell instance boundaries, not diseased leaf lesions.
  * Using them would require retraining on leaf imagery, which is out of scope.

To enable severity assessment, supply a checkpoint that maps a leaf RGB image to
a lesion probability mask, implement ``segment`` and ``leaf_area`` below, and set
SEGMENTATION_MODEL_PATH so SeverityService loads it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("plantpath.ml.segmentation")


class SegmentationBackend:
    """Loads a lesion-segmentation checkpoint and produces real masks."""

    def __init__(self) -> None:
        self._model = None
        self._path: Optional[Path] = None
        self._input_size = (256, 256)

    def load(self, path: Path) -> None:
        raise NotImplementedError(
            "No lesion-segmentation checkpoint is connected. PlantSeg/PanSeg "
            "checkpoints are 3D microscopy cell-segmentation networks and are not "
            "compatible with RGB leaf images. Supply a lesion-segmentation model to "
            "enable severity assessment."
        )

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("No segmentation model is loaded.")

    def leaf_area(self, image: np.ndarray) -> int:
        raise NotImplementedError("No segmentation model is loaded.")