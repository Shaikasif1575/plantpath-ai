"""Severity assessment service.

Severity is only reported when a real segmentation model produces a real mask.
Classification confidence is never converted into severity, and classical
image-processing heuristics (thresholding, colour clustering, saliency) are
never presented as AI segmentation.

No pretrained segmentation checkpoint is connected. PlantSeg (Wolny et al.,
2020, eLife 9:e57613, MIT licensed) was investigated as the requested
segmentation source. Its shipped checkpoints are 3D U-Nets trained on confocal
and light-sheet microscopy volumes of Arabidopsis tissue, and it segments
individual cell boundaries. That is a different modality (volumetric microscopy,
not RGB leaf photographs) and a different target (cells, not disease lesions),
so those weights cannot be applied to leaf images without retraining. Severity
is therefore reported as unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("plantpath.ml.severity")

UNAVAILABLE_MESSAGE = "Severity assessment is currently unavailable for this model."


class SeverityService:
    """Severity via a real segmentation mask, or an honest unavailable result.

    Integration seam: set SEGMENTATION_MODEL_PATH to a checkpoint that outputs a
    lesion/leaf mask. Until such a checkpoint is supplied and validated, every
    call reports available = false.
    """

    def __init__(self, segmentation_model_path: Optional[Path] = None) -> None:
        self._model_path = segmentation_model_path
        self._model = None
        self._available = False
        self._error: Optional[str] = None
        self._inference = None
        if segmentation_model_path and Path(segmentation_model_path).exists():
            self._load(Path(segmentation_model_path))

    def register_inference(self, callable_) -> None:
        """Attach a real mask-producing callable: (image) -> binary mask array."""
        self._inference = callable_

    def _load(self, path: Path) -> None:
        try:
            from app.ml.segmentation import SegmentationBackend

            backend = SegmentationBackend()
            backend.load(path)
            self._model = backend
            self._inference = backend.segment
            self._available = True
            self._error = None
            logger.info("Segmentation model loaded from %s", path)
        except Exception as exc:
            self._model = None
            self._available = False
            self._error = f"Segmentation model failed to load: {exc}"
            logger.warning(self._error)

    def is_available(self) -> bool:
        return self._available and self._inference is not None

    def get_method_info(self) -> Dict[str, Any]:
        return {
            "name": "Diseased region area percentage",
            "formula": "diseased region area / leaf region area x 100",
            "requires": "A segmentation model producing a lesion mask and a leaf mask",
            "available": self.is_available(),
            "severity_categories": {"mild": "< 10%", "moderate": "10-40%", "severe": ">= 40%"},
            "note": (
                "Severity is never derived from classifier confidence. No segmentation "
                "checkpoint compatible with RGB leaf photographs is connected, so no "
                "severity value is produced."
            ),
            "unavailable_reason": (
                "PlantSeg checkpoints are 3D microscopy cell-segmentation networks and are "
                "not compatible with RGB leaf images. No alternative pretrained lesion "
                "segmentation model is connected."
            ),
        }

    def analyze(self, image: "Any") -> Dict[str, Any]:
        if not self.is_available():
            return {
                "available": False,
                "message": UNAVAILABLE_MESSAGE,
                "reason": self._error
                or "No segmentation model is connected; affected area cannot be measured.",
                "affected_area_percent": None,
                "severity_level": None,
                "method": None,
                "original_image": None,
                "mask_image": None,
            }
        try:
            import cv2
            import numpy as np

            mask = np.asarray(self._inference(image))
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.dtype != bool and mask.max() > 1:
                mask = mask > 0.5
            lesion_pixels = int(np.count_nonzero(mask))
            leaf_pixels = int(self._model.leaf_area(image)) if hasattr(self._model, "leaf_area") else lesion_pixels
            if leaf_pixels <= 0:
                raise RuntimeError("leaf region empty")
            affected = min(100.0, 100.0 * lesion_pixels / leaf_pixels)
            if affected < 10:
                level = "Mild"
            elif affected < 40:
                level = "Moderate"
            else:
                level = "Severe"
            mask_vis = (mask.astype(np.uint8)) * 255
            return {
                "available": True,
                "message": None,
                "affected_area_percent": round(affected, 1),
                "severity_level": level,
                "method": "diseased region area / leaf region area x 100",
                "original_image": None,
                "mask_image": (mask_vis * 255).tolist(),
                "_mask_array": mask_vis,
                "note": "Derived from the segmentation mask, not from classifier confidence.",
            }
        except Exception as exc:
            logger.exception("Severity analysis failed")
            return {
                "available": False,
                "message": UNAVAILABLE_MESSAGE,
                "reason": f"Segmentation failed: {type(exc).__name__}.",
                "affected_area_percent": None,
                "severity_level": None,
                "method": None,
                "original_image": None,
                "mask_image": None,
            }