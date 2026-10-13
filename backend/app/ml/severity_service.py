"""Severity assessment service.

Severity is only reported when a real segmentation model produces a real mask.
Classification confidence is never converted into severity, and classical
image-processing heuristics (thresholding, colour clustering, saliency) are
never presented as AI segmentation.

A segmentation model trained by this project is connected when the checkpoints
in ``models/segmentation/`` exist (see ``scripts/train_segmentation.py`` and
``models/segmentation_audit.json``): two U-Nets, one for the lesion region and
one for the leaf region, trained on public Apache-2.0 datasets. The lesion
labels in the training data are SAM2 model pseudo-labels rather than human
annotations, and that limitation is surfaced with every severity result. When
no checkpoint is present, severity is reported as unavailable rather than
estimated.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("plantpath.ml.severity")

UNAVAILABLE_MESSAGE = "Severity assessment is currently unavailable for this model."

MAX_SIDE = 384  # cap embedded preview images


def _encode_png(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _downscale(image, mask_vis):
    """Downscale original + mask overlay to MAX_SIDE for embedding in the API response."""
    import numpy as np
    from PIL import Image

    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1.0:
        return image, mask_vis
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    img = np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))
    msk = np.asarray(Image.fromarray(mask_vis).resize(size, Image.NEAREST))
    return img, msk


class SeverityService:
    """Severity via a real segmentation mask, or an honest unavailable result.

    Integration seam: SEGMENTATION_MODEL_PATH points at models/segmentation/
    (meta.json + two U-Net checkpoints). When it is missing or broken, every
    call reports available = false with the load error as the reason.
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
        base = {
            "name": "Diseased region area percentage",
            "formula": "diseased region area / leaf region area x 100",
            "requires": "A segmentation model producing a lesion mask and a leaf mask",
            "available": self.is_available(),
            "severity_categories": {"mild": "< 10%", "moderate": "10-40%", "severe": ">= 40%"},
        }
        if self.is_available():
            model_info = self._model.info() if hasattr(self._model, "info") else {}
            base.update(
                {
                    "model": model_info,
                    "note": (
                        "Severity is computed from the segmentation model's masks "
                        "(lesion pixels / leaf pixels), never from classifier confidence. "
                        "The U-Nets were trained by this project on public Apache-2.0 "
                        "datasets; the lesion labels are SAM2 model pseudo-labels rather "
                        "than human annotations, and severity values are unvalidated "
                        "estimates for research use. See models/segmentation_audit.json."
                    ),
                }
            )
        else:
            base.update(
                {
                    "note": (
                        "Severity is never derived from classifier confidence. Without a "
                        "loaded segmentation model, no severity value is produced."
                    ),
                    "unavailable_reason": self._error
                    or "No segmentation model is connected; affected area cannot be measured.",
                }
            )
        return base

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

            # Visualisation: lesion area tinted red over a dimmed copy of the original.
            orig = np.asarray(image, dtype=np.uint8) if image.dtype != np.uint8 else image
            dimmed = (orig * 0.45).astype(np.uint8)
            overlay = dimmed.copy()
            overlay[mask] = (
                0.55 * np.array([220, 38, 38], dtype=np.float32)
                + 0.45 * dimmed[mask].astype(np.float32)
            ).astype(np.uint8)
            small_orig, small_overlay = _downscale(orig, overlay)

            return {
                "available": True,
                "message": "",
                "affected_area_percent": round(affected, 1),
                "severity_level": level,
                "method": "diseased region area / leaf region area x 100",
                "original_image": _encode_png(small_orig),
                "mask_image": _encode_png(small_overlay),
                "note": (
                    "Derived from the segmentation model's lesion and leaf masks, "
                    "not from classifier confidence."
                ),
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
