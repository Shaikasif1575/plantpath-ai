"""Segmentation backend for severity assessment.

Loads the two small U-Nets trained by this project (see
``scripts/train_segmentation.py``) and produces real lesion and leaf masks:

* ``models/segmentation/lesion_unet.pth`` - lesion region, trained on
  Apache-2.0 dataset godoldol99/SAM2-based-plant-disease-lesion-segmentation-
  model-DATASET whose lesion masks are SAM2 model pseudo-labels (NOT human
  annotations; see models/segmentation_audit.json).
* ``models/segmentation/leaf_unet.pth`` - leaf vs background, trained on
  Apache-2.0 dataset LeafNet75/Leaf-Disease-Segmentation-masks.

``meta.json`` (written by the training script) records architectures, held-out
metrics and dataset provenance; it is the contract between training and serving.
Both models are real trained networks; no mask is ever produced heuristically.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("plantpath.ml.segmentation")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentationBackend:
    """Loads the project's U-Net checkpoints and produces real masks."""

    def __init__(self) -> None:
        self._lesion = None
        self._leaf = None
        self._meta: Optional[Dict[str, Any]] = None
        self._size = (192, 192)
        self._thr = 0.5
        self._cache_fp: Optional[str] = None
        self._cache_lesion: Optional[np.ndarray] = None
        self._cache_leaf_count: int = 0

    # ------------------------------------------------------------------ load
    def load(self, path: Path) -> None:
        import torch

        from app.ml.unet import build_unet

        p = Path(path)
        if p.is_dir():
            meta_path = p / "meta.json"
        elif p.name == "meta.json":
            meta_path = p
        else:
            raise ValueError(
                "Segmentation model path must be a directory containing meta.json "
                "or the meta.json file itself."
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Segmentation meta.json not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base = meta_path.parent
        models: Dict[str, Any] = {}
        for task in ("lesion", "leaf"):
            entry = meta.get(task)
            if not entry or "file" not in entry:
                raise ValueError(f"{meta_path} has no '{task}' model entry.")
            ckpt_file = base / entry["file"]
            try:
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            except TypeError:  # older torch without weights_only
                ckpt = torch.load(ckpt_file, map_location="cpu")
            model = build_unet(ckpt["arch"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            models[task] = {"model": model, "checkpoint": ckpt}
            logger.info(
                "Loaded %s U-Net from %s (val metrics: %s)",
                task, ckpt_file, ckpt.get("val_metrics"),
            )

        self._lesion = models["lesion"]["model"]
        self._leaf = models["leaf"]["model"]
        self._meta = meta
        self._size = tuple(meta.get("input_size", [192, 192]))  # type: ignore[assignment]
        self._thr = float(meta.get("threshold", 0.5))

    # ------------------------------------------------------------- inference
    def _fingerprint(self, image: np.ndarray) -> str:
        arr = np.ascontiguousarray(image)
        return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()

    def _predict_probs(self, image: np.ndarray):
        """Return (lesion_prob, leaf_prob) at the ORIGINAL image resolution."""
        import torch
        import torch.nn.functional as F

        if self._lesion is None or self._leaf is None:
            raise RuntimeError("No segmentation model is loaded.")

        h, w = image.shape[:2]
        pil_size = (self._size[1], self._size[0])  # (width, height)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(image).resize(pil_size, Image.BILINEAR), dtype=np.float32
        ) / 255.0
        x = (resized - IMAGENET_MEAN) / IMAGENET_STD
        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

        with torch.no_grad():
            lesion_prob = torch.sigmoid(self._lesion(xt))[0, 0]
            leaf_prob = torch.sigmoid(self._leaf(xt))[0, 0]
        # back to original resolution before thresholding (bilinear on probabilities)
        lesion_prob = F.interpolate(
            lesion_prob[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        leaf_prob = F.interpolate(
            leaf_prob[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        return lesion_prob.numpy(), leaf_prob.numpy()

    def _predict(self, image: np.ndarray) -> None:
        """Run both models and cache lesion mask + leaf count for this image."""
        fp = self._fingerprint(image)
        if fp == self._cache_fp:
            return
        lesion_prob, leaf_prob = self._predict_probs(image)
        leaf_mask = leaf_prob > self._thr
        lesion_mask = (lesion_prob > self._thr) & leaf_mask  # lesions must sit on the leaf
        self._cache_fp = fp
        self._cache_lesion = lesion_mask
        self._cache_leaf_count = int(np.count_nonzero(leaf_mask))

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binary lesion mask (bool, original image resolution)."""
        self._predict(image)
        return self._cache_lesion  # type: ignore[return-value]

    def leaf_area(self, image: np.ndarray) -> int:
        """Leaf-region pixel count at original image resolution."""
        self._predict(image)
        return self._cache_leaf_count

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        """Model card for API/UI display (datasets, metrics, honesty notes)."""
        if self._meta is None:
            return {}
        out: Dict[str, Any] = {
            "architecture": "U-Net (2 binary models: lesion + leaf)",
            "framework": "pytorch",
            "input_size": list(self._size),
            "threshold": self._thr,
            "models": {},
        }
        for task in ("lesion", "leaf"):
            entry = self._meta.get(task) or {}
            out["models"][task] = {
                "dataset_repo": (entry.get("dataset") or {}).get("repo"),
                "dataset_license": (entry.get("dataset") or {}).get("license"),
                "label_source": (entry.get("dataset") or {}).get("label_source"),
                "val_metrics": entry.get("best_val_metrics"),
                "train_pairs": entry.get("train_pairs"),
                "val_pairs": entry.get("val_pairs"),
            }
        out["limitations"] = list(self._meta.get("limitations", []))
        return out
