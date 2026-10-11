"""Real inference with the connected trained classifier.

Primary path: BiernyVR/crop-disease-classifier (Hugging Face, MIT license) —
EfficientNetV2-S, 38 PlantVillage classes. The architecture is rebuilt from
torchvision exactly as in the checkpoint's own models/infer.py, the class
mapping is read from models/classes.json shipped with the checkpoint, and the
weights are loaded verbatim via load_state_dict(strict=True). No training, no
fine-tuning.

Legacy path: the original MobileNetV2 checkpoint (models/plant_model.pth) can
still be rebuilt, but models/classifier_audit.json records its trained head as
degenerate, so the verification gate refuses to serve predictions from it.

A checkpoint is only accepted for live predictions if it clears the
verification gate in models/classifier_audit.json (tests/verify_new_model.py).
A checkpoint that fails verification is reported as unavailable rather than
used to emit meaningless predictions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("plantpath.ml.torch_classifier")

TORCH_AVAILABLE = False
TORCH_ERROR: Optional[str] = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_ERROR = str(exc)
    logger.warning("PyTorch unavailable: %s", exc)


# Audit statuses that permit serving live predictions.
VERIFIED_STATUSES = frozenset({"verified", "verified_working_classifier"})

# (expand_ratio, out_channels, num_blocks, stride) - canonical MobileNetV2
_MBV2_CFG: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)

INPUT_SIZE = (224, 224)

PREPROCESSING_CANDIDATES = {
    "imagenet_normalized": {
        "label": (
            "resize 224x224 (PIL bilinear), scale to [0,1], "
            "ImageNet mean/std normalization"
        ),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "scale": True,
    },
    "unit_range": {
        "label": "resize 224x224, scale to [0,1], no mean/std normalization",
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "scale": True,
    },
    "raw_255": {
        "label": "resize 224x224, pixel values 0-255, no scaling",
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "scale": False,
    },
}

DEFAULT_PREPROCESSING = "imagenet_normalized"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def build_efficientnet_v2_s(num_classes: int):
    """Rebuild the checkpoint's architecture (models/infer.py recipe)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(TORCH_ERROR or "PyTorch is not installed.")
    from torchvision import models as tv_models

    model = tv_models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2_legacy(num_classes: int = 2):
    """Build the legacy MobileNetV2 module tree (original checkpoint)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(TORCH_ERROR or "PyTorch is not installed.")

    class ConvBNReLU(nn.Sequential):
        def __init__(self, inp, oup, kernel=3, stride=1, groups=1):
            pad = (kernel - 1) // 2
            super().__init__(
                nn.Conv2d(inp, oup, kernel, stride, pad, groups=groups, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super().__init__()
            hidden = int(round(inp * expand_ratio))
            layers: List[nn.Module] = []
            if expand_ratio != 1:
                layers.append(ConvBNReLU(inp, hidden, kernel=1))
            layers.append(ConvBNReLU(hidden, hidden, stride=stride, groups=hidden))
            layers.append(nn.Conv2d(hidden, oup, 1, bias=False))
            layers.append(nn.BatchNorm2d(oup))
            self.conv = nn.Sequential(*layers)
            self.use_residual = stride == 1 and inp == oup

        def forward(self, x):
            out = self.conv(x)
            return x + out if self.use_residual else out

    class PlantPathMobileNetV2(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            features = [ConvBNReLU(3, 32, stride=2)]
            inp = 32
            for expand, oup, num_blocks, stride in _MBV2_CFG:
                for i in range(num_blocks):
                    features.append(
                        InvertedResidual(inp, oup, stride if i == 0 else 1, expand)
                    )
                    inp = oup
            features.append(ConvBNReLU(inp, 1280, kernel=1))
            self.features = nn.Sequential(*features)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return PlantPathMobileNetV2(num_classes=num_classes)


# Backwards-compatible alias used by older call sites/tests.
def build_architecture():
    return build_mobilenet_v2_legacy(num_classes=2)


def preprocess_image(image: np.ndarray, mode: str = DEFAULT_PREPROCESSING) -> "np.ndarray":
    """Prepare an RGB uint8 HWC array for the model input.

    Matches the checkpoint's own inference recipe (models/infer.py): PIL
    bilinear resize to 224x224, scale to [0,1], ImageNet mean/std, HWC -> CHW,
    batch dimension added. Channel order is RGB (the input contract of
    app.services.image_validation.load_image_rgb); no channel swap is applied.
    """
    from PIL import Image

    cfg = PREPROCESSING_CANDIDATES.get(mode, PREPROCESSING_CANDIDATES[DEFAULT_PREPROCESSING])
    if image is None:
        raise ValueError("No image data supplied to preprocessing.")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image, mode="RGB")
    pil = pil.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype="float32")
    if cfg["scale"]:
        arr = arr / 255.0
    mean = np.array(cfg["mean"], dtype="float32")
    std = np.array(cfg["std"], dtype="float32")
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]


def _checkpoint_state(ckpt: Any) -> Dict[str, Any]:
    """Unwrap common checkpoint containers to the raw state dict."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            inner = ckpt.get(key)
            if isinstance(inner, dict):
                return inner
    return ckpt if isinstance(ckpt, dict) else {}


def _looks_like_efficientnet(state: Dict[str, Any], ckpt: Any) -> bool:
    """Detect the torchvision EfficientNetV2-S checkpoint layout."""
    if isinstance(ckpt, dict):
        name = str(ckpt.get("model_name", "")).lower()
        if "efficientnet" in name:
            return True
    return any(str(k).startswith("features.1.0.block") for k in state)


class TorchClassifier:
    """Loads the connected trained checkpoint once and serves local inference."""

    def __init__(self) -> None:
        self._model = None
        self._classes: List[str] = []
        self._provenance: Optional[str] = None
        self._preprocessing: str = DEFAULT_PREPROCESSING
        self._model_path: Optional[Path] = None
        self._classes_path: Optional[Path] = None
        self._audit_path: Optional[Path] = None
        self._loaded = False
        self._error: Optional[str] = None
        self._parameters: Optional[int] = None
        self._architecture: Optional[str] = None
        self._source: Optional[str] = None
        self._gradcam_layer: Optional[str] = None

    def configure(
        self,
        model_path: Optional[Path],
        classes_path: Optional[Path],
        preprocessing: Optional[str] = None,
        audit_path: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path
        self._classes_path = classes_path
        self._audit_path = audit_path
        if preprocessing:
            self._preprocessing = preprocessing
        self._load_classes()
        if model_path and Path(model_path).exists():
            self.load(Path(model_path))
        else:
            self._loaded = False
            self._error = f"Model file not found: {model_path}"

    def _load_classes(self) -> None:
        self._classes = []
        self._provenance = None
        if not self._classes_path or not Path(self._classes_path).exists():
            self._error = "class mapping not configured"
            return
        try:
            with Path(self._classes_path).open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self._error = f"class mapping unreadable: {exc}"
            return
        classes = data.get("classes") if isinstance(data, dict) else data
        status = data.get("status") if isinstance(data, dict) else None
        if not isinstance(classes, list) or len(classes) == 0 or status == "unresolved":
            self._error = "class mapping not configured"
            return
        self._classes = [str(c) for c in classes]
        self._provenance = data.get("provenance") if isinstance(data, dict) else None

    def load(self, model_path: Path) -> bool:
        if not TORCH_AVAILABLE:
            self._error = f"PyTorch is not installed: {TORCH_ERROR}"
            self._loaded = False
            return False
        if not self._classes:
            self._loaded = False
            self._error = self._error or "class mapping not configured"
            return False
        try:
            try:
                ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
            except Exception:
                logger.warning("weights_only load failed; retrying with full unpickle.")
                ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            state = _checkpoint_state(ckpt)

            # Integrity: a checkpoint that embeds its own class list must agree
            # with the configured mapping; otherwise labels would be invented.
            if isinstance(ckpt, dict) and isinstance(ckpt.get("classes"), list):
                ckpt_classes = [str(c) for c in ckpt["classes"]]
                if ckpt_classes != self._classes:
                    self._loaded = False
                    self._error = (
                        "class mapping in the configured classes file does not match "
                        "the checkpoint's own embedded class list"
                    )
                    logger.error(self._error)
                    return False

            num_classes = len(self._classes)
            if _looks_like_efficientnet(state, ckpt):
                model = build_efficientnet_v2_s(num_classes)
                cleaned = {}
                for k, v in state.items():
                    key = str(k)
                    if "classifier.1.1." in key:
                        key = key.replace("classifier.1.1.", "classifier.1.")
                    cleaned[key] = v
                model.load_state_dict(cleaned, strict=True)
                architecture = f"EfficientNetV2-S (torchvision) + Linear({num_classes}) head"
                source = (
                    "Hugging Face: BiernyVR/crop-disease-classifier (MIT license)"
                )
            else:
                if num_classes != 2:
                    self._loaded = False
                    self._error = (
                        "legacy checkpoint expects its 2-class mapping, which could not "
                        "be established; refusing to guess labels"
                    )
                    logger.error(self._error)
                    return False
                model = build_mobilenet_v2_legacy(num_classes=num_classes)
                model.load_state_dict(state, strict=True)
                architecture = "MobileNetV2 backbone + custom 2-class head"
                source = None

            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
            self._parameters = int(sum(p.numel() for p in model.parameters()))
            self._architecture = architecture
            self._source = source
            convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
            self._gradcam_layer = convs[-1] if convs else None
            self._error = None
            blocked = self._verification_gate(model_path)
            if blocked:
                self._loaded = False
                self._error = blocked
                logger.error("Checkpoint rejected by verification gate: %s", blocked)
                return False
            self._loaded = True
            logger.info(
                "Loaded %s strictly. parameters=%d classes=%d preprocessing=%s",
                model_path,
                self._parameters,
                len(self._classes),
                self._preprocessing,
            )
            return True
        except Exception as exc:
            self._model = None
            self._loaded = False
            self._error = f"Model loading failed: {exc}"
            logger.exception("Model load error")
            return False

    def _verification_gate(self, model_path: Optional[Path] = None) -> Optional[str]:
        """Refuse to serve predictions from a checkpoint that fails verification.

        The audit records whether the connected checkpoint was shown to
        discriminate known images (liveness) and to classify the model
        author's own in-domain sample correctly. Without that evidence the
        model's output is untrustworthy, and emitting a confident-looking
        class name would be a fabrication.
        """
        audit_path = self._audit_path
        if audit_path is None:
            return None
        path = Path(audit_path)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                audit = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Audit file unreadable (%s); continuing without it.", exc)
            return None
        if not isinstance(audit, dict):
            return None
        # The audit must belong to the checkpoint that is being loaded.
        audit_file = audit.get("model_file")
        if audit_file and model_path is not None and Path(audit_file).name != Path(model_path).name:
            return (
                "checkpoint failed verification: the audit file refers to "
                f"'{audit_file}', not '{Path(model_path).name}'"
            )
        status = audit.get("status")
        if status in (None, *VERIFIED_STATUSES):
            return None
        verdict = audit.get("verdict") or audit.get("status") or "checkpoint failed verification"
        return f"checkpoint failed verification: {verdict}"

    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None and len(self._classes) > 0

    def get_error(self) -> Optional[str]:
        return self._error

    def get_classes(self) -> List[str]:
        return list(self._classes)

    def get_gradcam_target_layer(self) -> Optional[str]:
        """Name of the last convolutional layer (Grad-CAM target)."""
        return self._gradcam_layer

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "PlantPath AI",
            "version": "1.0.0",
            "loaded": self.is_loaded(),
            "model_format": "PyTorch state_dict (.pth)",
            "architecture": self._architecture,
            "framework": "PyTorch" if TORCH_AVAILABLE else "Unavailable",
            "input_size": [INPUT_SIZE[0], INPUT_SIZE[1]],
            "input_channels": 3,
            "num_classes": len(self._classes) if self._classes else None,
            "output_classes": len(self._classes) if self._classes else None,
            "classes": list(self._classes),
            "class_names": list(self._classes),
            "parameters": self._parameters,
            "model_file": str(self._model_path) if self._model_path else None,
            "source": self._source,
            "preprocessing": PREPROCESSING_CANDIDATES.get(
                self._preprocessing, {}
            ).get("label", self._preprocessing),
            "preprocessing_key": self._preprocessing,
            "class_mapping_provenance": self._provenance,
            "error": self._error,
        }

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError(self._error or "Model is not loaded.")
        x = torch.from_numpy(
            preprocess_image(image, self._preprocessing).astype("float32")
        )
        with torch.no_grad():
            logits = self._model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1]
        top = [
            {"class_name": self._classes[int(i)], "confidence": float(probs[int(i)])}
            for i in order
        ]
        return {
            "class_index": int(order[0]),
            "class_name": self._classes[int(order[0])],
            "confidence": float(probs[int(order[0])]),
            "top_predictions": top,
        }

    def predict_logits_for_gradcam(self, image: np.ndarray):
        if not self.is_loaded():
            raise RuntimeError(self._error or "Model is not loaded.")
        return torch.from_numpy(preprocess_image(image, self._preprocessing).astype("float32"))

    def get_model(self):
        return self._model


classifier = TorchClassifier()


def parse_class_name(class_name: str) -> Tuple[str, str, str]:
    """Split a class label into crop / disease / condition.

    Handles both binary labels ("Healthy", "Diseased") and PlantVillage-style
    names (Tomato___Late_blight).
    """
    if "___" in class_name:
        crop_part, disease_part = class_name.split("___", 1)
    else:
        crop_part, disease_part = class_name, class_name
    crop = crop_part.replace(",", "").replace("_", " ").strip().title()
    disease_clean = disease_part.replace("_", " ").strip()
    if disease_clean.lower() == "healthy":
        condition = "healthy"
        disease_label = "Healthy"
    elif disease_clean.lower() in ("diseased", "disease", "unhealthy"):
        condition = "diseased"
        disease_label = "Diseased (unspecified)"
    else:
        condition = "diseased"
        disease_label = disease_clean
    return crop, disease_label, condition
