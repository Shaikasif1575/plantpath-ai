"""Grad-CAM explainability service.

Produces a real Grad-CAM heatmap from the loaded PyTorch classifier by
backpropagating the score of the predicted class to the last convolutional
feature map. If no trained model is loaded, or the architecture exposes no
convolutional feature map, Grad-CAM is reported as unavailable. No decorative
or synthetic heatmap is ever returned.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("plantpath.ml.gradcam")

try:
    import torch

    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    logger.warning("PyTorch unavailable for Grad-CAM: %s", exc)

UNAVAILABLE_MESSAGE = "Explainability unavailable for this model."

# Fallback target: last convolutional layer of the legacy MobileNetV2 backbone,
# before global pooling. The connected classifier reports its own last conv
# layer via get_gradcam_target_layer() when available.
TARGET_LAYER = "features.17.conv.1.0"


def _encode_png(arr: np.ndarray) -> str:
    import cv2

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class GradCamService:
    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def is_available(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if not self.classifier.is_loaded():
            return False
        return self._target_module() is not None

    def _target_name(self) -> Optional[str]:
        getter = getattr(self.classifier, "get_gradcam_target_layer", None)
        if callable(getter):
            name = getter()
            if name:
                return name
        return TARGET_LAYER

    def _target_module(self):
        model = self.classifier.get_model()
        if model is None:
            return None
        return dict(model.named_modules()).get(self._target_name())

    def generate(
        self,
        image: np.ndarray,
        class_index: int,
        class_name: str,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"available": False, "message": UNAVAILABLE_MESSAGE}
        try:
            import cv2

            model = self.classifier.get_model()
            target = self._target_module()
            activations: List[Any] = []
            gradients: List[Any] = []

            def forward_hook(_m, _i, output):
                activations.append(output)

            def backward_hook(_m, _g_in, g_out):
                gradients.append(g_out[0])

            fh = target.register_forward_hook(forward_hook)
            bh = target.register_full_backward_hook(backward_hook)
            try:
                model.zero_grad(set_to_none=True)
                x = self.classifier.predict_logits_for_gradcam(image).requires_grad_(True)
                logits = model(x)
                score = logits[0, class_index]
                score.backward()
            finally:
                fh.remove()
                bh.remove()

            if not activations or not gradients:
                return {"available": False, "message": UNAVAILABLE_MESSAGE}

            acts = activations[0].detach()[0]
            grads = gradients[0].detach()[0]
            if acts.ndim == 4:
                acts = acts[0]
            if grads.ndim == 4:
                grads = grads[0]
            weights = grads.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()
            cam_max = float(cam.max())
            if cam_max <= 0:
                return {"available": False, "message": UNAVAILABLE_MESSAGE}
            cam = cam / cam_max

            original = image
            if original.dtype != np.uint8:
                original = np.clip(original, 0, 255).astype(np.uint8)
            h_img, w_img = original.shape[:2]
            cam_up = cv2.resize(cam.astype("float32"), (w_img, h_img))
            colored = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cam_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            overlay = np.clip(
                original.astype("float32") * 0.55 + cam_rgb.astype("float32") * 0.45,
                0,
                255,
            ).astype(np.uint8)

            return {
                "available": True,
                "message": None,
                "original_image": "data:image/png;base64," + _encode_png(original),
                "heatmap": "data:image/png;base64," + _encode_png(cam_rgb),
                "overlay": "data:image/png;base64," + _encode_png(overlay),
                "target_class": class_name,
                "target_layer": self._target_name(),
            }
        except Exception as exc:
            logger.exception("Grad-CAM generation failed")
            return {
                "available": False,
                "message": f"Grad-CAM generation failed: {type(exc).__name__}.",
            }