"""In-process smoke test: configure the classifier exactly as main.py does."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.ml.torch_classifier import TorchClassifier  # noqa: E402

def main() -> int:
    c = TorchClassifier()
    c.configure(
        model_path=ROOT / "models" / "efficientnet_v2_s_best.pth",
        classes_path=ROOT / "data" / "model_classes.json",
        preprocessing="imagenet_normalized",
        audit_path=ROOT / "models" / "classifier_audit.json",
    )
    print("loaded:", c.is_loaded())
    print("error:", c.get_error())
    print("n classes:", len(c.get_classes()))
    info = c.get_model_info()
    print("arch:", info["architecture"])
    print("params:", info["parameters"])
    print("source:", info["source"])
    print("gradcam layer:", c.get_gradcam_target_layer())
    if not c.is_loaded():
        return 1

    probe = ROOT / "tests" / "probe_set"
    for name in ("author_sample_apple_scab.jpg", "healthy_tomato_1.jpg",
                 "diseased_tomato_lateblight_1.jpg"):
        arr = np.asarray(Image.open(probe / name).convert("RGB"))
        t = time.time()
        r = c.predict(arr)
        dt = (time.time() - t) * 1000
        top3 = [(p["class_name"], round(p["confidence"], 3)) for p in r["top_predictions"][:3]]
        print(f"{name}: {r['class_name']} p={r['confidence']:.4f} ({dt:.0f}ms) top3={top3}")

    # Grad-CAM path (as used by gradcam_service)
    from app.ml.gradcam_service import GradCamService

    g = GradCamService(c)
    print("gradcam available:", g.is_available())
    arr = np.asarray(Image.open(probe / "author_sample_apple_scab.jpg").convert("RGB"))
    res = g.generate(arr, 0, c.get_classes()[0])
    print("gradcam result:", {k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
                              for k, v in res.items() if k in ("available", "message", "target_layer")})
    if res.get("available"):
        print("overlay prefix:", res["overlay"][:30], "len:", len(res["overlay"]))
    return 0 if res.get("available") else 2


if __name__ == "__main__":
    sys.exit(main())
