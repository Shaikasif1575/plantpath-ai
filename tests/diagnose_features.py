"""Is the checkpoint's classifier head itself degenerate, independent of activation?

If the final Linear cannot separate the pooled features for ANY linear readout,
the checkpoint does not carry a usable classifier and no amount of correct
preprocessing can recover a real prediction from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.ml.torch_classifier import build_architecture, preprocess_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
state = torch.load(str(ROOT / "models" / "plant_model.pth"), map_location="cpu", weights_only=True)
model = build_architecture()
model.load_state_dict(state, strict=True)
model.eval()

files = {
    "diseased": [
        "diseased_tomato_lateblight_1.jpg", "diseased_tomato_earlyblight_1.jpg",
        "diseased_tomato_phytophthora_1.jpg", "diseased_tomato_leafmold_1.jpg",
    ],
    "healthy": ["healthy_tomato_1.jpg", "healthy_tomato_2.jpg", "healthy_tomato_3.jpg"],
}

print("=== 1. Pooled 1280-d features: are they even distinguishable? ===")
feats = {}
with torch.no_grad():
    for label, names in files.items():
        for n in names:
            img = cv2.imread(str(ROOT / "tests" / "probe_set" / n))
            for mode in ("imagenet_normalized", "unit_range", "raw_255"):
                x = torch.from_numpy(preprocess_image(img, mode).astype("float32"))
                pooled = torch.flatten(model.avgpool(model.features(x)), 1)
                feats[(label, n, mode)] = pooled[0].numpy()
for mode in ("imagenet_normalized", "unit_range", "raw_255"):
    d = np.stack([feats[("diseased", n, mode)] for n in files["diseased"]])
    h = np.stack([feats[("healthy", n, mode)] for n in files["healthy"]])
    print(f"\n  [{mode}]")
    print(f"    feature norm   diseased={np.linalg.norm(d,axis=1).round(3).tolist()}")
    print(f"                    healthy={np.linalg.norm(h,axis=1).round(3).tolist()}")
    cos = []
    for i in range(len(d)):
        for j in range(len(h)):
            cos.append(
                float(d[i] @ h[j] / (np.linalg.norm(d[i]) * np.linalg.norm(h[j])))
            )
    print(f"    cross-group cosine similarity: min={min(cos):.4f} max={max(cos):.4f} mean={np.mean(cos):.4f}")

print("\n=== 2. Best possible linear probe on the frozen features (oracle upper bound) ===")
X = np.stack([feats[k] for k in feats])
y = np.array([0 if k[0] == "healthy" else 1 for k in feats])
X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
Xtr, ytr = X[:5], y[:5]
Xte, yte = X[5:], y[5:]
# least-squares linear fit on train, evaluate on test
w, *_ = np.linalg.lstsq(Xtr, ytr.astype("float64"), rcond=None)
pred = Xte @ w
pred_lbl = (pred > 0.5).astype(int)
print(f"    oracle train images={len(Xtr)}, held-out images={len(Xte)}")
print(f"    held-out true labels : {yte.tolist()}")
print(f"    held-out predictions : {pred_lbl.tolist()}")
print(f"    held-out raw scores  : {pred.round(4).tolist()}")
print(f"    correct: {int((pred_lbl == yte).sum())}/{len(yte)}")
print("    NOTE: a linear probe fitted on 5 images is an upper bound only; if even")
print("          this cannot separate, the checkpoint carries no usable signal.")

print("\n=== 3. Head weight magnitudes ===")
for n in ("classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"):
    t = state[n]
    print(f"  {n:22s} shape={tuple(t.shape)} std={float(t.std()):.6f} absmax={float(t.abs().max()):.6f}")

print("\n=== 4. Effective signal at the last layer ===")
W = state["classifier.4.weight"].numpy().astype("float64")
mid = {}
with torch.no_grad():
    for label, names in files.items():
        for n in names:
            img = cv2.imread(str(ROOT / "tests" / "probe_set" / n))
            x = torch.from_numpy(preprocess_image(img, "unit_range").astype("float32"))
            pooled = torch.flatten(model.avgpool(model.features(x)), 1)[0].numpy().astype("float64")
            mid[(label, n)] = pooled @ state["classifier.1.weight"].numpy().astype("float64").T + state["classifier.1.bias"].numpy().astype("float64")
allm = np.stack(list(mid.values()))
print(f"  hidden pre-activation: mean={allm.mean():.4f} std={allm.std():.4f} max={allm.max():.4f} min={allm.min():.4f}")
print(f"  fraction strictly positive: {(allm > 0).mean():.6f}")
print(f"  fraction strictly negative: {(allm < 0).mean():.6f}")
print("\n  projected contribution of hidden layer to each logit (per image):")
for k, v in mid.items():
    contrib = W @ v
    print(f"    {k[0]:9s} {k[1][:34]:34s} logit_delta=[{contrib[0]:+.5f}, {contrib[1]:+.5f}]")

print("\n=== 5. Sanity check: does the BACKBONE carry semantic structure? ===")
d = np.stack([feats[("diseased", n, "unit_range")] for n in files["diseased"]])
h = np.stack([feats[("healthy", n, "unit_range")] for n in files["healthy"]])
within_d = np.mean([float(d[i] @ d[j] / (np.linalg.norm(d[i]) * np.linalg.norm(d[j]))) for i in range(len(d)) for j in range(len(d)) if i != j])
within_h = np.mean([float(h[i] @ h[j] / (np.linalg.norm(h[i]) * np.linalg.norm(h[j]))) for i in range(len(h)) for j in range(len(h)) if i != j])
print(f"  within-diseased cosine: {within_d:.4f}")
print(f"  within-healthy cosine : {within_h:.4f}")
print(f"  cross-group cosine    : {np.mean(cos):.4f}" if False else f"  cross-group cosine    : {np.mean([float(d[i] @ h[j] / (np.linalg.norm(d[i]) * np.linalg.norm(h[j]))) for i in range(len(d)) for j in range(len(h))]):.4f}")
print("\n  If within-group >> cross-group, the backbone separates healthy from")
print("  diseased, but the classifier head discards that signal -> checkpoint unusable.")