"""Locate where the signal dies, and test the head layouts the keys permit."""
from __future__ import annotations

import itertools
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

imgs = {}
for k, f in {
    "lateblight": "diseased_tomato_lateblight_1.jpg",
    "earlyblight": "diseased_tomato_earlyblight_1.jpg",
    "healthy1": "healthy_tomato_1.jpg",
    "healthy2": "healthy_tomato_2.jpg",
}.items():
    imgs[k] = cv2.imread(str(ROOT / "tests" / "probe_set" / f))
imgs["black"] = np.zeros((300, 300, 3), dtype="uint8")
imgs["white"] = np.full((300, 300, 3), 255, dtype="uint8")


def run(m, name):
    feats = {}
    h = m.features[-1].register_forward_hook(
        lambda _m, _i, o: feats.__setitem__("last", o.detach().clone())
    )
    outs = {}
    with torch.no_grad():
        for k, arr in imgs.items():
            x = torch.from_numpy(preprocess_image(arr, "imagenet_normalized").astype("float32"))
            if k == "healthy1":
                pooled = m.avgpool(m.features(x))
                pooled_flat = torch.flatten(pooled, 1)
                pre = m.classifier[1](pooled_flat)
                feats["pooled_std"] = float(pooled_flat.std())
                feats["pre_mean"] = float(pre.mean())
                feats["pre_std"] = float(pre.std())
                feats["pre_max"] = float(pre.max())
                feats["pre_frac_pos"] = float((pre > 0).float().mean())
            outs[k] = m(x).numpy().copy()
    h.remove()
    return outs, feats


outs, feats = run(model, "current")
print("=== pooled / pre-ReLU statistics (healthy1) ===")
for k in ("pooled_std", "pre_mean", "pre_std", "pre_max", "pre_frac_pos"):
    print(f"  {k:16s} {feats[k]:+.6f}")

print("\n=== current head [Dropout, Linear, ReLU, Dropout, Linear] ===")
for k, v in outs.items():
    print(f"  {k:12s} {v[0].round(6).tolist()}")
print("  all images identical:", len({tuple(v[0].round(9)) for v in outs.values()}) == 1)

# The checkpoint keys fix params at slots 1 and 4 only, so slots 0,2,3 may be any
# combination of Dropout/ReLU. Test every layout the keys permit.
class Head(nn.Module):
    pass


layouts = {
    "D_L_relu_D_L (current)": lambda: nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
    ),
    "D_L_relu6_D_L": lambda: nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 128), nn.ReLU6(), nn.Dropout(0.2), nn.Linear(128, 2)
    ),
    "D_D_L_relu_L": lambda: nn.Sequential(
        nn.Dropout(0.2), nn.Dropout(0.2), nn.Linear(1280, 128), nn.ReLU(), nn.Linear(128, 2)
    ),
    "D_L_D_relu_L": lambda: nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 128), nn.Dropout(0.2), nn.ReLU(), nn.Linear(128, 2)
    ),
    "D_L_D_D_L (no activation)": lambda: nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 128), nn.Dropout(0.2), nn.Dropout(0.2), nn.Linear(128, 2)
    ),
}

print("\n=== candidate head layouts (only those matching the checkpoint keys) ===")
results = {}
for name, ctor in layouts.items():
    m2 = build_architecture()
    m2.classifier = ctor()
    missing, unexpected = [], []
    try:
        m2.load_state_dict(state, strict=True)
    except Exception as exc:
        msg = str(exc)
        (missing if "Missing key" in msg else unexpected).append(msg[:120])
    m2.load_state_dict(state, strict=False)
    m2.eval()
    o, _ = run(m2, name)
    unique = len({tuple(v[0].round(9)) for v in o.values()})
    logits = np.array([v[0] for v in o.values()])
    spread = float(logits.max(axis=0).max() - logits.max(axis=0).min())
    softmax = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = softmax / softmax.sum(axis=1, keepdims=True)
    results[name] = {
        "strict_load": not missing and not unexpected,
        "unique_outputs": unique,
        "logit_abs_max": float(np.abs(logits).max()),
        "max_prob_range": float(softmax.max(axis=1).max() - softmax.max(axis=1).min()),
        "probs": {k: [round(float(v[0][0]), 4), round(float(v[0][1]), 4)] for k, v in o.items()},
    }
    r = results[name]
    print(f"\n  {name}")
    print(f"    strict_load={r['strict_load']} unique_outputs={r['unique_outputs']}/7 "
          f"logit_absmax={r['logit_abs_max']:.6f} prob_range={r['max_prob_range']:.4f}")
    for k, v in r["probs"].items():
        print(f"      {k:12s} {v}")

print("\n" + "=" * 70)
best = max(results.items(), key=lambda kv: (kv[1]["unique_outputs"], kv[1]["max_prob_range"]))
print("most informative layout:", best[0], "->", {k: v for k, v in best[1].items() if k != "probs"})