"""Test every parameterless activation the checkpoint keys permit in the head.

The keys fix parameters at classifier slots 1 and 4 only, so slots 0/2/3 may be
any non-parametric module. ReLU kills the signal (pre-activations are entirely
negative), so the actual activation must be identified empirically.
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

imgs = {}
for k, f in {
    "lb_tomato": "diseased_tomato_lateblight_1.jpg",
    "eb_tomato": "diseased_tomato_earlyblight_1.jpg",
    "pb_tomato": "diseased_tomato_phytophthora_1.jpg",
    "lm_tomato": "diseased_tomato_leafmold_1.jpg",
    "h_tomato1": "healthy_tomato_1.jpg",
    "h_tomato2": "healthy_tomato_2.jpg",
    "h_tomato3": "healthy_tomato_3.jpg",
}.items():
    imgs[k] = cv2.imread(str(ROOT / "tests" / "probe_set" / f))
imgs["black"] = np.zeros((300, 300, 3), dtype="uint8")
imgs["white"] = np.full((300, 300, 3), 255, dtype="uint8")
imgs["user"] = cv2.imread(str(ROOT / "tests" / "probe_set" / "user_leaf_unknown.jpg"))

ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "ReLU6": nn.ReLU6(),
    "LeakyReLU(0.01)": nn.LeakyReLU(0.01),
    "LeakyReLU(0.2)": nn.LeakyReLU(0.2),
    "ELU": nn.ELU(),
    "SELU": nn.SELU(),
    "CELU": nn.CELU(),
    "GELU": nn.GELU(),
    "SiLU": nn.SiLU(),
    "Mish": nn.Mish(),
    "Softplus": nn.Softplus(),
    "Softsign": nn.Softsign(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "Identity": nn.Identity(),
    "PReLU": nn.PReLU(),
    "Hardtanh": nn.Hardtanh(),
    "Softshrink": nn.Softshrink(),
    "Tanhshrink": nn.Tanhshrink(),
    "Threshold": nn.Threshold(0.0, 0.0),
}


def evaluate(act):
    m = build_architecture()
    m.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 128), act, nn.Dropout(0.2), nn.Linear(128, 2)
    )
    m.load_state_dict(state, strict=True)
    m.eval()
    rows, pre_info = {}, None
    with torch.no_grad():
        for k, arr in imgs.items():
            x = torch.from_numpy(preprocess_image(arr, "imagenet_normalized").astype("float32"))
            if k == "h_tomato1":
                pooled = torch.flatten(m.avgpool(m.features(x)), 1)
                pre = m.classifier[1](pooled)
                pre_info = (
                    float(pre.mean()), float(pre.std()), float(pre.max()),
                    float((pre > 0).float().mean()),
                )
            logits = m(x)
            p = torch.softmax(logits, dim=1)[0].numpy()
            rows[k] = (float(p[0]), float(p[1]))
    return rows, pre_info


print(f"{'activation':20s} {'unique':>7s} {'sep':>7s} {'minconf':>8s}  verdict")
print("-" * 78)

healthy_keys = ["h_tomato1", "h_tomato2", "h_tomato3"]
diseased_keys = ["lb_tomato", "eb_tomato", "pb_tomato", "lm_tomato"]

results = {}
for name, act in ACTIVATIONS.items():
    rows, pre = evaluate(act)
    unique = len({round(v[1], 6) for v in rows.values()})
    idx_h = [int(np.argmax(rows[k])) for k in healthy_keys]
    idx_d = [int(np.argmax(rows[k])) for k in diseased_keys]
    separated = len(set(idx_h)) == 1 and len(set(idx_d)) == 1 and idx_h[0] != idx_d[0]
    minconf = min(max(rows[k]) for k in healthy_keys + diseased_keys)
    sep_val = 1.0 if separated else 0.0
    verdict = ""
    if separated and minconf >= 0.60:
        verdict = "USABLE -> index0=" + ("Healthy" if idx_h[0] == 0 else "Diseased")
    elif separated:
        verdict = "separates but not decisive"
    elif unique <= 2:
        verdict = "DEAD (constant output)"
    else:
        verdict = "no separation"
    results[name] = {
        "unique": unique, "separated": separated, "min_conf": minconf,
        "healthy_index": idx_h[0] if len(set(idx_h)) == 1 else None,
        "diseased_index": idx_d[0] if len(set(idx_d)) == 1 else None,
        "verdict": verdict, "probs": {k: [round(v[0], 4), round(v[1], 4)] for k, v in rows.items()},
        "pre_activation": pre,
    }
    print(f"{name:20s} {unique:7d} {sep_val:7.1f} {minconf:8.4f}  {verdict}")

print("\n" + "=" * 78)
print("pre-ReLU-activation stats (healthy_tomato_1): mean / std / max / frac_positive")
print(" ", results["ReLU"]["pre_activation"])

usable = [n for n, r in results.items() if r["verdict"].startswith("USABLE")]
print("usable activations:", usable)
for n in usable:
    r = results[n]
    print(f"\n  {n}: healthy->index{r['healthy_index']}, diseased->index{r['diseased_index']}")
    for k, v in r["probs"].items():
        print(f"    {k:12s} {v}")