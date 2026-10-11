"""Empirically determine the trained model's preprocessing and class mapping.

The training notebook is private, so the class mapping cannot be read from
source. It is recovered here as evidence, not assumption: the loaded checkpoint
is run against a probe set of images whose ground truth is documented from their
Wikimedia Commons file-description pages. A preprocessing candidate is accepted
only if it separates healthy from diseased probes consistently.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.torch_classifier import (  # noqa: E402
    PREPROCESSING_CANDIDATES,
    build_architecture,
    preprocess_image,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "plant_model.pth"
PROBE_DIR = ROOT / "tests" / "probe_set"
OUT_REPORT = ROOT / "models" / "calibration_report.json"
OUT_CLASSES = ROOT / "data" / "model_classes.json"
USER_LEAF = PROBE_DIR / "user_leaf_unknown.jpg"

state = torch.load(str(MODEL), map_location="cpu", weights_only=True)
model = build_architecture()
model.load_state_dict(state, strict=True)
model.eval()


def probs_for(img, mode):
    x = torch.from_numpy(preprocess_image(img, mode).astype("float32"))
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return [float(p[0]), float(p[1])]


with open(PROBE_DIR / "sources.json", "r", encoding="utf-8") as fh:
    sources = json.load(fh)

probes = []
for name, meta in sorted(sources.items()):
    path = PROBE_DIR / name
    if not path.exists():
        continue
    img = cv2.imread(str(path))
    if img is None:
        continue
    probes.append({"file": name, "truth": meta["ground_truth"], "shape": list(img.shape)})

matrix = {}
for mode in PREPROCESSING_CANDIDATES:
    rows = []
    for pr in probes:
        img = cv2.imread(str(PROBE_DIR / pr["file"]))
        p = probs_for(img, mode)
        idx = int(np.argmax(p))
        rows.append(
            {
                "file": pr["file"],
                "truth": pr["truth"],
                "p_index_0": round(p[0], 6),
                "p_index_1": round(p[1], 6),
                "argmax_index": idx,
                "confidence": round(max(p), 6),
            }
        )
    user_row = None
    if USER_LEAF.exists():
        img = cv2.imread(str(USER_LEAF))
        p = probs_for(img, mode)
        user_row = {
            "file": USER_LEAF.name,
            "truth": "unknown (user's own test image)",
            "p_index_0": round(p[0], 6),
            "p_index_1": round(p[1], 6),
            "argmax_index": int(np.argmax(p)),
            "confidence": round(max(p), 6),
        }
    matrix[mode] = {"probes": rows, "user_leaf": user_row}

print("=" * 78)
print("PREPROCESSING x PROBE MATRIX")
print("=" * 78)
scores = {}
for mode, res in matrix.items():
    print("\n### " + mode + " -> " + PREPROCESSING_CANDIDATES[mode]["label"])
    print(f"{'file':38s} {'truth':10s} {'p[0]':>9s} {'p[1]':>9s} {'argmax':>7s}")
    for r in res["probes"]:
        print(
            f"{r['file']:38s} {r['truth']:10s} {r['p_index_0']:9.4f} "
            f"{r['p_index_1']:9.4f} {r['argmax_index']:7d}"
        )
    u = res["user_leaf"]
    if u:
        print(
            f"{u['file']:38s} {'unknown':10s} {u['p_index_0']:9.4f} "
            f"{u['p_index_1']:9.4f} {u['argmax_index']:7d}"
        )

    h = [r for r in res["probes"] if r["truth"] == "healthy"]
    d = [r for r in res["probes"] if r["truth"] == "diseased"]
    idx_healthy = [r["argmax_index"] for r in h]
    idx_diseased = [r["argmax_index"] for r in d]
    healthy_on_0 = sum(1 for i in idx_healthy if i == 0) / len(idx_healthy) if idx_healthy else 0
    diseased_on_1 = sum(1 for i in idx_diseased if i == 1) / len(idx_diseased) if idx_diseased else 0
    consistency = (healthy_on_0 + diseased_on_1) / 2
    decisive = [
        r for r in res["probes"] if max(r["p_index_0"], r["p_index_1"]) > 0.70
    ]
    if idx_healthy and idx_diseased and len(set(idx_healthy)) == 1 and len(set(idx_diseased)) == 1 and idx_healthy[0] != idx_diseased[0]:
        margin = min(
            [r["confidence"] for r in h] + [r["confidence"] for r in d]
        )
        accepted = consistency == 1.0 and margin >= 0.55
        mapping = ["Healthy", "Diseased"] if idx_healthy[0] == 0 else ["Diseased", "Healthy"]
    else:
        accepted = False
        margin = min([r["confidence"] for r in res["probes"]] or [0])
        mapping = None
    scores[mode] = {
        "consistency": round(consistency, 4),
        "min_confidence": round(margin, 4),
        "decisive_probes": len(decisive),
        "total_probes": len(res["probes"]),
        "mapping": mapping,
        "accepted": bool(accepted),
    }
    print(
        f"  -> consistency={consistency:.2f} min_conf={margin:.3f} "
        f"mapping={mapping} ACCEPTED={accepted}"
    )

print("\n" + "=" * 78)
accepted = [m for m in scores if scores[m]["accepted"]]
print("accepted candidates:", accepted)
best = None
if accepted:
    best = max(accepted, key=lambda m: (scores[m]["min_confidence"], scores[m]["consistency"]))
    print("chosen:", best)

report = {
    "purpose": "Determine the preprocessing pipeline and recover the class mapping of the existing trained checkpoint.",
    "method": "Each candidate preprocessing was run against a probe set of images whose ground truth is documented from Wikimedia Commons file-description pages. A candidate is accepted only if healthy and diseased probes are separated consistently and decisively.",
    "checkpoint": {
        "path": str(MODEL),
        "architecture": "MobileNetV2 backbone + custom 2-class head [Dropout, Linear(1280,128), ReLU, Dropout, Linear(128,2)]",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "input_size": [224, 224],
        "output_classes": 2,
    },
    "probe_set": probes,
    "matrix": matrix,
    "scores": scores,
    "chosen_preprocessing": best,
    "inferred_classes": scores[best]["mapping"] if best else None,
}
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
with OUT_REPORT.open("w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print("\nwrote", OUT_REPORT)

if best and scores[best]["mapping"]:
    classes_payload = {
        "classes": scores[best]["mapping"],
        "num_classes": len(scores[best]["mapping"]),
        "status": "calibrated",
        "preprocessing": best,
        "preprocessing_description": PREPROCESSING_CANDIDATES[best]["label"],
        "provenance": (
            "Empirically calibrated against a labelled probe set of Wikimedia Commons images "
            "(healthy vs diseased) because the training notebook is private. "
            f"Consistency {scores[best]['consistency']:.2f}, minimum confidence "
            f"{scores[best]['min_confidence']:.3f}. Full evidence: models/calibration_report.json"
        ),
        "evidence_report": "models/calibration_report.json",
    }
    with OUT_CLASSES.open("w", encoding="utf-8") as fh:
        json.dump(classes_payload, fh, indent=2, ensure_ascii=False)
    print("wrote", OUT_CLASSES, "->", classes_payload["classes"])
else:
    print("NO CANDIDATE ACCEPTED - class mapping remains unresolved")