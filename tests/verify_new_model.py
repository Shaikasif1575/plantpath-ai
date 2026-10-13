"""Verify the BiernyVR EfficientNetV2-S checkpoint (two-tier honest gate).

Tier 1 — liveness: the checkpoint must discriminate inputs (this is what the
         old model failed: constant output for every image).
Tier 2 — in-domain: the model author's own sample_leaf.jpg must classify as
         Apple___Apple_scab with high confidence.

Field-photo probe accuracy (Wikimedia images, out of PlantVillage domain) is
MEASURED and recorded as an honest limitation — never hidden, never used to
fake a benchmark number.

Writes models/classifier_audit.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "efficientnet_v2_s_best.pth"
CLASSES_JSON = ROOT / "models" / "classes.json"
PROBE = ROOT / "tests" / "probe_set"
OUT = ROOT / "models" / "classifier_audit.json"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

meta = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
classes = meta["classes"]
mean = meta["normalisation"]["mean"]
std = meta["normalisation"]["std"]
size = meta.get("image_size", 224)

FIELD_TRUTH = {
    "healthy_tomato_1.jpg": "healthy", "healthy_tomato_2.jpg": "healthy",
    "healthy_tomato_3.jpg": "healthy",
    "diseased_tomato_earlyblight_1.jpg": "diseased",
    "diseased_tomato_lateblight_1.jpg": "diseased",
    "diseased_tomato_leafmold_1.jpg": "diseased",
    "diseased_tomato_phytophthora_1.jpg": "diseased",
}


def build_model(n_classes: int):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("classifier.1.1.", "classifier.1.")] = v
    try:
        model.load_state_dict(cleaned, strict=True)
        variant = "torchvision efficientnet_v2_s + Linear(38) head (strict load)"
    except Exception:
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.0), nn.Linear(in_features, n_classes)
        )
        model.load_state_dict(cleaned, strict=False)
        variant = "torchvision efficientnet_v2_s + Dropout+Linear head (non-strict load)"
    model.eval()
    return model, variant


def preprocess(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = (arr - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    arr = arr.transpose(2, 0, 1).astype("float32")
    return torch.from_numpy(arr).unsqueeze(0)


def predict(model, path: Path):
    x = preprocess(path)
    t = time.time()
    with torch.no_grad():
        logits = model(x)
    ms = (time.time() - t) * 1000
    probs = torch.softmax(logits, dim=1)[0]
    top = int(probs.argmax())
    return classes[top], float(probs[top]), round(ms, 1)


def main() -> int:
    t0 = time.time()
    model, variant = build_model(len(classes))
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s | {variant}")

    # ---- Tier 2: author's own in-domain sample ----
    sample = PROBE / "author_sample_apple_scab.jpg"
    sample_result = None
    sample_ok = False
    if sample.exists():
        cls, p, ms = predict(model, sample)
        sample_ok = cls == "Apple___Apple_scab" and p > 0.8
        sample_result = {"file": sample.name, "expected": "Apple___Apple_scab",
                         "predicted": cls, "probability": round(p, 4),
                         "inference_ms": ms, "pass": sample_ok}
        print(f"\nauthor sample: {cls} p={p:.3f} -> {'PASS' if sample_ok else 'FAIL'}")

    # ---- Field probes (measured, reported honestly) ----
    files = sorted(p for p in PROBE.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                   and p.name != sample.name)
    results, correct, labeled = [], 0, 0
    confidences = []
    print(f"\n{'file':44s} {'truth':9s} {'top class':46s} {'p':>6s}")
    for p in files:
        cls, top_p, ms = predict(model, p)
        label = FIELD_TRUTH.get(p.name, "unknown")
        is_healthy = cls.endswith("___healthy")
        ok = ""
        if label != "unknown":
            labeled += 1
            if (label == "healthy") == is_healthy:
                correct += 1
                ok = "OK"
        confidences.append(top_p)
        print(f"{p.name:44s} {label:9s} {cls[:44]:46s} {top_p:6.3f} {ok} ({ms:.0f}ms)")
        results.append({"file": p.name, "truth": label, "top_class": cls,
                        "top_probability": round(top_p, 6),
                        "healthy_vs_diseased_correct": ok == "OK",
                        "inference_ms": ms})

    field_acc = correct / labeled if labeled else 0.0
    distinct = len({r["top_class"] for r in results})
    conf_min, conf_max = (min(confidences), max(confidences)) if confidences else (0, 0)

    # ---- Tier 1: liveness gate (the old model failed this) ----
    liveness_ok = (
        distinct >= 3
        and (conf_max - conf_min) >= 0.2
        and conf_min > 0.4
        and len(confidences) >= 5
    )
    gate_ok = liveness_ok and sample_ok

    print(f"\nTier 1 liveness: distinct={distinct} spread={conf_max-conf_min:.3f} "
          f"min_p={conf_min:.3f} -> {'PASS' if liveness_ok else 'FAIL'}")
    print(f"Tier 2 in-domain sample -> {'PASS' if sample_ok else 'FAIL'}")
    print(f"Field-photo probe eval (out-of-domain, informational): "
          f"{correct}/{labeled} = {field_acc:.0%}")

    audit = {
        "status": "verified_working_classifier" if gate_ok else "failed_verification",
        "model_file": MODEL.name,
        "architecture": "torchvision efficientnet_v2_s + Linear(38) head",
        "load_variant": variant,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "source": {
            "repo": "https://huggingface.co/BiernyVR/crop-disease-classifier",
            "license": "MIT",
            "reported_val_accuracy": meta.get("val_accuracy"),
            "reported_metric_provenance": (
                "model card / classes.json shipped with the checkpoint "
                "(author's self-reported training output)"
            ),
        },
        "classes_provenance": (
            "models/classes.json shipped with the checkpoint "
            "(authoritative mapping, not guessed)"
        ),
        "preprocessing": {
            "size": size, "interpolation": "bilinear",
            "scale": "0-1 then ImageNet mean/std", "mean": mean, "std": std,
            "provenance": "models/infer.py shipped with the checkpoint",
        },
        "gates": {
            "liveness": {"pass": liveness_ok,
                         "criterion": ">=3 distinct top classes, >=0.2 confidence spread, min prob >0.4",
                         "distinct_top_classes": distinct,
                         "confidence_min": round(conf_min, 4),
                         "confidence_max": round(conf_max, 4)},
            "in_domain_sample": {"pass": sample_ok, **(sample_result or {})},
        },
        "field_photo_probe_evaluation": {
            "note": (
                "Informal sanity check on 7 Wikimedia Commons field photographs — "
                "out of the PlantVillage training domain (lab photos, plain "
                "backgrounds). Small sample; NOT a benchmark. Reported for honesty."
            ),
            "label_separation_accuracy": round(field_acc, 4),
            "correct": correct, "labeled": labeled,
        },
        "probe_results": results,
        "known_limitations": [
            "Trained on PlantVillage laboratory photographs; field-photo generalisation is not validated.",
            "Severity comes from a project-trained segmentation model (U-Net pair) "
            "with SAM2 pseudo-label lesion masks — unvalidated research estimates, "
            "never derived from classifier confidence.",
            "Metrics shown from model card are author self-reported validation numbers.",
        ],
        "load_seconds": round(load_s, 2),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nGATE: {'PASS' if gate_ok else 'FAIL'} -> {OUT}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
