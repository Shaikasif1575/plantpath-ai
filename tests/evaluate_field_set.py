"""Evaluate the connected classifier on the Wikimedia Commons field set.

Runs the *production* request path (POST /api/predict through FastAPI's
TestClient) over every image in data/field_eval/images/ with its verified
label from data/field_eval/sources.json, then computes per-class and overall
metrics. Results are written to:

* models/field_eval_results.json  - per-image predictions + aggregate metrics
* data/model_metrics.json         - project_evaluation block (attributed)

Honesty rules honoured here: metrics come only from this actual run; the
ground truth is the verified uploader description recorded per image; the set
is a small, non-balanced convenience sample of field photographs, and is
labelled as such. No metric is invented and none is copied from anywhere else.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("DEMO_MODE", "false")
os.environ.setdefault("MODEL_PATH", "models/efficientnet_v2_s_best.pth")
os.environ.setdefault("MODEL_CLASSES_PATH", "data/model_classes.json")
os.environ.setdefault("MODEL_AUDIT_PATH", "models/classifier_audit.json")
os.environ.setdefault("CLASS_NAMES_PATH", "data/class_names.json")
os.environ.setdefault("DISEASES_PATH", "data/diseases.json")
os.environ.setdefault("MAX_UPLOAD_SIZE_MB", "10")

EVAL_DIR = ROOT / "data" / "field_eval"
IMG_DIR = EVAL_DIR / "images"


def main() -> None:
    sources_path = EVAL_DIR / "sources.json"
    if not sources_path.exists():
        raise SystemExit("data/field_eval/sources.json missing; run tests/fetch_field_eval_set.py first.")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    images = sorted(p for p in IMG_DIR.glob("*.jpg") if p.name in sources)
    if not images:
        raise SystemExit("No field-eval images found; run tests/fetch_field_eval_set.py first.")
    print(f"Evaluating {len(images)} field images...")

    from fastapi.testclient import TestClient

    from app.config import get_settings
    import app.main as main_app

    get_settings.cache_clear()
    os.environ["DEMO_MODE"] = "false"
    with TestClient(main_app.app) as client:  # `with` runs lifespan startup -> loads the model
        _run(client, sources, images)


def _run(client, sources, images) -> None:
    health = client.get("/api/health").json()
    print(f"model_loaded={health.get('model_loaded')} demo_mode={health.get('demo_mode')}")
    if not health.get("model_loaded"):
        raise SystemExit("Classifier not loaded; cannot evaluate.")

    per_image = []
    cm: dict[str, Counter] = defaultdict(Counter)
    correct = crop_correct = low_conf = 0
    confs = []

    for path in images:
        gt = sources[path.name]["ground_truth"]
        with path.open("rb") as fh:
            resp = client.post("/api/predict", files={"image": (path.name, fh, "image/jpeg")})
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {path.name}")
            per_image.append({"image": path.name, "ground_truth": gt, "error": resp.status_code})
            continue
        data = resp.json()
        pred = data["top_prediction"]["class_name"]
        prob = data["top_prediction"]["probability"]
        lq = bool(data.get("low_confidence"))
        gt_crop, pred_crop = gt.split("___")[0], pred.split("___")[0]
        hit = pred == gt
        correct += int(hit)
        crop_correct += int(gt_crop == pred_crop)
        low_conf += int(lq)
        confs.append(prob)
        cm[gt][pred] += 1
        per_image.append(
            {
                "image": path.name,
                "ground_truth": gt,
                "prediction": pred,
                "probability": round(prob, 4),
                "correct": hit,
                "crop_correct": gt_crop == pred_crop,
                "low_confidence": lq,
                "severity_available": data.get("severity", {}).get("available"),
                "commons_page": sources[path.name].get("commons_page"),
                "license": sources[path.name].get("license"),
            }
        )
        flag = "OK " if hit else "MISS"
        print(f"  {flag} {path.name:52s} gt={gt:34s} pred={pred:34s} p={prob:.3f}{' LQ' if lq else ''}")

    n = len(per_image)
    # Per-class metrics over classes present in ground truth
    per_class = {}
    gt_counts = Counter(r["ground_truth"] for r in per_image if "prediction" in r)
    pred_counts = Counter(r["prediction"] for r in per_image if "prediction" in r)
    for cls in sorted(gt_counts):
        tp = sum(1 for r in per_image if r.get("ground_truth") == cls and r.get("prediction") == cls)
        support = gt_counts[cls]
        precision = tp / pred_counts[cls] if pred_counts[cls] else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "support": support,
            "predicted_as_class": pred_counts[cls],
            "correct": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    evaluated = [r for r in per_image if "prediction" in r]
    overall = {
        "n_images": len(evaluated),
        "n_ground_truth_classes": len(gt_counts),
        "exact_class_accuracy": round(correct / len(evaluated), 4) if evaluated else None,
        "crop_accuracy": round(crop_correct / len(evaluated), 4) if evaluated else None,
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "low_confidence_rate": round(low_conf / len(evaluated), 4) if evaluated else None,
        "per_class": per_class,
        "confusion": {gt: dict(c) for gt, c in sorted(cm.items())},
    }

    stamp = datetime.now(timezone.utc).isoformat()
    results = {
        "evaluated_at": stamp,
        "protocol": {
            "fetch_script": "tests/fetch_field_eval_set.py",
            "eval_script": "tests/evaluate_field_set.py",
            "protocol_version": 2,
            "label_source": "Wikimedia Commons uploader descriptions, verified programmatically "
                            "by keyword match on title+description (see data/field_eval/sources.json)",
            "inclusion_rule": (
                "crop keyword + disease keyword in title/description; leaf-tissue keyword "
                "required (leaf/laub/blad/...); electron-microscopy and non-leaf imagery rejected"
            ),
            "request_path": "POST /api/predict via FastAPI TestClient (production code path)",
            "image_count": len(evaluated),
        },
        "metrics": overall,
        "per_image": per_image,
    }
    out_path = ROOT / "models" / "field_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # Merge attributed summary into data/model_metrics.json
    metrics_path = ROOT / "data" / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["project_evaluation"] = {
        **metrics.get("project_evaluation", {}),
        "field_evaluation": {
            "evaluated_at": stamp,
            "protocol": results["protocol"],
            "n_images": overall["n_images"],
            "n_classes": overall["n_ground_truth_classes"],
            "exact_class_accuracy": overall["exact_class_accuracy"],
            "crop_accuracy": overall["crop_accuracy"],
            "mean_confidence": overall["mean_confidence"],
            "low_confidence_rate": overall["low_confidence_rate"],
            "per_class": overall["per_class"],
            "results_file": "models/field_eval_results.json",
            "note": "Measured by this project on verified-label Wikimedia Commons field "
                    "photographs (out-of-domain versus PlantVillage-style training data). "
                    "Small, non-balanced convenience sample; an informal robustness probe, "
                    "not a benchmark. Labels are uploader descriptions, verified by keyword "
                    "match and listed in data/field_eval/sources.json.",
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"updated {metrics_path}")
    print(
        f"\nexact={overall['exact_class_accuracy']} crop={overall['crop_accuracy']} "
        f"mean_p={overall['mean_confidence']} low_conf_rate={overall['low_confidence_rate']}"
    )


if __name__ == "__main__":
    main()
