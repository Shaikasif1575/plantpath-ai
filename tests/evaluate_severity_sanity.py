"""Directional field sanity check for the severity segmentation model.

Runs the production request path (POST /api/predict through FastAPI's
TestClient) over the verified-label Wikimedia Commons field set and groups
results by the ground-truth *condition* recorded per image: leaves labelled
healthy versus leaves labelled diseased. A segmentation-based severity
estimate should, on average, assign diseased leaves more affected area than
healthy ones. This is a *known-groups* check of direction, not severity
accuracy: the field set carries disease labels, not annotated severity, and
the U-Net's lesion labels are SAM2 pseudo-labels.

Results are written to:

* models/severity_sanity_results.json - per-image values + group aggregates
* data/model_metrics.json             - project_evaluation.severity_sanity
                                        block (attributed, merged without
                                        touching other evaluation blocks)

No metric is invented and none is copied from anywhere else.
"""
from __future__ import annotations

import json
import os
import statistics
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
OUT_JSON = ROOT / "models" / "severity_sanity_results.json"
METRICS = ROOT / "data" / "model_metrics.json"

NOTE = (
    "Known-groups directional sanity check, not severity accuracy: the field "
    "set's ground truth is disease presence from uploader descriptions, not "
    "annotated severity; lesion training labels are SAM2 pseudo-labels."
)


def main() -> None:
    sources_path = EVAL_DIR / "sources.json"
    if not sources_path.exists():
        raise SystemExit("data/field_eval/sources.json missing; run tests/fetch_field_eval_set.py first.")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    images = sorted(p for p in IMG_DIR.glob("*.jpg") if p.name in sources)
    if not images:
        raise SystemExit("No field-eval images found; run tests/fetch_field_eval_set.py first.")
    print(f"Severity sanity check over {len(images)} field images...")

    from fastapi.testclient import TestClient

    from app.config import get_settings
    import app.main as main_app

    get_settings.cache_clear()
    os.environ["DEMO_MODE"] = "false"
    with TestClient(main_app.app) as client:  # `with` runs lifespan startup -> loads the models
        _run(client, sources, images)


def _run(client, sources, images) -> None:
    health = client.get("/api/health").json()
    print(f"model_loaded={health.get('model_loaded')} severity_available={health.get('severity_available')}")
    if not health.get("severity_available"):
        raise SystemExit("Segmentation model not loaded; cannot run the severity sanity check.")

    per_image = []
    groups: dict[str, list[float]] = defaultdict(list)
    per_class: dict[str, list[float]] = defaultdict(list)
    levels: dict[str, Counter] = {"healthy": Counter(), "diseased": Counter()}
    unavailable = 0

    for path in images:
        gt = sources[path.name]["ground_truth"]
        group = "healthy" if gt.endswith("___healthy") else "diseased"
        r = client.post("/api/predict", files={"image": (path.name, path.read_bytes(), "image/jpeg")})
        if r.status_code != 200:
            print(f"  ERROR {path.name}: HTTP {r.status_code}")
            unavailable += 1
            continue
        sev = r.json().get("severity") or {}
        row = {
            "image": path.name,
            "ground_truth": gt,
            "group": group,
            "available": sev.get("available") is True,
            "affected_area_percent": sev.get("affected_area_percent"),
            "severity_level": sev.get("severity_level"),
            "reason": sev.get("reason") if sev.get("available") is not True else None,
        }
        per_image.append(row)
        if not row["available"]:
            unavailable += 1
            print(f"  UNAVAILABLE {path.name}: {row['reason']}")
            continue
        pct = float(row["affected_area_percent"])
        groups[group].append(pct)
        per_class[gt].append(pct)
        levels[group][row["severity_level"]] += 1
        print(f"  {group:8s} {pct:6.1f}% {row['severity_level']:8s} {path.name}")

    def agg(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "mean": round(statistics.fmean(vals), 2) if vals else None,
            "median": round(statistics.median(vals), 2) if vals else None,
            "min": round(min(vals), 1) if vals else None,
            "max": round(max(vals), 1) if vals else None,
        }

    healthy, diseased = agg(groups["healthy"]), agg(groups["diseased"])
    directional = (
        healthy["mean"] is not None
        and diseased["mean"] is not None
        and diseased["mean"] > healthy["mean"]
    )
    result = {
        "protocol": "known-groups directional check on the verified-label field set",
        "note": NOTE,
        "n_images": len(images),
        "n_available": len(per_image) - unavailable,
        "n_unavailable": unavailable,
        "groups": {
            "healthy": {**healthy, "levels": dict(levels["healthy"])},
            "diseased": {**diseased, "levels": dict(levels["diseased"])},
        },
        "diseased_mean_exceeds_healthy_mean": directional,
        "per_class_mean_percent": {
            cls: round(statistics.fmean(v), 2) for cls, v in sorted(per_class.items())
        },
        "per_image": per_image,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    metrics.setdefault("project_evaluation", {})["severity_sanity"] = {
        "n_images": result["n_images"],
        "n_available": result["n_available"],
        "healthy_mean_percent": healthy["mean"],
        "diseased_mean_percent": diseased["mean"],
        "diseased_mean_exceeds_healthy_mean": directional,
        "note": NOTE,
        "results_file": "models/severity_sanity_results.json",
        "run_at": result["run_at"],
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nhealthy  n={healthy['n']} mean={healthy['mean']}% median={healthy['median']}%")
    print(f"diseased n={diseased['n']} mean={diseased['mean']}% median={diseased['median']}%")
    print(f"diseased mean exceeds healthy mean: {directional}")
    print(f"coverage: {result['n_available']}/{result['n_images']} produced severity")
    print(f"wrote {OUT_JSON}")
    print(f"updated {METRICS} (project_evaluation.severity_sanity)")


if __name__ == "__main__":
    main()
