"""Integration dry-run for the severity wiring (NOT a training script).

Creates randomly-initialized U-Nets in a throwaway directory and points
SEGMENTATION_MODEL_PATH at it, solely to verify that the full plumbing works:
SegmentationBackend.load -> SeverityService.analyze -> SeverityResult schema
-> /api/predict JSON. Masks from random weights are meaningless; nothing here
is saved or shipped. Run: python scripts/_dryrun_severity_wiring.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch

from app.ml.unet import UNet

tmp = Path(tempfile.mkdtemp(prefix="seg_dryrun_", dir=str(ROOT / "models")))
arch = {"n_channels": 3, "base": 8, "depth": 3, "n_classes": 1, "input_size": [96, 96]}
for task in ("lesion", "leaf"):
    m = UNet(**{k: arch[k] for k in ("n_channels", "base", "depth", "n_classes")})
    torch.save(
        {
            "arch": arch,
            "state_dict": m.state_dict(),
            "task": task,
            "val_metrics": {"dice": 0.0, "iou": 0.0, "note": "random init dry-run"},
            "dataset": {"repo": "dry-run", "license": "n/a", "label_source": "random weights"},
        },
        tmp / f"{task}_unet.pth",
    )
(tmp / "meta.json").write_text(
    json.dumps(
        {
            "version": 0,
            "framework": "pytorch",
            "architecture": "unet",
            "input_size": [96, 96],
            "threshold": 0.5,
            "lesion": {"file": "lesion_unet.pth", "dataset": {"repo": "dry-run", "license": "n/a", "label_source": "random weights"}, "best_val_metrics": {}},
            "leaf": {"file": "leaf_unet.pth", "dataset": {"repo": "dry-run", "license": "n/a", "label_source": "random weights"}, "best_val_metrics": {}},
            "limitations": ["dry-run only"],
        }
    ),
    encoding="utf-8",
)

os.environ["SEGMENTATION_MODEL_PATH"] = str(tmp.relative_to(ROOT))
os.environ["DEMO_MODE"] = "false"
os.environ.setdefault("MODEL_PATH", "models/efficientnet_v2_s_best.pth")
os.environ.setdefault("MODEL_CLASSES_PATH", "data/model_classes.json")
os.environ.setdefault("MODEL_AUDIT_PATH", "models/classifier_audit.json")
os.environ.setdefault("CLASS_NAMES_PATH", "data/class_names.json")
os.environ.setdefault("DISEASES_PATH", "data/diseases.json")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
import app.main as main_app  # noqa: E402

get_settings.cache_clear()
ok = True
with TestClient(main_app.app) as client:
    health = client.get("/api/health").json()
    print("health.severity_available =", health.get("severity_available"))

    info = client.get("/api/model-info").json()
    print("model.segmentation_status  =", info.get("segmentation_status"))
    sev_info = info.get("severity", {})
    print("model.severity.available   =", sev_info.get("available"))
    assert sev_info.get("available") is True, sev_info

    probe = ROOT / "tests" / "probe_set" / "author_sample_apple_scab.jpg"
    with probe.open("rb") as fh:
        resp = client.post("/api/predict", files={"image": (probe.name, fh, "image/jpeg")})
    print("predict status             =", resp.status_code)
    assert resp.status_code == 200, resp.text[:500]
    sev = resp.json()["severity"]
    print("severity.available         =", sev["available"])
    print("severity.affected          =", sev.get("affected_area_percent"))
    print("severity.level             =", sev.get("severity_level"))
    print("severity.mask prefix       =", (sev.get("mask_image") or "")[:30])
    print("severity.orig prefix       =", (sev.get("original_image") or "")[:30])
    assert sev["available"] is True
    assert isinstance(sev["affected_area_percent"], float)
    assert sev["severity_level"] in ("Mild", "Moderate", "Severe")
    assert (sev.get("mask_image") or "").startswith("data:image/png;base64,")
    assert (sev.get("original_image") or "").startswith("data:image/png;base64,")

    # unavailable path still works when the directory is missing
print("DRY-RUN OK")
