"""Backend tests for PlantPath AI.

These tests assert honest behaviour in both live mode (DEMO_MODE=false with a
connected verified model) and demo mode. They never assert a fabricated
prediction: when no verified model is connected, live prediction must fail
cleanly rather than return a made-up disease.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("DEMO_MODE", "false")
os.environ.setdefault("MODEL_PATH", "models/efficientnet_v2_s_best.pth")
os.environ.setdefault("MODEL_CLASSES_PATH", "data/model_classes.json")
os.environ.setdefault("MODEL_AUDIT_PATH", "models/classifier_audit.json")
os.environ.setdefault("CLASS_NAMES_PATH", "data/class_names.json")
os.environ.setdefault("DISEASES_PATH", "data/diseases.json")
os.environ.setdefault("MAX_UPLOAD_SIZE_MB", "10")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

# Probe image with a known in-domain label (model author's own sample).
PROBE_IMAGE = ROOT / "tests" / "probe_set" / "author_sample_apple_scab.jpg"


def _make_client(demo_mode: str):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    import app.main as main

    get_settings.cache_clear()
    os.environ["DEMO_MODE"] = demo_mode
    return TestClient(main.app)


@pytest.fixture(scope="session")
def client_live():
    with _make_client("false") as c:
        yield c


@pytest.fixture(scope="session")
def client_demo():
    with _make_client("true") as c:
        yield c


def _make_png(width: int = 224, height: int = 224, color=(34, 139, 34)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 224, height: int = 224, color=(34, 139, 34)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _model_is_loaded(client) -> bool:
    return bool(client.get("/api/model-info").json().get("loaded"))


class TestHealth:
    def test_health_ok(self, client_live):
        r = client_live.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert data["demo_mode"] is False
        assert "timestamp" in data

    def test_root(self, client_live):
        r = client_live.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "PlantPath AI API"


class TestModelInfo:
    def test_model_info_contract(self, client_live):
        r = client_live.get("/api/model-info")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "PlantPath AI"
        assert "loaded" in data and "demo_mode" in data
        assert data["demo_mode"] is False
        for field in (
            "model_format", "architecture", "input_size", "num_classes",
            "classes", "preprocessing", "gradcam_available",
            "segmentation_available", "segmentation_status",
            "low_confidence_threshold", "metrics", "training",
        ):
            assert field in data, f"missing {field}"

    def test_severity_never_claims_available_without_model(self, client_live):
        data = client_live.get("/api/model-info").json()
        sev = data.get("severity", {})
        assert sev.get("available") is False
        assert "confidence" in (sev.get("note") or "").lower()

    def test_metrics_are_sourced_not_fabricated(self, client_live):
        """Metrics may only appear with an attribution note; the rest stay null."""
        data = client_live.get("/api/model-info").json()
        metrics = data["metrics"]
        training = data["training"]
        if metrics.get("accuracy") is not None:
            assert metrics.get("note"), "populated accuracy must carry an attribution note"
        for key in ("precision", "recall", "f1"):
            assert metrics.get(key) is None, f"{key} must not be invented"
        for key in ("training_accuracy", "training_loss", "validation_loss"):
            assert training.get(key) is None, f"{key} must not be invented"
        if training.get("validation_accuracy") is not None:
            assert training.get("note"), "populated validation accuracy must be attributed"

    def test_crops_reference_library(self, client_live):
        r = client_live.get("/api/crops")
        assert r.status_code == 200
        data = r.json()
        # The supported set must mirror the real model class list exactly.
        assert data["supported_classes"] == client_live.get("/api/model-info").json()["classes"]
        # The reference library is educational and always populated.
        assert data["reference_library"]["total"] > 0
        assert "not the classes" in data["reference_library"]["note"].lower()


class TestDiseases:
    def test_list_diseases(self, client_live):
        r = client_live.get("/api/diseases")
        assert r.status_code == 200
        assert r.json()["total"] > 0

    def test_filter_crop(self, client_live):
        r = client_live.get("/api/diseases?crop=Tomato")
        assert r.status_code == 200
        items = r.json()["items"]
        assert items
        for it in items:
            assert it["crop"] == "Tomato"

    def test_search(self, client_live):
        r = client_live.get("/api/diseases?search=blight")
        assert r.status_code == 200
        assert r.json()["total"] > 0

    def test_get_single(self, client_live):
        r = client_live.get("/api/diseases/tomato-early-blight")
        assert r.status_code == 200
        data = r.json()
        assert data["crop"] == "Tomato"
        assert "scientific_name" in data
        assert "sources" in data

    def test_missing_disease(self, client_live):
        assert client_live.get("/api/diseases/does-not-exist").status_code == 404


class TestPredictLive:
    def test_no_fake_prediction_when_model_unavailable(self, client_live):
        """Without a verified model, live prediction must fail, not invent a disease."""
        if _model_is_loaded(client_live):
            pytest.skip("a verified model is connected; covered by TestPredictWithModel")
        r = client_live.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 503
        detail = r.json()["detail"].lower()
        assert "no verified model" in detail or "unavailable" in detail
        # must not leak a disease name
        assert "late blight" not in detail
        assert "healthy" not in detail

    def test_invalid_file(self, client_live):
        r = client_live.post("/api/predict", files={"image": ("bad.txt", b"not an image", "text/plain")})
        assert r.status_code in (400, 422, 503)

    def test_empty_upload(self, client_live):
        r = client_live.post("/api/predict", files={"image": ("empty.png", b"", "image/png")})
        assert r.status_code in (400, 422, 503)

    def test_unsupported_extension(self, client_live):
        r = client_live.post("/api/predict", files={"image": ("leaf.gif", _make_png(), "image/gif")})
        assert r.status_code == 400


class TestPredictWithModel:
    def test_real_prediction_fields(self, client_live):
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        r = client_live.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["live_model"] is True
        assert data["demo_mode"] is False
        assert data["top_prediction"]["class_name"] in data["supported_classes"]
        assert 0.0 <= data["top_prediction"]["probability"] <= 1.0
        assert len(data["top_predictions"]) >= 1
        assert isinstance(data["low_confidence"], bool)
        if data["low_confidence"]:
            assert "Low-confidence" in data["low_confidence_message"]
        assert data["severity"]["available"] is False
        assert "unavailable" in data["severity"]["message"].lower()

    def test_low_confidence_flag_forced(self, client_live, monkeypatch):
        """Raising the threshold above the observed probability must flip the
        low-confidence flag and attach the honest message. This exercises the
        branch end-to-end by overriding one setting — no retraining, no faking
        of any prediction value (the probabilities stay the model's own)."""
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        from app.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "confidence_low", 0.9999)
        r = client_live.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["low_confidence"] is True
        assert "Low-confidence prediction" in data["low_confidence_message"]
        # The prediction itself is unchanged — still a real model output.
        assert data["live_model"] is True
        assert data["top_prediction"]["class_name"] in data["supported_classes"]

    def test_in_domain_probe_prediction(self, client_live):
        """The model author's own sample must classify as its known label."""
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        assert PROBE_IMAGE.exists(), f"probe image missing: {PROBE_IMAGE}"
        r = client_live.post(
            "/api/predict",
            files={"image": ("apple_scab.jpg", PROBE_IMAGE.read_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["top_prediction"]["class_name"] == "Apple___Apple_scab"
        assert data["top_prediction"]["probability"] > 0.5
        # The predicted class must come from the connected model's class list.
        assert data["supported_classes"] == client_live.get("/api/model-info").json()["classes"]

    def test_real_gradcam(self, client_live):
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        r = client_live.post("/api/explain", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is True
        assert data["overlay"].startswith("data:image/png;base64,")

    def test_predict_returns_gradcam_and_unavailable_severity(self, client_live):
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        r = client_live.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        data = r.json()
        assert data["explainability"]["available"] is True
        assert data["explainability"]["overlay"].startswith("data:image/png;base64,")
        assert data["severity"]["available"] is False
        assert data["severity"]["affected_area_percent"] is None
        assert data["severity"]["severity_level"] is None


class TestPredictDemo:
    def test_demo_mode_is_labelled(self, client_demo):
        r = client_demo.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["demo_mode"] is True
        assert data["live_model"] is False
        assert data["top_prediction"] is None, "demo mode must not emit a prediction"
        assert data["supported_classes"] == []
        assert any("DEMO MODE" in lim for lim in data["limitations"])

    def test_demo_severity_unavailable(self, client_demo):
        r = client_demo.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.json()["severity"]["available"] is False


class TestExplain:
    def test_explain_unavailable_without_model(self, client_live):
        r = client_live.post("/api/explain", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        if not _model_is_loaded(client_live):
            assert data["available"] is False
            assert "unavailable" in data["message"].lower()
            assert "heatmap" not in data or data.get("heatmap") is None


class TestHistory:
    def test_history_empty_or_populated(self, client_live):
        r = client_live.get("/api/history")
        assert r.status_code == 200
        assert "items" in r.json() and "total" in r.json()

    def test_history_not_populated_without_prediction(self, client_live):
        client_live.delete("/api/history")
        if not _model_is_loaded(client_live):
            client_live.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
            assert client_live.get("/api/history").json()["total"] == 0

    def test_clear_history(self, client_live):
        r = client_live.delete("/api/history")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert client_live.get("/api/history").json()["total"] == 0


class TestSeverityHonesty:
    def test_severity_message_is_explicit(self):
        from app.ml.severity_service import SeverityService

        svc = SeverityService(None)
        out = svc.analyze(None)
        assert out["available"] is False
        assert out["affected_area_percent"] is None
        assert out["severity_level"] is None
        assert "unavailable" in out["message"].lower()

    def test_severity_not_derived_from_confidence(self):
        from app.ml.severity_service import SeverityService

        svc = SeverityService(None)
        info = svc.get_method_info()
        assert info["available"] is False
        assert "never" in info["note"].lower()


class TestDataFiles:
    def test_class_names_reference_library(self):
        path = ROOT / "data" / "class_names.json"
        assert path.exists()
        assert len(json.loads(path.read_text(encoding="utf-8"))["class_names"]) == 38

    def test_model_classes_mapping_shape(self):
        path = ROOT / "data" / "model_classes.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "classes" in data and "status" in data
        if data["status"] == "unresolved":
            assert data["classes"] == []
        else:
            assert len(data["classes"]) == data["num_classes"]

    def test_diseases_file(self):
        path = ROOT / "data" / "diseases.json"
        assert path.exists()
        assert len(json.loads(path.read_text(encoding="utf-8"))["diseases"]) > 0

    def test_metrics_are_sourced_not_invented(self):
        path = ROOT / "data" / "model_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        training = data["training"]
        # Populated values must carry an attribution note (checkpoint/model card).
        if metrics.get("accuracy") is not None:
            assert metrics.get("note"), "accuracy must state its source"
        if training.get("validation_accuracy") is not None:
            assert training.get("note"), "validation accuracy must state its source"
        # Values that were never produced must stay null.
        for key in ("precision", "recall", "f1"):
            assert metrics.get(key) is None, f"{key} must not be invented"
        for key in ("training_accuracy", "training_loss", "validation_loss"):
            assert training.get(key) is None, f"{key} must not be invented"
        assert data["confusion_matrix"] is None
        assert data["class_wise"] is None