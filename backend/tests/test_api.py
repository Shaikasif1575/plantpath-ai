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

    def test_severity_available_with_connected_model(self, client_live):
        """With the project-trained segmentation model present, model-info must
        report available severity AND its honest attribution (pseudo-labels)."""
        data = client_live.get("/api/model-info").json()
        sev = data.get("severity", {})
        assert sev.get("available") is True
        assert "confidence" in (sev.get("note") or "").lower()
        assert "pseudo-label" in (sev.get("note") or "").lower()
        # The without-model branch still reports honestly (unit-level, below).
        assert data.get("segmentation_available") is True
        assert data.get("segmentation_status") == "available"

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
        # Severity must be coherent: either segmentation-derived values that
        # match the published thresholds, or an honest unavailable result.
        sev = data["severity"]
        if sev["available"]:
            assert isinstance(sev["affected_area_percent"], float)
            assert sev["severity_level"] in ("Mild", "Moderate", "Severe")
        else:
            assert "unavailable" in (sev.get("message") or "").lower()
            assert sev["affected_area_percent"] is None

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

    def test_predict_includes_treatment_guidance(self, client_live):
        """Live predictions must carry treatment + pesticide guidance for the
        predicted disease (the user-facing 'how do I cure it' requirement)."""
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        assert PROBE_IMAGE.exists(), f"probe image missing: {PROBE_IMAGE}"
        r = client_live.post(
            "/api/predict",
            files={"image": ("apple_scab.jpg", PROBE_IMAGE.read_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        info = r.json().get("disease_info")
        assert info is not None, "disease_info missing from live prediction"
        assert info["id"] == "apple-apple-scab"
        assert info["condition"] == "diseased"
        assert info["treatment"], "treatment steps missing"
        assert info["sources"], "knowledge sources missing"
        assert info["disclaimer"], "label-law disclaimer missing"
        assert info["pesticides"], "pesticide guidance missing for a fungal disease"
        p = info["pesticides"][0]
        assert p["category"] and p["active_ingredients"] and p["application"]
        # Guidance must belong to the predicted disease, not another record.
        assert "scab" in info["disease"].lower()

    def test_disease_kb_treatment_integrity(self, client_live):
        """Every diseased record must carry real treatment steps; pesticide
        entries must be well-formed. Empty pesticide lists are allowed only
        where no chemical applies (e.g., viruses)."""
        import json as _json

        data = _json.loads((ROOT / "data" / "diseases.json").read_text(encoding="utf-8"))
        records = data.get("diseases", [])
        assert len(records) == 38, f"unexpected record count: {len(records)}"
        diseased = [r for r in records if r.get("condition") == "diseased"]
        healthy = [r for r in records if r.get("condition") == "healthy"]
        assert len(diseased) == 26 and len(healthy) == 12
        for r in records:
            assert "treatment" in r and "pesticides" in r, f"missing keys in {r.get('id')}"
        for r in diseased:
            assert r["treatment"], f"diseased record {r['id']} has no treatment steps"
            for p in r["pesticides"]:
                assert p.get("category"), f"pesticide missing category in {r['id']}"
                assert p.get("active_ingredients"), f"no ingredients in {r['id']}"
                assert p.get("application"), f"no application notes in {r['id']}"
        no_chem = [r["id"] for r in diseased if not r["pesticides"]]
        # Known honest exceptions: no pesticide controls a plant virus directly
        # (mosaic virus); everything else must list what to apply.
        assert set(no_chem) <= {"tomato-mosaic-virus"}, f"unexpected empty pesticide lists: {no_chem}"
        incurable = {r["id"] for r in records if r.get("incurable")}
        assert "orange-hlb-citrus-greening" in incurable
        assert "tomato-yellow-leaf-curl-virus" in incurable
        assert "tomato-mosaic-virus" in incurable

    def test_api_top1_matches_direct_classifier(self, client_live):
        """Regression: the API's top-1 must equal the classifier's own top-1.

        Pins the class-order probability mapping — a confidence-sorted rank list
        used as class indices would report class_names[0] for every image (this
        bug once made every prediction 'Apple___Apple_scab').
        """
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        import numpy as np
        from PIL import Image

        import app.main as main_app

        clf = main_app.AppState.classifier
        # A field image whose true top-1 is NOT the first class alphabetically.
        field = ROOT / "data" / "field_eval" / "images" / "Tomato___Late_blight__00.jpg"
        img_path = field if field.exists() else PROBE_IMAGE
        arr = np.asarray(Image.open(img_path).convert("RGB"))
        direct = clf.predict(arr)
        r = client_live.post(
            "/api/predict", files={"image": (img_path.name, img_path.read_bytes(), "image/jpeg")}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["top_prediction"]["class_name"] == direct["class_name"]
        assert abs(data["top_prediction"]["probability"] - direct["confidence"]) < 0.01
        if direct["class_name"] != clf.get_classes()[0]:
            assert data["top_prediction"]["class_name"] != clf.get_classes()[0], (
                "top-1 collapsed to the first class of the mapping"
            )

    def test_real_gradcam(self, client_live):
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        r = client_live.post("/api/explain", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is True
        assert data["overlay"].startswith("data:image/png;base64,")

    def test_predict_returns_gradcam_and_real_severity(self, client_live):
        if not _model_is_loaded(client_live):
            pytest.skip("no verified model connected")
        assert PROBE_IMAGE.exists(), f"probe image missing: {PROBE_IMAGE}"
        # A real leaf photograph so the leaf segmenter has tissue to find.
        r = client_live.post(
            "/api/predict",
            files={"image": ("apple_scab.jpg", PROBE_IMAGE.read_bytes(), "image/jpeg")},
        )
        data = r.json()
        assert data["explainability"]["available"] is True
        assert data["explainability"]["overlay"].startswith("data:image/png;base64,")
        sev = data["severity"]
        assert sev["available"] is True
        assert isinstance(sev["affected_area_percent"], float)
        assert 0.0 <= sev["affected_area_percent"] <= 100.0
        assert sev["severity_level"] in ("Mild", "Moderate", "Severe")
        # Severity level must agree with the published thresholds.
        pct = sev["affected_area_percent"]
        expected = "Mild" if pct < 10 else ("Moderate" if pct < 40 else "Severe")
        assert sev["severity_level"] == expected
        assert sev["method"] == "diseased region area / leaf region area x 100"
        assert (sev.get("mask_image") or "").startswith("data:image/png;base64,")
        assert (sev.get("original_image") or "").startswith("data:image/png;base64,")
        # It must be labelled as segmentation-derived, not confidence-derived.
        assert "confidence" in (sev.get("note") or "").lower() or "classifier" in (sev.get("note") or "").lower()


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

    def test_disease_info_absent_in_demo_mode(self, client_demo):
        """Demo mode must not attach guidance (no real prediction to treat)."""
        r = client_demo.post("/api/predict", files={"image": ("leaf.png", _make_png(), "image/png")})
        assert r.status_code == 200
        assert r.json().get("disease_info") is None


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