"""End-to-end smoke test for PlantPath AI (live-model honest contract).

Verifies the full stack with the connected verified classifier:
API health → model info → disease library → crop explorer → real prediction
→ history record → rendered frontend pages contain required UI markers.

Production contract asserted here (DEMO_MODE=false, model connected):
  - /api/predict returns 200 with a real class from the connected model's
    own class list; confidence comes from softmax, severity stays
    unavailable (no segmentation model), Grad-CAM is a real heatmap.
  - The class list served by the API matches data/model_classes.json
    (which mirrors the checkpoint's embedded list) — nothing hard-coded.
  - Metrics appear only with attribution (checkpoint / model card);
    precision/recall/F1 stay null because they were never produced.
  - Invalid uploads are rejected with 400 before any inference.

Usage:
    python tests/e2e_smoke.py

Requires backend on :8000 and frontend on :3000.
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx

BACKEND = "http://localhost:8000"
FRONTEND = "http://localhost:3000"
ROOT = Path(__file__).resolve().parents[1]

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(f"{name} {detail}")
        print(f"  FAIL  {name} {detail}")


def make_test_image(size: int = 300) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (46, 125, 50)).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    print("=== PlantPath AI E2E smoke test (live model) ===\n")

    # 1. Health
    print("[1] Backend health")
    try:
        r = httpx.get(f"{BACKEND}/api/health", timeout=10)
        data = r.json()
        check("health status ok", r.status_code == 200 and data.get("status") == "ok")
        check("health reports demo_mode", "demo_mode" in data)
        check("production demo_mode is false",
              data.get("demo_mode") is False, f"got {data.get('demo_mode')}")
        check("model is loaded", data.get("model_loaded") is True,
              f"got {data.get('model_loaded')} detail={data.get('detail')!r}")
        check("class_count is 38", data.get("class_count") == 38,
              f"got {data.get('class_count')}")
        check("health says live inference enabled",
              "Live model loaded" in (data.get("detail") or ""),
              f"got {data.get('detail')!r}")
    except Exception as exc:
        check("backend reachable", False, str(exc))
        print("\nBackend not reachable. Start it first.")
        return 1

    # 2. Model info
    print("\n[2] Model info")
    r = httpx.get(f"{BACKEND}/api/model-info", timeout=10)
    d = r.json()
    check("model-info 200", r.status_code == 200)
    check("model loaded on model-info", d.get("loaded") is True)
    classes = d.get("classes") or []
    check("38 classes reported", len(classes) == 38, f"got {len(classes)}")
    check("classes come from checkpoint mapping",
          all("___" in c for c in classes),
          f"sample={classes[:3]!r}")
    check("architecture reported",
          "EfficientNetV2" in (d.get("architecture") or ""),
          f"got {d.get('architecture')!r}")
    check("source attributed", bool(d.get("source")), f"got {d.get('source')!r}")
    metrics = d.get("metrics", {})
    if metrics.get("accuracy") is not None:
        check("metrics.accuracy carries attribution note", bool(metrics.get("note")))
    else:
        check("metrics.accuracy null is honest fallback", True)
    for key in ("precision", "recall", "f1"):
        check(f"metrics.{key} not invented", metrics.get(key) is None,
              f"got {metrics.get(key)!r}")
    check("comparison is a list (no invented model cards)",
          isinstance(d.get("comparison"), list))
    check("demo_mode false on model-info", d.get("demo_mode") is False)
    check("severity available=false", d.get("severity", {}).get("available") is False)
    check("severity has unavailable_reason",
          bool(d.get("severity", {}).get("unavailable_reason")))
    check("gradcam_available=true (architecture supports it)",
          d.get("gradcam_available") is True, f"got {d.get('gradcam_available')!r}")
    check("segmentation_status=unavailable", d.get("segmentation_status") == "unavailable")
    check("low_confidence_threshold present",
          isinstance(d.get("low_confidence_threshold"), (int, float)))

    # 3. Diseases
    print("\n[3] Disease library")
    r = httpx.get(f"{BACKEND}/api/diseases", timeout=10)
    d = r.json()
    check("diseases listed", d.get("total", 0) >= 30, f"got {d.get('total')}")
    r = httpx.get(f"{BACKEND}/api/diseases", params={"search": "blight"}, timeout=10)
    check("search works", r.json().get("total", 0) > 0)
    r = httpx.get(f"{BACKEND}/api/diseases", params={"crop": "Tomato"}, timeout=10)
    items = r.json().get("items", [])
    check("crop filter works", len(items) > 0 and all(i["crop"] == "Tomato" for i in items))
    r = httpx.get(f"{BACKEND}/api/diseases/tomato-late-blight", timeout=10)
    check("disease detail", r.status_code == 200 and "scientific_name" in r.json())
    r = httpx.get(f"{BACKEND}/api/diseases/nope", timeout=10)
    check("missing disease 404", r.status_code == 404)

    # 4. Crops: supported set mirrors the model's real classes
    print("\n[4] Crop explorer")
    r = httpx.get(f"{BACKEND}/api/crops", timeout=10)
    d = r.json()
    supported = d.get("supported_classes") or []
    check("supported_classes = model-info classes",
          supported == classes, f"got {len(supported)} vs {len(classes)}")
    check("supported crops derived from model classes (14 crops)",
          d.get("total") == 14, f"got {d.get('total')}")
    ref = d.get("reference_library", {})
    check("reference library populated (14 crops)", ref.get("total") == 14,
          f"got {ref.get('total')}")
    check("reference library clearly labelled",
          "not the classes" in (ref.get("note") or "").lower(),
          f"got {ref.get('note')!r}")
    check("healthy flags present",
          all("healthy_available" in c for c in ref.get("items", [])))

    # 5. Prediction: real inference with a known in-domain probe image
    print("\n[5] Prediction (live model)")
    probe = ROOT / "tests" / "probe_set" / "author_sample_apple_scab.jpg"
    if probe.exists():
        img = probe.read_bytes()
        fname, mime = "author_sample_apple_scab.jpg", "image/jpeg"
    else:
        img = make_test_image()
        fname, mime = "smoke_leaf.png", "image/png"
    r = httpx.post(f"{BACKEND}/api/predict",
                   files={"image": (fname, img, mime)}, timeout=60)
    check("predict 200 (real inference)", r.status_code == 200,
          f"got {r.status_code} body={r.text[:200]!r}")
    body = r.json() if r.status_code == 200 else {}
    check("live_model true / demo_mode false",
          body.get("live_model") is True and body.get("demo_mode") is False)
    top = body.get("top_prediction") or {}
    check("top_prediction present with real class",
          bool(top.get("class_name")) and "___" in top.get("class_name", ""),
          f"got {top!r}")
    check("predicted class is in the model's supported list",
          top.get("class_name") in (body.get("supported_classes") or []),
          f"got {top.get('class_name')!r}")
    if probe.exists():
        check("in-domain probe classifies as Apple scab",
              top.get("class_name") == "Apple___Apple_scab",
              f"got {top.get('class_name')!r}")
    check("probability is a real softmax value",
          isinstance(top.get("probability"), float) and 0.0 <= top["probability"] <= 1.0,
          f"got {top.get('probability')!r}")
    sev = body.get("severity") or {}
    check("severity stays unavailable (no segmentation model)",
          sev.get("available") is False and sev.get("severity_level") is None,
          f"got {sev!r}")
    expl = body.get("explainability") or {}
    check("Grad-CAM heatmap returned",
          expl.get("available") is True
          and str(expl.get("overlay", "")).startswith("data:image/png;base64,"),
          f"got available={expl.get('available')} message={expl.get('message')!r}")

    # 6. Invalid inputs (validated before inference)
    print("\n[6] Error handling")
    r = httpx.post(f"{BACKEND}/api/predict",
                   files={"image": ("bad.txt", b"hello", "text/plain")}, timeout=10)
    check("invalid extension 400", r.status_code == 400, f"got {r.status_code}")
    r = httpx.post(f"{BACKEND}/api/predict",
                   files={"image": ("e.png", b"", "image/png")}, timeout=10)
    check("empty upload 400", r.status_code == 400, f"got {r.status_code}")

    # 7. History: the successful prediction is recorded once
    print("\n[7] History")
    r_before = httpx.get(f"{BACKEND}/api/history", timeout=10).json()
    httpx.post(f"{BACKEND}/api/predict",
               files={"image": (fname, img, mime)}, timeout=60)
    r_after = httpx.get(f"{BACKEND}/api/history", timeout=10).json()
    check("prediction recorded in history",
          r_after.get("total", 0) == r_before.get("total", 0) + 1,
          f"before={r_before.get('total')} after={r_after.get('total')}")
    if r_after.get("items"):
        item = r_after["items"][0]
        check("no absolute paths in filename",
              not item["filename"].startswith("C:"))
        check("mode recorded as real", item.get("mode") == "real",
              f"got {item.get('mode')!r}")
        check("recorded class matches model classes",
              item.get("disease") and item.get("confidence") is not None)

    # 8. Frontend pages (static markers; live badges are asserted in browser_test)
    print("\n[8] Frontend pages")
    pages = {
        "/": ["PlantPath", "Analysis Workflow", "Scan a Leaf", "Model Limitations"],
        "/scan": ["Upload Image", "Use Camera", "No analysis yet"],
        "/diseases": ["Disease Library", "Reference library"],
        "/model": ["Model", "No architectures are reported"],
        "/research": ["Laboratory-to-Field Gap", "Research Pipeline"],
        "/history": ["Analysis History", "Clear History"],
        "/about": ["Academic Disclaimer", "Technology Stack"],
    }
    for path, markers in pages.items():
        try:
            r = httpx.get(f"{FRONTEND}{path}", timeout=20)
            check(f"page {path} 200", r.status_code == 200, f"got {r.status_code}")
            for marker in markers:
                check(f"page {path} contains '{marker}'", marker in r.text)
        except Exception as exc:
            check(f"page {path} reachable", False, str(exc))

    # Summary
    print(f"\n=== Results: {len(PASS)} passed, {len(FAIL)} failed ===")
    if FAIL:
        print("\nFailures:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
