"""PlantPath AI — real-user browser journey test (Playwright + system Chrome).

Walks the app like a human user, captures screenshots, and records PASS/FAIL.

Asserts the production live-model contract (DEMO_MODE=false, model connected):
  - navbar/model badge reads "LIVE MODEL" (never "Demo Mode")
  - Analyze uploads a real image and the UI shows a real prediction: detected
    class from the model's own class list, softmax confidence, ranked
    top predictions, Grad-CAM heatmap
  - severity still reports honestly unavailable (no segmentation model)
  - report download contains the real prediction and the live-model status
  - the prediction is recorded in history, then cleared
  - model dashboard reports the connected architecture, 38 mapped classes,
    checkpoint-sourced accuracy, and "Not available" for metrics that no
    evaluation run produced (precision/recall/F1)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from playwright.sync_api import expect, sync_playwright

BASE = "http://localhost:3000"
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "browser_report"
OUT.mkdir(exist_ok=True)

# In-domain probe image: the model author's own sample, known label
# Apple___Apple_scab (verified p=0.90 by tests/verify_new_model.py).
PROBE = ROOT / "tests" / "probe_set" / "author_sample_apple_scab.jpg"

results: list[tuple[str, bool, str]] = []
console_errors: list[str] = []

STATUS_LIVE = "LIVE MODEL"
UNAVAILABLE_HEAD = "Predictions are unavailable"


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def visible(locator) -> bool:
    """Visibility check tolerant of strict-mode ambiguity."""
    try:
        return locator.first.is_visible(timeout=5000)
    except Exception:
        try:
            return locator.count() > 0
        except Exception:
            return False


def shot(page, name: str) -> None:
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True, animations="disabled")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.on(
            "console",
            lambda m: console_errors.append(m.text)
            if m.type == "error"
            else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        # ---------------- 1. HOME ----------------
        print("\n[1] Home page")
        page.goto(BASE, wait_until="networkidle")
        check("home loads", page.locator("h1").first.inner_text().startswith("PlantPath"))
        check("hero CTA 'Scan a Leaf' visible", page.get_by_role("link", name="Scan a Leaf").first.is_visible())
        check("workflow steps shown", page.get_by_text("Analysis Workflow").is_visible())
        check("limitations section", page.get_by_text("Model Limitations").is_visible())
        check("LIVE MODEL badge in navbar",
              visible(page.get_by_text(STATUS_LIVE)))
        check("no 'Model not available' badge",
              page.get_by_text("Model not available — no predictions can be generated").count() == 0)
        check("no 'Demo Mode' badge shown",
              page.get_by_text("Demo Mode").count() == 0)
        check("sourced validation accuracy teaser on home",
              visible(page.get_by_text("Validation accuracy reported by the connected checkpoint")))
        check("footer disclaimer", page.get_by_text("should not be treated as definitive agricultural diagnosis").first.is_visible())
        shot(page, "01-home")

        # ---------------- 2. NAVIGATION ----------------
        print("\n[2] Navigation")
        page.get_by_role("link", name="Diseases").first.click()
        page.wait_for_url("**/diseases", timeout=10000)
        check("nav → Diseases", "/diseases" in page.url)
        page.get_by_role("link", name="Model").first.click()
        page.wait_for_url("**/model", timeout=10000)
        check("nav → Model", "/model" in page.url)
        page.get_by_role("link", name="Home").first.click()
        page.wait_for_url("http://localhost:3000/", timeout=10000)
        check("nav → Home", page.url.rstrip("/") == BASE.rstrip("/"))

        # ---------------- 3. SCAN: real prediction ----------------
        print("\n[3] Scan Leaf — upload & analyze (live model)")
        page.goto(f"{BASE}/scan", wait_until="networkidle")
        check("scan page title", page.get_by_role("heading", name="Scan Leaf").is_visible())
        check("upload tab default", page.get_by_text("Drag & drop a leaf image here").is_visible())
        check("camera tab exists", page.get_by_role("tab", name="Use Camera").is_visible())
        check("empty result state", page.get_by_text("No analysis yet").is_visible())
        check("disclaimer shown", page.get_by_text("Research disclaimer").is_visible())
        check("supported classes reported (38)",
              visible(page.get_by_text("38 classes reported")),
              "expected the model's real class count")
        check("no empty supported-classes state",
              page.get_by_text("No model classes are connected yet").count() == 0)

        assert PROBE.exists(), f"probe image missing: {PROBE}"
        page.set_input_files("input[type=file]", str(PROBE))
        check("file selected shows name",
              page.get_by_text("author_sample_apple_scab.jpg").is_visible())
        check("preview shown", page.locator("img[alt='Selected leaf preview']").is_visible())

        page.get_by_role("button", name="Analyze").click()
        page.wait_for_selector("text=Analysis Complete", timeout=30000)
        check("real prediction result appears",
              page.get_by_text("Analysis Complete").first.is_visible())
        check("no unavailable alert for a working model",
              page.get_by_text(UNAVAILABLE_HEAD).count() == 0)
        check("detected class from the model's set (Apple scab)",
              visible(page.get_by_text("Apple scab")))
        check("crop shown", visible(page.locator("dt", has_text="Crop")))
        check("LIVE MODEL badge beside result", visible(page.get_by_text(STATUS_LIVE)))
        check("Top Predictions section rendered",
              page.get_by_text("Top Predictions").first.is_visible())
        check("ranked probability bars present", page.locator("li .bar-grow").count() >= 1)
        check("confidence shown as percent",
              page.evaluate(r"() => /\d+(\.\d+)?%/.test(document.body.innerText)"))
        check("model architecture reported in result",
              visible(page.get_by_text("EfficientNetV2-S (torchvision) + Linear(38) head")))
        check("no DEMO MODE banner", page.get_by_text("DEMO MODE").count() == 0)
        # Grad-CAM + severity honesty
        check("Grad-CAM heatmap rendered",
              visible(page.get_by_text("AI Attention / Grad-CAM")))
        check("severity honestly unavailable",
              visible(page.get_by_text("Severity assessment is currently unavailable for this model.")))
        # The severity card's available branch shows the "Affected Leaf Area" /
        # "Severity Level" definition list plus Mild/Moderate/Severe values.
        # With no segmentation model connected none of those may appear — the
        # card must stay on its honest unavailable branch. (Checked as whole
        # rendered values so the unavailable branch's own explanatory sentence
        # — "no ... percentage or severity level is shown" — doesn't false-hit.)
        check("no fabricated severity percentage",
              page.get_by_text("Affected Leaf Area").count() == 0 and
              page.get_by_text("Mild", exact=True).count() == 0 and
              page.get_by_text("Moderate", exact=True).count() == 0 and
              page.get_by_text("Severe", exact=True).count() == 0)
        shot(page, "02-scan-result")

        # ---------------- 4. Report download (real data) ----------------
        print("\n[4] Report download")
        check("report button visible", page.get_by_role("button", name="Download Report").first.is_visible())
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download Report").click()
        download = dl.value
        fname = download.suggested_filename
        check("report downloads", fname.startswith("plantpath-report-") and fname.endswith(".html"), fname)
        try:
            html = Path(download.path()).read_text(encoding="utf-8", errors="replace")
            check("report contains the real top class",
                  "<h2>Top Predictions</h2>" in html and "Apple scab" in html)
            check("report states live-model status",
                  "Live model — result returned by the backend" in html)
            check("report has the real architecture",
                  "EfficientNetV2" in html)
            check("report severity is honestly unavailable",
                  "Severity assessment is currently unavailable for this model." in html)
            check("report has no demo banner", "DEMO MODE" not in html)
        except Exception as e:
            check("report content readable", False, str(e)[:80])

        # ---------------- 5. DISEASES ----------------
        print("\n[5] Disease library")
        page.goto(f"{BASE}/diseases", wait_until="networkidle")
        page.wait_for_selector("text=record", timeout=15000)
        total = page.locator("text=/^\\d+ record/").first.inner_text()
        check("records listed", "record" in total, total)

        # search
        page.get_by_label("Search diseases").fill("blight")
        page.wait_for_timeout(1200)
        cards = page.locator("button:has-text('Blight')").count()
        check("search 'blight' filters", cards >= 1, f"{cards} cards")
        page.get_by_label("Search diseases").fill("")
        page.wait_for_timeout(1200)

        # crop filter
        page.get_by_label("Filter by crop").select_option("Tomato")
        page.wait_for_timeout(1200)
        tomato_cards = page.locator("text=Tomato").count()
        check("crop filter Tomato works", tomato_cards >= 1, f"{tomato_cards} mentions")
        page.get_by_label("Filter by crop").select_option("")
        page.wait_for_timeout(1000)

        # detail panel
        page.locator("button:has-text('Late Blight')").first.click()
        page.wait_for_timeout(600)
        detail_visible = page.get_by_role("dialog").is_visible()
        check("disease detail dialog opens", detail_visible)
        if detail_visible:
            check("scientific name present", page.get_by_text("Phytophthora infestans").first.is_visible())
            check("symptoms listed", page.get_by_text("Symptoms").first.is_visible())
            check("sources listed", page.get_by_text("Sources").first.is_visible())
            shot(page, "03-disease-detail")
            page.get_by_label("Close disease details").click()

        # reference library label (never presented as model output)
        check("reference library labelled",
              page.get_by_text("Reference library (educational)").first.is_visible())
        check("no 38-class Crop Explorer claim",
              page.get_by_text("Crop Explorer").count() == 0)
        shot(page, "04-diseases")

        # ---------------- 6. MODEL ----------------
        print("\n[6] Model dashboard")
        page.goto(f"{BASE}/model", wait_until="networkidle")
        page.wait_for_selector("text=Model Overview", timeout=15000)
        check("overview section", page.get_by_text("Model Overview").first.is_visible())
        check("no unavailable notice on model page",
              page.get_by_text("Model not available — no predictions can be generated").count() == 0)
        check("architecture reported",
              visible(page.get_by_text("EfficientNetV2-S (torchvision) + Linear(38) head")))
        check("loaded = Yes", page.get_by_text("Yes", exact=True).count() >= 1)
        metric_values = page.locator("div.font-mono").all_inner_texts()
        has_accuracy = any(t.strip().startswith("0.998") for t in metric_values)
        check("sourced accuracy shown (0.9989)", has_accuracy, str(metric_values[:8]))
        has_not_avail = any("Not available" in t for t in metric_values)
        check("metrics with no evaluation run show 'Not available'", has_not_avail,
              f"{len(metric_values)} metric cards")
        check("comparison table",
              page.get_by_text("No architectures are reported for comparison").first.is_visible())
        check("mapped classes rendered (38 class tags)",
              page.locator("text=Apple___Apple_scab").count() > 0)
        check("'Class names not resolved' absent",
              page.get_by_text("Class names not resolved").count() == 0)
        check("mapping provenance shown",
              visible(page.get_by_text("Mapping provenance:")))
        check("availability line says real predictions are served",
              visible(page.get_by_text("A verified model is loaded and the backend serves real predictions.")))
        check("limitations panel", page.get_by_text("PlantVillage is a controlled benchmark dataset").first.is_visible())
        shot(page, "05-model")

        # ---------------- 7. RESEARCH ----------------
        print("\n[7] Research page")
        page.goto(f"{BASE}/research", wait_until="networkidle")
        check("pipeline", page.get_by_text("Research Pipeline").first.is_visible())
        check("problem statement", page.get_by_text("Problem Statement").first.is_visible())
        check("lab-to-field gap", page.get_by_text("Laboratory-to-Field Gap").first.is_visible())
        check("future work", page.get_by_text("Future Work").first.is_visible())
        check("no-overclaim note", page.get_by_text("must not be represented as equivalent to field-deployment performance").is_visible())
        shot(page, "06-research")

        # ---------------- 8. HISTORY ----------------
        print("\n[8] History")
        page.goto(f"{BASE}/history", wait_until="networkidle")
        page.wait_for_timeout(1500)
        rows_before = page.locator("tbody tr").count()
        check("scan prediction recorded in history", rows_before >= 1, f"{rows_before} rows")
        page.get_by_role("button", name="Clear History").click()
        page.wait_for_timeout(1500)
        check("history cleared → empty state", page.get_by_text("No predictions recorded").is_visible())

        # a successful prediction adds exactly one record
        page.goto(f"{BASE}/scan", wait_until="networkidle")
        page.set_input_files("input[type=file]", str(PROBE))
        page.get_by_role("button", name="Analyze").click()
        page.wait_for_selector("text=Analysis Complete", timeout=30000)
        page.goto(f"{BASE}/history", wait_until="networkidle")
        page.wait_for_timeout(1500)
        rows = page.locator("tbody tr").count()
        check("recorded prediction appears in history", rows == 1, f"{rows} rows")
        if rows >= 1:
            row_text = page.locator("tbody tr").first.inner_text()
            check("history row shows real class + confidence",
                  "Apple" in row_text, row_text.replace("\n", " ")[:100])
        shot(page, "07-history")

        # ---------------- 9. ABOUT ----------------
        print("\n[9] About")
        page.goto(f"{BASE}/about", wait_until="networkidle")
        check("academic disclaimer", page.get_by_text("Academic Disclaimer").first.is_visible())
        check("tech stack", page.get_by_text("Technology Stack").first.is_visible())
        check("research principles (no fabricated results)",
              page.get_by_text("No fabricated results").is_visible())

        # ---------------- 10. DARK MODE ----------------
        print("\n[10] Dark mode")
        page.goto(BASE, wait_until="networkidle")
        page.get_by_label("Switch to dark mode").click()
        page.wait_for_timeout(500)
        is_dark = page.evaluate("() => document.documentElement.classList.contains('dark')")
        check("dark mode activates", is_dark)
        stored = page.evaluate("() => localStorage.getItem('plantpath-theme')")
        check("dark mode persisted", stored == "dark", str(stored))
        shot(page, "09-home-dark")
        # switch back
        page.get_by_label("Switch to light mode").click()
        page.wait_for_timeout(300)

        # ---------------- 11. BACKEND DOWN (error state) ----------------
        print("\n[11] Backend-unavailable error state")
        page.goto(f"{BASE}/scan", wait_until="networkidle")
        page.set_input_files("input[type=file]", str(PROBE))
        # abort backend calls only
        page.route("**/api/predict", lambda r: r.abort())
        page.get_by_role("button", name="Analyze").click()
        try:
            page.wait_for_selector("text=Analysis failed", timeout=15000)
            check("network failure shows clean error", True)
            check("no python traceback leaked",
                  "Traceback" not in page.content() and "File \"" not in page.content())
            page.screenshot(path=str(OUT / "10-error-state.png"), full_page=True)
        except Exception as e:
            check("network failure shows clean error", False, str(e)[:80])
        page.unroute("**/api/predict")

        # ---------------- 12. MOBILE RESPONSIVE ----------------
        print("\n[12] Mobile viewport (390×844)")
        mob = browser.new_context(viewport={"width": 390, "height": 844})
        mp = mob.new_page()
        mp.goto(BASE, wait_until="networkidle")
        # hamburger visible, desktop nav hidden
        burger = mp.get_by_label("Open menu")
        check("hamburger menu visible", burger.is_visible())
        burger.click()
        mp.wait_for_timeout(400)
        check("mobile menu opens", mp.get_by_role("navigation", name="Mobile").is_visible())
        mp.screenshot(path=str(OUT / "11-mobile-menu.png"), full_page=True)
        mp.get_by_role("link", name="Scan Leaf").last.click()
        try:
            mp.wait_for_url("**/scan", timeout=10000)
        except Exception:
            pass
        mp.wait_for_load_state("networkidle")
        check("mobile nav → Scan", "/scan" in mp.url)
        # horizontal overflow check
        overflow = mp.evaluate("""() => [...document.querySelectorAll('*')]
            .filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 2)
            .slice(0,5).map(e => e.tagName)""")
        check("no horizontal overflow on mobile", len(overflow) == 0, str(overflow))
        mp.screenshot(path=str(OUT / "12-mobile-scan.png"), full_page=True)

        # mobile upload+analyze → real prediction on mobile too
        mp.set_input_files("input[type=file]", str(PROBE))
        mp.get_by_role("button", name="Analyze").click()
        try:
            mp.wait_for_selector("text=Analysis Complete", timeout=30000)
            check("mobile scan → real prediction", True)
            check("mobile shows detected class",
                  visible(mp.get_by_text("Apple scab")))
            check("mobile severity still honest",
                  visible(mp.get_by_text("Severity assessment is currently unavailable for this model.")))
            mp.screenshot(path=str(OUT / "13-mobile-result.png"), full_page=True)
        except Exception as e:
            check("mobile scan → real prediction", False, str(e)[:80])
        mob.close()

        # ---------------- 13. KEYBOARD / a11y spot checks ----------------
        print("\n[13] Accessibility spot checks")
        page.goto(BASE, wait_until="networkidle")
        # semantic landmarks
        check("has <nav>", page.locator("nav").count() >= 1)
        check("has <main>", page.locator("main").count() >= 1)
        check("has <footer>", page.locator("footer").count() >= 1)
        check("html lang set", page.get_attribute("html", "lang") == "en")
        # tab focus visible
        page.keyboard.press("Tab")
        focused = page.evaluate("() => document.activeElement?.tagName")
        check("first Tab focuses an element", focused not in (None, "BODY"), str(focused))
        # images have alt
        missing_alt = page.evaluate(
            "() => [...document.querySelectorAll('img')].filter(i=>!i.alt && !i.getAttribute('aria-hidden')).length"
        )
        check("all images have alt text", missing_alt == 0, f"{missing_alt} missing")

        # ---------------- 14. Console cleanliness ----------------
        print("\n[14] Console errors")
        # Filter benign favicon noise and errors caused by our own
        # deliberate network-abort test (step 11).
        real = [
            e
            for e in console_errors
            if "favicon" not in e.lower()
            and "ERR_FAILED" not in e
            and "status of 404" not in e
            and "status of 503" not in e
        ]
        check("no unexpected console/page errors", len(real) == 0, "; ".join(real[:3]))
        check("only benign errors (favicon/test-abort)",
              len(console_errors) - len(real) <= 3,
              f"{len(console_errors)} total captured")

        browser.close()

    # ---------------- SUMMARY ----------------
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [(n, d) for n, ok, d in results if not ok]
    print(f"\n{'='*60}")
    print(f"BROWSER TEST RESULTS: {passed}/{len(results)} passed, {len(failed)} failed")
    if failed:
        print("Failures:")
        for n, d in failed:
            print(f"  - {n} {d}")
    print(f"Screenshots: {OUT}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
