"""Real browser verification: open /scan and /diseases, take screenshots, print errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from playwright.sync_api import sync_playwright

def main():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome_path, headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"])
        page = b.new_page(viewport={"width": 1280, "height": 900})

        # 1. Diseases library — open Apple Scab detail
        print("[1] /diseases")
        page.goto("http://localhost:3000/diseases", wait_until="networkidle", timeout=15000)
        try:
            page.wait_for_selector("text=Apple Scab", timeout=10000)
            page.locator("button:has-text('Apple Scab')").first.click()
            page.wait_for_selector("text=Treatment Steps", timeout=8000)
            page.screenshot(path="tests/preview/01-diseases-apple-scab-detail.png", full_page=False)
            print("  screenshot: 01-diseases-apple-scab-detail.png")
        except Exception as e:
            print("  ERROR diseases detail:", e)

        # 2. Scan page — upload probe, click Analyze, capture result with treatment panel
        print("[2] /scan with probe")
        page.goto("http://localhost:3000/scan", wait_until="networkidle", timeout=15000)
        try:
            page.set_input_files("input[type=file]", "tests/probe_set/author_sample_apple_scab.jpg")
            page.get_by_role("button", name="Analyze").click()
            page.wait_for_selector("[data-testid='treatment-panel']", timeout=30000)
            page.screenshot(path="tests/preview/02-scan-result-treatment-panel.png", full_page=False)
            print("  screenshot: 02-scan-result-treatment-panel.png")
            # Also capture any console/page errors
            errors = page.evaluate("() => window.__browser_errors || []")
            print("  errors captured:", len(errors) if errors else 0)
        except Exception as e:
            print("  ERROR scan result:", e)
            page.screenshot(path="tests/preview/02-scan-result-ERROR.png", full_page=True)

        # 3. Health/model info (quick sanity)
        print("[3] /api/health check via fetch in page")
        health = page.evaluate('() => fetch("/api/health").then(r=>r.json())')
        print("  health:", health.get("status"), "model_loaded:", health.get("model_loaded"), "demo:", health.get("demo_mode"))

        # 4. Check if TreatmentPanel renders (check inner text)
        text = page.inner_text("[data-testid='treatment-panel']")
        print("[4] treatment panel text length:", len(text) if text else 0, "contains 'How to Treat It':", "How to Treat It" in (text or ""))

        b.close()

if __name__ == "__main__":
    main()
