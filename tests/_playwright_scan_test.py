from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"],
    )
    page = b.new_page(viewport={"width": 1280, "height": 900})
    print("[1] Opening /scan")
    page.goto("http://localhost:3000/scan", wait_until="networkidle", timeout=15000)
    page.wait_for_selector("text=Scan a Leaf", timeout=10000)
    print("[2] Scan page loaded")
    page.screenshot(path="tests/preview/04-scan-page-load.png", full_page=False)
    page.set_input_files("input[type=file]", "tests/probe_set/author_sample_apple_scab.jpg")
    print("[3] File selected")
    page.get_by_role("button", name="Analyze").click()
    print("[4] Analyze clicked, waiting for result...")
    page.wait_for_selector("[data-testid='treatment-panel']", timeout=30000)
    print("[5] TreatmentPanel found — feature working!")
    page.screenshot(path="tests/preview/05-scan-with-treatment.png", full_page=False)
    # Capture treatment text
    panel_text = page.inner_text("[data-testid='treatment-panel']")
    print("[6] Panel content length:", len(panel_text))
    print("[6] Contains 'How to Treat It':", "How to Treat It" in panel_text)
    print("[6] Contains pesticide pills:", "Recommended Pesticides" in panel_text)
    print("[6] Contains disclaimer:", "product label" in panel_text)
    b.close()
