from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"],
    )
    page = b.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://localhost:3000/scan", wait_until="networkidle", timeout=15000)
    page.set_input_files("input[type=file]", "tests/probe_set/author_sample_apple_scab.jpg")
    page.get_by_role("button", name="Analyze").click()
    page.wait_for_selector("[data-testid='treatment-panel']", timeout=30000)
    page.wait_for_timeout(2000)
    text = page.inner_text("[data-testid='treatment-panel']")
    print("FULL PANEL TEXT:")
    print(text)
    print("\n=== CHECK ===")
    print("Length:", len(text))
    for s in ["How", "Treat", "Pesticide", "label", "Source", "Apple", "Scab", "fungicide"]:
        print(f"  '{s}' in text:", s in text)
    b.close()
