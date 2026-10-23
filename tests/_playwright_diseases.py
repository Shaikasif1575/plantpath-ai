from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"],
    )
    page = b.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://localhost:3000/diseases", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(3000)
    page.screenshot(path="tests/preview/03-diseases-live.png", full_page=True)
    # Print visible text and console errors
    print("VISIBLE TEXT:")
    text = page.inner_text("body")
    print(text[:4000])
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    print("\nCONSOLE ERRORS:", errors[-5:] if errors else "none captured")
    b.close()
