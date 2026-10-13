"""One-off: verify the model dashboard's Severity Segmentation section renders
real trained-metric values (new UI added with the connected severity model)."""
from playwright.sync_api import sync_playwright

CHECKS = [
    ("section header", "Severity Segmentation"),
    ("lesion dice label", "Lesion Dice (val)"),
    ("lesion dice value", "0.6845"),
    ("lesion iou value", "0.5246"),
    ("leaf dice value", "0.6868"),
    ("leaf iou value", "0.5284"),
    ("lesion dataset repo", "godoldol99"),
    ("leaf dataset repo", "LeafNet75"),
    ("pseudo-label caveat", "pseudo-labels"),
    ("val-metric honesty note", "Held-out validation against the same label source"),
]

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    page = b.new_page()
    page.goto("http://127.0.0.1:3000/model", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2500)
    body = page.inner_text("body")
    body_l = body.lower()
    ok = fail = 0
    for name, needle in CHECKS:
        if needle.lower() in body_l:
            print(f"  PASS  {name}: {needle!r}")
            ok += 1
        else:
            print(f"  FAIL  {name}: {needle!r} not found")
            fail += 1
    # section must not render when values missing
    page.screenshot(path="tests/browser_report/14-model-severity-section.png", full_page=True)
    b.close()
    print(f"\nMODEL SEVERITY SECTION: {ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
