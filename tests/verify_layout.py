"""Quick visual verification of the scan-result layout fixes."""
from PIL import Image, ImageDraw
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "browser_report"
OUT.mkdir(exist_ok=True)

img = Image.new("RGB", (400, 400), (56, 142, 60))
d = ImageDraw.Draw(img)
d.ellipse((60, 40, 340, 360), fill=(46, 125, 50), outline=(27, 94, 32), width=6)
img.save(OUT / "verify_leaf.png")

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, channel="chrome")
    page = b.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:3000/scan", wait_until="networkidle")
    page.set_input_files("input[type=file]", str(OUT / "verify_leaf.png"))
    page.get_by_role("button", name="Analyze").click()
    page.wait_for_selector("text=Analysis Complete", timeout=30000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / "final-scan-result.png"), full_page=True, animations="disabled")

    n = page.get_by_text("Research disclaimer").count()
    print("disclaimer instances:", n)

    js = """() => {
      const heads = [...document.querySelectorAll('h3')];
      return heads.filter(h => /Grad-CAM|Severity Assessment/.test(h.textContent))
        .map(h => {
          const card = h.closest('section, div');
          const r = h.parentElement.parentElement.getBoundingClientRect();
          return {t: h.textContent.trim().slice(0, 22), w: Math.round(r.width), x: Math.round(r.x)};
        });
    }"""
    print("cards:", page.evaluate(js))
    b.close()
