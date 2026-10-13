import httpx
import re

h = httpx.get("http://127.0.0.1:3000/model", timeout=30).text
print("Severity Segmentation in html:", "Severity Segmentation" in h)
print("0.6845 in html:", "0.6845" in h)
m = re.search(r"Segmentation</div><div class=\"mt-1 font-mono text-xl font-semibold[^\"]*\">([^<]+)", h)
print("segmentation card value:", m.group(1) if m else "pattern not found")
