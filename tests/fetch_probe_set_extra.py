"""Extend the probe set with additional labelled images (verified Commons titles)."""
import json
import os
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_set")
os.makedirs(OUT, exist_ok=True)
API = "https://commons.wikimedia.org/w/api.php"
UA = "PlantPathAI-calibration/1.0 (educational model evaluation)"

EXTRA = [
    ("healthy_tomato_2.jpg", "File:Tomato leaf.jpg", "healthy"),
    ("healthy_tomato_3.jpg", "File:Tomato plant leaves.jpg", "healthy"),
]


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


src_path = os.path.join(OUT, "sources.json")
with open(src_path, "r", encoding="utf-8") as fh:
    sources = json.load(fh)

for local, title, truth in EXTRA:
    try:
        params = {
            "action": "query", "format": "json", "titles": title,
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1024",
        }
        data = fetch_json(API + "?" + urllib.parse.urlencode(params))
        page = list(data.get("query", {}).get("pages", {}).values())[0]
        if "imageinfo" not in page:
            print("MISSING:", title, "->", page.get("title"))
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        thumb = info.get("thumburl") or info.get("url")
        req = urllib.request.Request(thumb, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            b = r.read()
        with open(os.path.join(OUT, local), "wb") as fh:
            fh.write(b)
        sources[local] = {
            "commons_title": title,
            "commons_page": info.get("descriptionurl"),
            "file_url": info.get("url"),
            "description": (meta.get("ImageDescription", {}).get("value") or "").strip()[:400],
            "license": (meta.get("LicenseShortName", {}).get("value") or "").strip(),
            "author": (meta.get("Artist", {}).get("value") or "").strip()[:200],
            "ground_truth": truth,
        }
        print(f"OK {local:32s} [{truth}] {len(b)//1024} KB")
        time.sleep(0.4)
    except Exception as exc:
        print("FAIL", local, type(exc).__name__, exc)

with open(src_path, "w", encoding="utf-8") as fh:
    json.dump(sources, fh, indent=2, ensure_ascii=False)
print("total entries:", len(sources))