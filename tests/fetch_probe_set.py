"""Download a labelled probe set from Wikimedia Commons.

Each image's identity is verified from its Commons file-description page, so the
ground-truth label used for calibration is documented, not assumed.
"""
import json
import os
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_set")
os.makedirs(OUT, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA = "PlantPathAI-calibration/1.0 (educational model evaluation)"

PROBES = [
    # (local name, commons File title, ground-truth label)
    ("healthy_tomato_1.jpg", "File:Healthy tomato leaves (7871755330).jpg", "healthy"),
    ("diseased_tomato_lateblight_1.jpg", "File:Late blight on tomato leaf (7871756748).jpg", "diseased"),
    ("diseased_tomato_earlyblight_1.jpg", "File:Early blight on tomato leaves (7871930010).jpg", "diseased"),
    ("diseased_tomato_phytophthora_1.jpg", "File:Tomato with Phytophthora infestans (late blight).jpg", "diseased"),
    ("diseased_tomato_leafmold_1.jpg", "File:Tomaat bladvlekkenziekte.jpg", "diseased"),
]


def api_url_imageinfo(title):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "1024",
    }
    return API + "?" + urllib.parse.urlencode(params)


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


sources = {}
for local, title, truth in PROBES:
    try:
        data = fetch_json(api_url_imageinfo(title))
        pages = data.get("query", {}).get("pages", {})
        page = list(pages.values())[0]
        if "imageinfo" not in page:
            print("MISSING:", title)
            continue
        info = page["imageinfo"][0]
        thumb = info.get("thumburl") or info.get("url")
        meta = info.get("extmetadata", {})
        desc = (meta.get("ImageDescription", {}).get("value") or "").strip()
        lic = (meta.get("LicenseShortName", {}).get("value") or "").strip()
        artist = (meta.get("Artist", {}).get("value") or "").strip()
        req = urllib.request.Request(thumb, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            data_bytes = r.read()
        with open(os.path.join(OUT, local), "wb") as fh:
            fh.write(data_bytes)
        sources[local] = {
            "commons_title": title,
            "commons_page": info.get("descriptionurl"),
            "file_url": info.get("url"),
            "description": desc[:400],
            "license": lic,
            "author": artist[:200],
            "ground_truth": truth,
        }
        print(f"OK {local:42s} [{truth:9s}] {len(data_bytes)//1024} KB  lic={lic}")
        time.sleep(0.4)
    except Exception as exc:
        print("FAIL", local, type(exc).__name__, exc)

with open(os.path.join(OUT, "sources.json"), "w", encoding="utf-8") as fh:
    json.dump(sources, fh, indent=2, ensure_ascii=False)
print("wrote sources.json with", len(sources), "entries")