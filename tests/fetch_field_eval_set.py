"""Fetch a labelled field-evaluation image set from Wikimedia Commons.

Unlike the small hand-curated probe set (tests/fetch_probe_set.py), this script
uses the Commons search API to gather a larger per-class set of in-the-wild
photographs. A downloaded image is only kept when its file title AND its
file-description text match the expected crop AND the expected disease, so the
ground-truth label is *verified from the uploader's description*, not assumed
from a search hit. Every image's provenance (description page, uploader,
license, matched keywords) is recorded in data/field_eval/sources.json so the
labels are auditable.

These are field photographs: out-of-domain versus the PlantVillage-style
training data of the connected classifier. Results are therefore an informal
robustness probe, not a benchmark.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "field_eval"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA = "PlantPathAI-fieldeval/1.0 (educational model evaluation)"

# PlantVillage class name -> (search queries, disease keywords, crop keywords, target count)
CLASSES = {
    "Tomato___Late_blight": (
        ["late blight tomato leaf", "Phytophthora infestans tomato leaf"],
        ["late blight", "infestans"],
        ["tomato", "lycopersicum", "tomaat"],
        7,
    ),
    "Tomato___Early_blight": (
        ["early blight tomato leaf", "Alternaria solani tomato leaf", "tomato Alternaria leaf spot",
         "Alternaria leaf spot tomato", "tomaat vroege brand"],
        ["early blight", "alternaria"],
        ["tomato", "lycopersicum", "tomaat"],
        7,
    ),
    "Tomato___Septoria_leaf_spot": (
        ["Septoria lycopersici tomato leaf", "septoria leaf spot tomato"],
        ["septoria"],
        ["tomato", "lycopersicum", "tomaat"],
        5,
    ),
    "Tomato___Leaf_Mold": (
        ["tomato leaf mold", "Passalora fulva tomato", "Cladosporium fulvum tomato"],
        ["leaf mold", "leaf mould", "passalora", "cladosporium"],
        ["tomato", "lycopersicum", "tomaat"],
        5,
    ),
    "Tomato___healthy": (
        ["healthy tomato leaves", "Solanum lycopersicum leaf"],
        ["healthy", "leaf"],
        ["tomato", "lycopersicum", "tomaat"],
        6,
    ),
    "Potato___Late_blight": (
        ["late blight potato leaf", "Phytophthora infestans potato"],
        ["late blight", "infestans"],
        ["potato", "tuberosum", "aardappel"],
        6,
    ),
    "Potato___Early_blight": (
        ["early blight potato leaf", "Alternaria solani potato", "potato Alternaria leaf lesion",
         "Solanum tuberosum early blight", "Alternaria leaf spot potato", "potato leaf Alternaria solani"],
        ["early blight", "alternaria"],
        ["potato", "tuberosum", "aardappel"],
        5,
    ),
    "Apple___Apple_scab": (
        ["apple scab leaf", "Venturia inaequalis leaf", "apple scab Malus leaf",
         "Apfelschorf Blatt", "Venturia inaequalis apple"],
        ["apple scab", "scab", "venturia", "schorf"],
        ["apple", "malus", "apfel"],
        6,
    ),
    "Corn_(maize)___Common_rust": (
        ["common rust maize leaf", "Puccinia sorghi maize", "maize rust leaf", "corn rust Puccinia",
         "Zea mays rust leaves"],
        ["common rust", "puccinia", "rust"],
        ["maize", "corn", "zea mays"],
        5,
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        ["northern corn leaf blight", "Exserohilum turcicum maize", "northern leaf blight maize",
         "Helminthosporium turcicum maize", "maize leaf blight Exserohilum"],
        ["northern leaf blight", "northern corn leaf blight", "exserohilum", "helminthosporium"],
        ["maize", "corn", "zea mays"],
        5,
    ),
    "Grape___Black_rot": (
        ["grape black rot leaf", "Guignardia bidwellii grape", "black rot grapevine leaf",
         "grapevine black rot Phyllosticta", "Vitis black rot"],
        ["black rot", "guignardia", "phyllosticta"],
        ["grape", "vitis", "vine", "wein"],
        5,
    ),
}

BAD_PATTERNS = re.compile(
    r"\b(logo|icon|map|diagram|chart|drawing|illustration|painting|stamp|coin|sign|label)\b"
)
# Hard rejects: not photographs of leaf tissue (protocol requires leaf imagery
# for a leaf-classifier evaluation).
MICROSCOPY_PATTERNS = re.compile(
    r"\b(sem|electron|microscope|mikroskop|mikroskoo|spore|spores|hyphae|mycelium|fruiting body)\b"
)
# The image must be verifiably about leaf tissue (multi-language leaf keywords).
LEAF_PATTERNS = re.compile(r"leaf|leaves|laub|blad|blätter|blatt|foliage")


def api_get(params: dict, retries: int = 4) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            if "429" in str(exc) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  rate-limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def search_candidates(query: str, limit: int = 30) -> list[dict]:
    data = api_get(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "iiurlwidth": "1024",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if (info.get("width") or 0) < 640:
            continue
        meta = info.get("extmetadata", {})
        out.append(
            {
                "title": page.get("title", ""),
                "thumburl": info.get("thumburl") or info.get("url"),
                "descriptionurl": info.get("descriptionurl"),
                "license": strip_html(meta.get("LicenseShortName", {}).get("value", "")),
                "author": strip_html(meta.get("Artist", {}).get("value", ""))[:200],
                "description": strip_html(meta.get("ImageDescription", {}).get("value", ""))[:600],
                "date": strip_html(meta.get("DateTimeOriginal", {}).get("value", ""))[:40],
            }
        )
    return out


def verify(cand: dict, disease_kw: list[str], crop_kw: list[str]) -> list[str] | None:
    """Return matched keywords if title+description verify the label, else None.

    Protocol requirements (all mechanical, all auditable from sources.json):
    1. crop and disease keywords appear in title+description (label verified);
    2. no non-leaf-imagery markers (electron microscopy etc.);
    3. leaf tissue is mentioned (leaf/laub/blad/...), because the classifier
       classifies leaves - fruit, tree-landscape and lab images are out of
       protocol, not out of convenience.
    """
    text = (cand["title"] + " " + cand["description"]).lower()
    if BAD_PATTERNS.search(text):
        return None
    if MICROSCOPY_PATTERNS.search(text):
        return None
    if not LEAF_PATTERNS.search(text):
        return None
    disease = [k for k in disease_kw if k in text]
    crop = [k for k in crop_kw if k in text]
    if disease and crop:
        return [disease[0], crop[0]]
    return None


def main() -> None:
    sources_path = OUT / "sources.json"
    # Resume: keep previously fetched images and only fill the missing quota.
    sources: dict[str, dict] = {}
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    stats: dict[str, dict] = {}
    for cls, (queries, disease_kw, crop_kw, target) in CLASSES.items():
        have = sorted(k for k, v in sources.items() if v.get("ground_truth") == cls)
        kept = len(have)
        tried = 0
        seen_titles = {sources[k].get("commons_title") for k in have}
        if kept >= target:
            stats[cls] = {"kept": kept, "candidates_tried": 0, "target": target}
            print(f"{cls:38s} kept={kept}/{target} (already fetched)")
            continue
        for query in queries:
            if kept >= target:
                break
            try:
                cands = search_candidates(query)
            except Exception as exc:
                print(f"  SEARCH FAIL [{cls}] {query!r}: {type(exc).__name__} {exc}")
                continue
            for cand in cands:
                if kept >= target or cand["title"] in seen_titles:
                    continue
                seen_titles.add(cand["title"])
                tried += 1
                matched = verify(cand, disease_kw, crop_kw)
                if not matched:
                    continue
                local = f"{cls}__{kept:02d}.jpg"
                try:
                    req = urllib.request.Request(cand["thumburl"], headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        blob = r.read()
                    (IMG_DIR / local).write_bytes(blob)
                except Exception as exc:
                    print(f"  DL FAIL {local}: {type(exc).__name__} {exc}")
                    continue
                sources[local] = {
                    "ground_truth": cls,
                    "commons_title": cand["title"],
                    "commons_page": cand["descriptionurl"],
                    "license": cand["license"],
                    "author": cand["author"],
                    "description": cand["description"],
                    "date_original": cand["date"],
                    "matched_query": query,
                    "verified_keywords": matched,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                kept += 1
                time.sleep(0.6)
        stats[cls] = {"kept": kept, "candidates_tried": tried, "target": target}
        print(f"{cls:38s} kept={kept}/{target} (verified from {tried} new candidates)")

    (OUT / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "fetch_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    total = sum(s["kept"] for s in stats.values())
    print(f"\nwrote {len(sources)} images + sources.json ({total} total) -> {OUT}")


if __name__ == "__main__":
    main()
