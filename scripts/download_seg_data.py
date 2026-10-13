"""Download the two Apache-2.0 segmentation datasets used for severity training.

1. godoldol99/...  -> lesion masks (SAM2 pseudo-labels) -> models/_segdata/lesion/
2. LeafNet75/...   -> leaf vs background masks (parquet) -> models/_segdata/leaf/

Pairing is by numeric id (images/NNNNN.jpg <-> masks/NNNNN.png). Only pairs
present in both folders are kept. Provenance is recorded in
models/_segdata/dataset_sources.json.
"""
from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "_segdata"
LESION = OUT / "lesion"
LEAF = OUT / "leaf"
LESION.mkdir(parents=True, exist_ok=True)
LEAF.mkdir(parents=True, exist_ok=True)

HF = "https://huggingface.co/datasets"   # for /resolve/main file downloads
HFAPI = "https://huggingface.co/api"     # for tree listings
GOD = "godoldol99/SAM2-based-plant-disease-lesion-segmentation-model-DATASET"
LEAFNET = "LeafNet75/Leaf-Disease-Segmentation-masks"

HEADERS = {"User-Agent": "plantpath-research"}


def fetch(client: httpx.Client, url: str, timeout: int = 180) -> bytes:
    r = client.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def download_pair(client: httpx.Client, stem: str) -> tuple[str, bool, str]:
    try:
        img = fetch(client, f"{HF}/{GOD}/resolve/main/images/{stem}.jpg")
        msk = fetch(client, f"{HF}/{GOD}/resolve/main/masks/{stem}.png")
        (LESION / f"{stem}.jpg").write_bytes(img)
        (LESION / f"{stem}.png").write_bytes(msk)
        return stem, True, ""
    except Exception as exc:
        return stem, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    with httpx.Client(follow_redirects=True, timeout=180, headers=HEADERS) as client:
        # --- lesion dataset: list tree, pair stems ---
        r = client.get(f"{HFAPI}/datasets/{GOD}/tree/main", params={"recursive": "true"}, timeout=120)
        r.raise_for_status()
        entries = r.json()
        img_stems = {e["path"].split("/")[-1].rsplit(".", 1)[0]
                     for e in entries if e.get("type") == "file" and e["path"].startswith("images/")}
        msk_stems = {e["path"].split("/")[-1].rsplit(".", 1)[0]
                     for e in entries if e.get("type") == "file" and e["path"].startswith("masks/")}
        stems = sorted(img_stems & msk_stems)
        print(f"lesion: {len(img_stems)} images, {len(msk_stems)} masks -> {len(stems)} pairs")
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (stem, good, err) in enumerate(pool.map(lambda s: download_pair(client, s), stems), 1):
                ok += good
                fail += (not good)
                if i % 100 == 0 or i == len(stems):
                    print(f"  lesion {i}/{len(stems)} (ok={ok} fail={fail})")
        print(f"lesion done: {ok} pairs, {fail} failures")

        # --- leaf dataset: parquet ---
        try:
            import pandas as pd
            b = fetch(client, f"{HF}/{LEAFNET}/resolve/main/data/train-00000-of-00001.parquet", timeout=600)
            df = pd.read_parquet(io.BytesIO(b))
            print("leaf parquet rows:", len(df), "cols:", list(df.columns))
            from PIL import Image
            saved = 0
            for idx, row in df.iterrows():
                try:
                    im = row["image"]
                    mk = row["mask"]
                    # HF parquet rows store dicts with {'bytes': b, 'path': p} or PIL images
                    def to_pil(x):
                        if isinstance(x, dict) and "bytes" in x:
                            return Image.open(io.BytesIO(x["bytes"]))
                        if isinstance(x, (bytes, bytearray)):
                            return Image.open(io.BytesIO(x))
                        return x
                    im, mk = to_pil(im), to_pil(mk)
                    stem = f"{saved:04d}"
                    im.convert("RGB").save(LEAF / f"{stem}.jpg", quality=92)
                    mk.convert("L").save(LEAF / f"{stem}.png")
                    saved += 1
                except Exception as exc:
                    print("  leaf row fail", idx, type(exc).__name__, exc)
                if saved and saved % 100 == 0:
                    print(f"  leaf {saved} saved")
            print("leaf done:", saved, "pairs")
        except ImportError:
            print("leaf: pandas/pyarrow not installed in this env — will install then retry")

        (OUT / "dataset_sources.json").write_text(json.dumps({
            "lesion": {
                "repo": GOD, "license": "apache-2.0",
                "note": "Lesion masks are SAM2 model pseudo-labels (per dataset name), not human annotations.",
                "pairs": ok,
            },
            "leaf": {
                "repo": LEAFNET, "license": "apache-2.0",
                "note": "In-the-wild leaf images with binary leaf-vs-background masks.",
            },
        }, indent=2), encoding="utf-8")
        print("wrote dataset_sources.json")


if __name__ == "__main__":
    main()
