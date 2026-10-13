"""Inspect candidate lesion-segmentation datasets on Hugging Face.

Reports file-tree structure and downloads a few sample image+mask pairs to
verify the masks are real lesion/leaf annotations (non-trivial pixel content),
not empty or placeholder files. Nothing here is used for training yet.
"""
from __future__ import annotations

import io
import json
import sys

import httpx

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CANDIDATES = [
    "godoldol99/SAM2-based-plant-disease-lesion-segmentation-model-DATASET",
    "LeafNet75/Leaf-Disease-Segmentation-masks",
]

API = "https://huggingface.co/api/datasets"


def tree(client: httpx.Client, repo: str, path: str = "") -> list[dict]:
    params = {"recursive": "true", "expand": "false"}
    r = client.get(f"{API}/{repo}/tree/main/{path}" if path else f"{API}/{repo}/tree/main",
                   params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> None:
    with httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": "plantpath-research"}) as client:
        for repo in CANDIDATES:
            print("=" * 30, repo)
            try:
                entries = tree(client, repo)
            except Exception as exc:
                print("  TREE FAIL:", type(exc).__name__, exc)
                continue
            dirs: dict[str, int] = {}
            tops: list[str] = []
            for e in entries:
                p = e.get("path", "")
                if e.get("type") == "directory":
                    tops.append(p + "/")
                    continue
                top = p.split("/")[0]
                dirs[top] = dirs.get(top, 0) + 1
            print("  top-level dirs:", {k: v for k, v in dirs.items()})
            # Show a few sample paths from each non-trivial dir
            by_dir: dict[str, list[str]] = {}
            for e in entries:
                if e.get("type") == "file":
                    p = e["path"]
                    d = "/".join(p.split("/")[:-1]) or "."
                    by_dir.setdefault(d, []).append(p)
            for d, files in by_dir.items():
                if d in (".",) or files[0].endswith((".md", ".gitattributes", ".json", ".txt")):
                    if len(files) <= 4:
                        print(f"  [{d}] {files}")
                    continue
                print(f"  [{d}] {len(files)} files, e.g. {files[:3]}")

            # README
            try:
                rr = client.get(f"https://huggingface.co/datasets/{repo}/raw/main/README.md", timeout=60)
                if rr.status_code == 200:
                    txt = rr.text
                    keep = [ln for ln in txt.splitlines() if ln.strip()][:40]
                    print("  README head:")
                    for ln in keep[:25]:
                        print("   ", ln[:120])
            except Exception as exc:
                print("  README fail:", exc)

            # Sample a mask file (first file under any masks-ish dir) and an image
            mask_files = [e["path"] for e in entries
                          if e.get("type") == "file" and "mask" in e["path"].lower()
                          and not e["path"].endswith((".md", ".json"))]
            img_files = [e["path"] for e in entries
                         if e.get("type") == "file" and e["path"].lower().endswith((".jpg", ".jpeg", ".png"))
                         and "mask" not in e["path"].lower()]
            print(f"  mask-like files: {len(mask_files)} e.g. {mask_files[:2]}")
            print(f"  image files: {len(img_files)} e.g. {img_files[:2]}")

            try:
                from PIL import Image
                import numpy as np
                if mask_files:
                    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{mask_files[0]}"
                    b = client.get(url, timeout=120).content
                    m = np.asarray(Image.open(io.BytesIO(b)))
                    print(f"  sample mask {mask_files[0]}: shape={m.shape} dtype={m.dtype} "
                          f"min={m.min()} max={m.max()} nonzero%={100.0 * (m > 0).mean():.1f}")
                if img_files:
                    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{img_files[0]}"
                    b = client.get(url, timeout=120).content
                    im = Image.open(io.BytesIO(b))
                    print(f"  sample image {img_files[0]}: {im.size} {im.mode}")
            except Exception as exc:
                print("  sample fetch fail:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
