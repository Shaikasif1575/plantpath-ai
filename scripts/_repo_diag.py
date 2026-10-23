"""Diagnose the GitHub PATCH 400: test auth via GET, then PATCH with error body."""
import json
import subprocess
import urllib.error
import urllib.request

REPO = "Shaikasif1575/plantpath-ai"
BASE = f"https://api.github.com/repos/{REPO}"

inp = "protocol=https\nhost=github.com\n\n"
out = subprocess.run(
    ["git", "credential", "fill"], input=inp, capture_output=True, text=True, timeout=30
).stdout
tok = next(line.split("=", 1)[1] for line in out.splitlines() if line.startswith("password="))
print("token length:", len(tok), "prefix:", tok[:6])


def call(method, url, body=None):
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "plantpath-maintenance",
    })
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]


s, got = call("GET", BASE)
print("GET:", s, (got.get("visibility") if isinstance(got, dict) else got))
print("current description:", (got.get("description") if isinstance(got, dict) else None))

# isolate: PATCH with description only
s, r = call("PATCH", BASE, {"description": "Explainable deep learning for plant disease detection: live EfficientNetV2-S inference, Grad-CAM, project-trained U-Net severity segmentation, and an honest-metrics research contract (FastAPI + Next.js)."})
print("PATCH desc-only:", s, r if s != 200 else "OK")

# isolate: topics only
s, r = call("PATCH", BASE, {"topics": ["plant-disease", "deep-learning", "explainable-ai", "grad-cam", "image-classification", "semantic-segmentation", "unet", "efficientnet", "pytorch", "fastapi", "nextjs", "research-prototype"]})
print("PATCH topics-only:", s, r if s != 200 else "OK")
