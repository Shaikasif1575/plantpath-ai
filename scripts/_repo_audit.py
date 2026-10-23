"""Set GitHub repo description/topics (with correct Content-Type) + audit pushed tree."""
import json
import subprocess
import urllib.request

REPO = "Shaikasif1575/plantpath-ai"
BASE = f"https://api.github.com/repos/{REPO}"

inp = "protocol=https\nhost=github.com\n\n"
out = subprocess.run(
    ["git", "credential", "fill"], input=inp, capture_output=True, text=True, timeout=30
).stdout
tok = next(line.split("=", 1)[1] for line in out.splitlines() if line.startswith("password="))

HEADERS = {
    "Authorization": f"Bearer {tok}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "plantpath-maintenance",
}


def call(method, url, body=None):
    headers = dict(HEADERS)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, headers=headers, data=data)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


repo = call("PATCH", BASE, {
    "description": (
        "Explainable deep learning for plant disease detection: live "
        "EfficientNetV2-S inference, Grad-CAM, project-trained U-Net severity "
        "segmentation, field evaluation with honest-metrics contract "
        "(FastAPI + Next.js)."
    ),
    "topics": [
        "plant-disease", "deep-learning", "explainable-ai", "grad-cam",
        "image-classification", "semantic-segmentation", "unet", "efficientnet",
        "pytorch", "fastapi", "nextjs", "research-prototype",
    ],
})
print("description:", repo["description"])
print("topics:", ", ".join(repo["topics"]))
print("visibility:", repo["visibility"])

tree = call("GET", BASE + "/git/trees/main?recursive=1")
blobs = [t for t in tree.get("tree", []) if t.get("type") == "blob"]
total = sum(t.get("size", 0) for t in blobs)
paths = {t["path"] for t in blobs}
print(f"\nremote tree: {len(paths)} files, {total / 1_048_576:.1f} MB, truncated={tree.get('truncated')}")
KEY = [
    "README.md",
    "backend/app/ml/segmentation.py",
    "models/efficientnet_v2_s_best.pth",
    "models/segmentation/lesion_unet.pth",
    "models/segmentation/leaf_unet.pth",
    "models/segmentation/meta.json",
    "models/segmentation_audit.json",
    "models/field_eval_results.json",
    "models/severity_sanity_results.json",
    "data/field_eval/sources.json",
    "data/model_metrics.json",
    "tests/evaluate_severity_sanity.py",
    ".dockerignore",
]
missing = [k for k in KEY if k not in paths]
for k in KEY:
    print(f"  {'OK     ' if k in paths else 'MISSING'} {k}")
print("missing:", missing or "none")
