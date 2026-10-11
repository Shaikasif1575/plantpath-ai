"""Diagnose why the loaded checkpoint returns a constant output."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.ml.torch_classifier import build_architecture, preprocess_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
state = torch.load(str(ROOT / "models" / "plant_model.pth"), map_location="cpu", weights_only=True)
model = build_architecture()
model.load_state_dict(state, strict=True)
model.eval()

print("=== weight / BN statistics ===")
zero_params, tiny_params = [], []
for n, p in model.named_parameters():
    a = p.detach().abs()
    if float(a.max()) == 0.0:
        zero_params.append(n)
    elif float(a.max()) < 1e-6:
        tiny_params.append((n, float(a.max())))

print("all-zero parameter tensors:", len(zero_params))
for n in zero_params[:25]:
    print("   ZERO:", n)
print("near-zero (<1e-6) tensors:", len(tiny_params))
for n, v in tiny_params[:25]:
    print("   TINY:", n, v)

print("\n=== BatchNorm running stats (first 8 conv BNs) ===")
n_bn = 0
for mod_name, mod in model.named_modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        n_bn += 1
        if n_bn <= 8:
            rm = mod.running_mean
            rv = mod.running_var
            w = mod.weight
            print(
                f"  {mod_name:34s} mean|{float(rm.abs().mean()):9.5f} "
                f"var|{float(rv.abs().mean()):12.6f} w|{float(w.abs().mean()):8.5f} "
                f"running_mean_zero={bool((rm == 0).all())} running_var_one={bool((rv == 1).all())}"
            )
print("  total BN layers:", n_bn)

print("\n=== classifier head weights ===")
for n, p in model.named_parameters():
    if n.startswith("classifier"):
        print(f"  {n:26s} shape={tuple(p.shape)} absmax={float(p.abs().max()):.6f} std={float(p.std()):.6f}")

print("\n=== forward pass with hooks (eval mode) ===")
acts = {}


def hook(name):
    def fn(_m, _i, o):
        t = o.detach()
        acts[name] = (tuple(t.shape), float(t.mean()), float(t.std()), float(t.min()), float(t.max()))
    return fn


hooks = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Conv2d) and name in (
        "features.0.0", "features.2.conv.0.0", "features.9.conv.1.0",
        "features.17.conv.1.0", "features.18.0",
    ):
        hooks.append(mod.register_forward_hook(hook(name)))

imgs = {
    "lateblight": ROOT / "tests" / "probe_set" / "diseased_tomato_lateblight_1.jpg",
    "healthy": ROOT / "tests" / "probe_set" / "healthy_tomato_1.jpg",
    "black": None,
}
black = np.zeros((300, 300, 3), dtype="uint8")
outs = {}
with torch.no_grad():
    for key, path in imgs.items():
        if key == "black":
            arr = black
        else:
            arr = cv2.imread(str(path))
        acts.clear()
        x = torch.from_numpy(preprocess_image(arr, "imagenet_normalized").astype("float32"))
        y = model(x)
        outs[key] = y.numpy().copy()
        print(f"\n  [{key}] logits={y.numpy().round(6).tolist()}")
        for n in ("features.0.0", "features.2.conv.0.0", "features.9.conv.1.0", "features.17.conv.1.0", "features.18.0"):
            if n in acts:
                print(f"     {n:26s} shape={acts[n][0]} mean={acts[n][1]:+.6f} std={acts[n][2]:.6f} min={acts[n][3]:+.4f} max={acts[n][4]:+.4f}")

for h in hooks:
    h.remove()

print("\n=== output comparison ===")
for k, v in outs.items():
    print(f"  {k:12s} {v.round(6).tolist()}")
print("  lateblight vs healthy identical:", np.allclose(outs["lateblight"], outs["healthy"]))
print("  lateblight vs black identical:  ", np.allclose(outs["lateblight"], outs["black"]))

print("\n=== try train mode (batch stats instead of running stats) ===")
model.train()
with torch.no_grad():
    for key, path in imgs.items():
        arr = black if key == "black" else cv2.imread(str(path))
        x = torch.from_numpy(preprocess_image(arr, "imagenet_normalized").astype("float32"))
        print(f"  [{key}] logits={model(x).numpy().round(6).tolist()}")
model.eval()