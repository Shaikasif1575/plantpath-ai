"""Train the U-Net severity segmentation models (leaf region + lesion region).

Two small binary U-Nets are trained on CPU with PyTorch, using public
Apache-2.0 datasets downloaded by ``scripts/download_seg_data.py``:

* ``models/_segdata/lesion/`` - in-the-wild leaf images with lesion masks that
  are SAM2 model pseudo-labels (dataset ``godoldol99/SAM2-based-plant-disease-
  lesion-segmentation-model-DATASET``). NOT human annotations; this limitation
  is recorded in every checkpoint and in models/segmentation_audit.json.
* ``models/_segdata/leaf/`` - in-the-wild leaf images with binary
  leaf-vs-background masks (dataset ``LeafNet75/Leaf-Disease-Segmentation-
  masks``).

The classifier is NOT trained, retrained or fine-tuned by this script. Output:
``models/segmentation/{lesion,leaf}_unet.pth`` + ``meta.json`` consumed by
``backend/app/ml/segmentation.py``.

Usage (from repo root):
  python scripts/train_segmentation.py                 # both models
  python scripts/train_segmentation.py --task lesion --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.unet import UNet  # noqa: E402  (single architecture source of truth)

DATA_ROOT = ROOT / "models" / "_segdata"
OUT_DIR = ROOT / "models" / "segmentation"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATASET_META = {
    "lesion": {
        "repo": "godoldol99/SAM2-based-plant-disease-lesion-segmentation-model-DATASET",
        "license": "apache-2.0",
        "label_source": "SAM2 model pseudo-labels (per dataset name/description), not human annotation",
        "target": "diseased lesion region",
    },
    "leaf": {
        "repo": "LeafNet75/Leaf-Disease-Segmentation-masks",
        "license": "apache-2.0",
        "label_source": "binary leaf-vs-background masks as published in the dataset",
        "target": "leaf region",
    },
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class SegPairDataset(Dataset):
    """Image/mask pairs with paired geometric augmentation."""

    def __init__(self, pairs: list[tuple[Path, Path]], size: int, augment: bool) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(mask_path).convert("L")
        if img.size != msk.size:
            msk = msk.resize(img.size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img, msk = img.transpose(Image.FLIP_LEFT_RIGHT), msk.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img, msk = img.transpose(Image.FLIP_TOP_BOTTOM), msk.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k:
                img, msk = img.rotate(90 * k), msk.rotate(90 * k)

        img = img.resize((self.size, self.size), Image.BILINEAR)
        msk = msk.resize((self.size, self.size), Image.NEAREST)

        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            # photometric jitter (image only)
            alpha = random.uniform(0.85, 1.15)  # brightness
            beta = random.uniform(0.85, 1.15)   # contrast
            x = np.clip((x - 0.5) * beta + 0.5, 0.0, 1.0) * alpha
            x = np.clip(x, 0.0, 1.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        # Masks in these datasets are binary but NOT always 0/255 (observed values
        # are 0/1 and 0/38), so binarize by "nonzero" rather than "above 127".
        y = (np.asarray(msk, dtype=np.float32) > 0).astype(np.float32)

        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        yt = torch.from_numpy(y[None, ...])
        return xt, yt


def collect_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(data_dir.glob("*.jpg")):
        msk = img.with_suffix(".png")
        if msk.exists():
            pairs.append((img, msk))
    return pairs


def dice_coeff(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter / denom) if denom > 0 else 1.0


def iou_coeff(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter / union) if union > 0 else 1.0


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + dice


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    dices, ious, accs = [], [], []
    for xb, yb in loader:
        logits = model(xb)
        dices.append(dice_coeff(logits, yb))
        ious.append(iou_coeff(logits, yb))
        pred = (torch.sigmoid(logits) > 0.5).float()
        accs.append(((pred == yb).float().mean()).item())
    return {
        "dice": round(float(np.mean(dices)), 4),
        "iou": round(float(np.mean(ious)), 4),
        "pixel_accuracy": round(float(np.mean(accs)), 4),
    }


def train_task(task: str, args) -> dict:
    data_dir = DATA_ROOT / task
    pairs = collect_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"No pairs in {data_dir}; run scripts/download_seg_data.py first.")
    log(f"[{task}] {len(pairs)} pairs in {data_dir}")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(8, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        val_pairs = val_pairs[: max(4, args.limit // 5)]
    log(f"[{task}] train={len(train_pairs)} val={len(val_pairs)} (seed={args.seed})")

    train_ds = SegPairDataset(train_pairs, args.size, augment=True)
    val_ds = SegPairDataset(val_pairs, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    arch = {"n_channels": 3, "base": args.base, "depth": args.depth, "n_classes": 1, "input_size": args.size}
    model = UNet(**{k: arch[k] for k in ("n_channels", "base", "depth", "n_classes")})
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{task}] UNet params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_dice, best_state, best_epoch, history = -1.0, None, -1, []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"{task}_unet.pth"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = bce_dice_loss(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), **val})
        log(f"[{task}] epoch {epoch:2d}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_dice={val['dice']:.4f} val_iou={val['iou']:.4f}")
        if val["dice"] > best_dice:
            best_dice, best_epoch = val["dice"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Save immediately so an interrupted run still leaves the best-so-far model.
            torch.save(
                {
                    "arch": arch,
                    "state_dict": best_state,
                    "task": task,
                    "val_metrics": history[-1],
                    "dataset": DATASET_META[task],
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "epochs_run": epoch,
                    "seed": args.seed,
                },
                ckpt_path,
            )
            log(f"[{task}] saved checkpoint @ epoch {epoch} (val_dice={val['dice']:.4f})")

    elapsed = round(time.time() - t0)
    log(f"[{task}] best val_dice={best_dice:.4f} @ epoch {best_epoch} ({elapsed}s total)")
    return {
        "file": ckpt_path.name,
        "task": task,
        "arch": arch,
        "params": n_params,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_val_metrics": next(h for h in history if h["epoch"] == best_epoch),
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "dataset": DATASET_META[task],
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["both", "lesion", "leaf"], default="both")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train pairs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tasks = ["lesion", "leaf"] if args.task == "both" else [args.task]
    results: dict = {}
    for t in tasks:
        results[t] = train_task(t, args)
        # Refresh meta.json after each task so an interrupted run still leaves
        # a consistent serving contract for whatever finished.
        _write_meta(results, args)

    _write_audit(results)


def _write_meta(results: dict, args) -> None:
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(
        {
            "version": meta.get("version", 0) + 1,
            "framework": "pytorch",
            "architecture": "unet",
            "input_size": [args.size, args.size],
            "threshold": 0.5,
            "preprocessing": "resize to square, /255, ImageNet mean/std normalization",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_script": "scripts/train_segmentation.py",
            "hardware": "cpu",
            "limitations": [
                "Lesion masks in the training set are SAM2 model pseudo-labels, not human annotations.",
                "Leaf masks are dataset-provided binary leaf-vs-background masks.",
                "Severity percentages are model estimates on in-the-wild photographs and are not "
                "clinically or agronomically validated.",
            ],
        }
    )
    for t, res in results.items():
        meta[t] = res
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"updated {meta_path}")


def _write_audit(results: dict) -> None:
    audit = {
        "generated_by": "scripts/train_segmentation.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "severity assessment (affected leaf area %) for PlantPath AI",
        "models": {
            t: {
                "checkpoint": f"models/segmentation/{res['file']}",
                "dataset_repo": res["dataset"]["repo"],
                "dataset_license": res["dataset"]["license"],
                "label_source": res["dataset"]["label_source"],
                "train_pairs": res["train_pairs"],
                "val_pairs": res["val_pairs"],
                "best_val_metrics": res["best_val_metrics"],
                "epochs_run": res["epochs_run"],
                "hardware": "cpu",
            }
            for t, res in results.items()
        },
        "honesty_notes": [
            "Validation metrics are computed against the same label source used for training "
            "(pseudo-labels for lesion), so they measure fit to that labeler, not agreement with humans.",
            "The disease classifier was not trained, retrained or fine-tuned as part of this work.",
        ],
    }
    audit_path = ROOT / "models" / "segmentation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log(f"updated {audit_path}")


if __name__ == "__main__":
    main()
