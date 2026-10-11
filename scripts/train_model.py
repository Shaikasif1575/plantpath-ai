"""Train a PlantPath AI classification model on PlantVillage-style data.

This script is the official extension point for creating the connected model.
It does NOT run automatically and does NOT fabricate results — metrics are
written to data/model_metrics.json only after a real training run.

Usage (Python 3.10–3.12 with TensorFlow installed):

    python scripts/train_model.py \
        --data-dir path/to/plantvillage/split \
        --epochs 20 \
        --output models/plantpath_model.keras

Expected data layout (Keras image_dataset_from_directory):

    split/
      Apple___Apple_scab/...
      Apple___healthy/...
      ...

The directory order determines class index order; the script writes
data/class_names.json from the actual generator so class order always
matches the model output layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PlantPath AI model")
    parser.add_argument("--data-dir", required=True, help="Training data root directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--output", default="models/plantpath_model.keras")
    parser.add_argument("--class-names-out", default="data/class_names.json")
    parser.add_argument("--metrics-out", default="data/model_metrics.json")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--base", default="mobilenet_v2", choices=["mobilenet_v2", "resnet50", "efficientnet_b0"])
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("ERROR: TensorFlow is required for training (Python 3.10–3.12).", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    img_size = tuple(args.image_size)

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.val_split,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=args.batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.val_split,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=args.batch_size,
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Persist class names so the backend always matches model output order
    Path(args.class_names_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.class_names_out).write_text(
        json.dumps(
            {
                "dataset": "PlantVillage (training run)",
                "class_names": class_names,
                "generated_by": "scripts/train_model.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Preprocessing matched to the backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights="imagenet"
    ) if args.base == "mobilenet_v2" else (
        keras.applications.ResNet50(
            input_shape=(*img_size, 3), include_top=False, weights="imagenet"
        ) if args.base == "resnet50" else
        keras.applications.EfficientNetB0(
            input_shape=(*img_size, 3), include_top=False, weights="imagenet"
        )
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*img_size, 3))
    # MobileNetV2 / ResNet50 / EfficientNet all expect ~[-1,1] or [0,1] scaling
    x = keras.applications.mobilenet_v2.preprocess_input(inputs) if args.base == "mobilenet_v2" else (
        keras.applications.resnet50.preprocess_input(inputs) if args.base == "resnet50" else inputs
    )
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="PlantPathAI")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved model -> {out_path}")

    # Write actual training history (real numbers only)
    h = history.history
    metrics = {
        "metrics": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "note": "Confusion-matrix based metrics not computed in this run. Run evaluation to populate.",
        },
        "training": {
            "training_accuracy": float(h["accuracy"][-1]),
            "validation_accuracy": float(h["val_accuracy"][-1]),
            "training_loss": float(h["loss"][-1]),
            "validation_loss": float(h["val_loss"][-1]),
            "epochs": args.epochs,
            "note": "Values written from actual training logs by scripts/train_model.py.",
        },
        "history": {k: [float(v) for v in vals] for k, vals in h.items()},
        "confusion_matrix": None,
        "class_wise": None,
        "evaluation_note": "Metrics are only displayed when present in project files.",
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
