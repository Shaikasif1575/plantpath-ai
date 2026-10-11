import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.ml.torch_classifier import (  # noqa: E402
    INPUT_SIZE,
    PREPROCESSING_CANDIDATES,
    build_architecture,
)

import torch  # noqa: E402

MODEL = Path(r"C:\Users\Shaik Asif\OneDrive\Desktop\Plant Pathology\models\plant_model.pth")

state = torch.load(str(MODEL), map_location="cpu", weights_only=True)
print("checkpoint type:", type(state).__name__, "| keys:", len(state))

model = build_architecture()
try:
    model.load_state_dict(state, strict=True)
    print("STRICT LOAD: OK")
except Exception as exc:
    print("STRICT LOAD FAILED:", exc)
    sys.exit(1)

model.eval()
total = sum(p.numel() for p in model.parameters())
print("parameters:", total)
print("input size:", INPUT_SIZE)

with torch.no_grad():
    out = model(torch.zeros(1, 3, 224, 224))
print("output shape:", tuple(out.shape))

sp = sum(p.numel() for p in model.features.parameters())
cl = sum(p.numel() for p in model.classifier.parameters())
print("backbone params:", sp, "| head params:", cl)
print("preprocessing candidates:")
for k, v in PREPROCESSING_CANDIDATES.items():
    print("  ", k, "->", v["label"])