"""Quick check: classifier class-name parses -> knowledge records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.disease_service import DiseaseService  # noqa: E402

ds = DiseaseService(Path(__file__).resolve().parents[1] / "data" / "diseases.json")
CASES = [
    ("Tomato", "Late blight", "diseased", "tomato-late-blight"),
    ("Pepper", "bell Bacterial spot", "diseased", "pepper-bacterial-spot"),
    ("Tomato", "Spider mites Two-spotted spider mite", "diseased", "tomato-spider-mites"),
    ("Orange", "Huanglongbing", "diseased", "orange-hlb-citrus-greening"),
    ("Corn", "Cercospora leaf spot gray leaf spot", "diseased", "corn-cercospora-gray-leaf-spot"),
    ("Grape", "Esca (Black measles)", "diseased", "grape-esca"),
    ("Apple", "Apple scab", "diseased", "apple-apple-scab"),
    ("Strawberry", "Leaf scorch", "diseased", "strawberry-leaf-scorch"),
    ("Tomato", "Yellow Leaf Curl Virus", "diseased", "tomato-yellow-leaf-curl-virus"),
    ("Apple", "Healthy", "healthy", "apple-healthy"),
    ("Squash", "Healthy", "healthy", None),          # no healthy squash record
    ("Apple", "Totally Unknown Thing", "diseased", None),  # no confident match
]
fails = 0
for crop, disease, condition, expected in CASES:
    rec = ds.find_for_prediction(crop, disease, condition)
    got = rec["id"] if rec else None
    ok = got == expected
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {crop}/{disease!r} -> {got} (expected {expected})")
    if rec and condition == "diseased":
        assert rec["treatment"], f"empty treatment for {got}"
        assert rec["disclaimer"], f"missing disclaimer for {got}"
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
