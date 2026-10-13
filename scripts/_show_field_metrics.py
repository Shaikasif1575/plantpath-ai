import json
import pathlib

fe = json.loads(pathlib.Path("models/field_eval_results.json").read_text(encoding="utf-8"))
m = fe["metrics"]
for cls, v in m["per_class"].items():
    print(f"{cls:34s} n={v['support']} correct={v['correct']} "
          f"P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}")
print("overall:", {k: m[k] for k in ("n_images", "n_ground_truth_classes",
                                      "exact_class_accuracy", "crop_accuracy",
                                      "mean_confidence", "low_confidence_rate")})
