"""Compute aggregate field-evaluation metrics from per-class results."""
import json

with open('data/model_metrics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

per_class = data.get('project_evaluation', {}).get('field_evaluation', {}).get('per_class', {})
classes = list(per_class.keys())
if classes:
    macro_precision = sum(per_class[c].get('precision', 0.0) for c in classes) / len(classes)
    macro_recall = sum(per_class[c].get('recall', 0.0) for c in classes) / len(classes)
    macro_f1 = sum(per_class[c].get('f1', 0.0) for c in classes) / len(classes)
    data['metrics']['precision'] = round(macro_precision, 6)
    data['metrics']['recall'] = round(macro_recall, 6)
    data['metrics']['f1'] = round(macro_f1, 6)
    data['metrics']['note'] = (
        f"{data['metrics'].get('note', '')} "
        f"Macro-averaged precision/recall/F1 computed from {len(classes)} evaluated classes "
        f"(informal field-probe: {data['project_evaluation']['field_evaluation']['n_images']} images / {len(classes)} classes, "
        f"protocol v2, out-of-domain vs PlantVillage training data). "
        f"Not a benchmark; see models/field_eval_results.json."
    )
    print(f"Classes: {len(classes)}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
else:
    print("No per-class data found; metrics remain null.")

with open('data/model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
