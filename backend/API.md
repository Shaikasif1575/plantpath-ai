# PlantPath AI API documentation

Base URL (local): `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## GET /api/health

Returns backend status, model load state, and demo-mode flag.

```json
{
  "status": "ok",
  "model_loaded": false,
  "demo_mode": true,
  "model_name": "PlantPath AI",
  "model_version": "1.0.0",
  "gradcam_available": false,
  "severity_available": false,
  "leaf_validation_available": false,
  "class_count": 38,
  "timestamp": "2026-09-30T12:00:00+00:00",
  "detail": "Demo mode: predictions are simulated."
}
```

## GET /api/model-info

Model architecture, metrics (if available), training history (if available),
class names, and model-comparison slots.

Fields that are not present in project files are returned as `null` — they are
never invented.

## GET /api/crops

Dynamically derived crop explorer from the connected class-name mapping.

## GET /api/diseases

Knowledge base of plant diseases. Query params:
- `crop` — filter by crop name
- `condition` — `healthy` or `diseased`
- `search` — free-text search

## GET /api/diseases/{id}

Single disease record.

## POST /api/predict

`multipart/form-data` with field `image`.

Accepts JPG, JPEG, PNG, WEBP, BMP. Enforces `MAX_UPLOAD_SIZE_MB`.

Returns:
- validation + quality warnings
- crop / disease / condition
- top prediction + top-3 probabilities
- confidence band (low / medium / high)
- model metadata
- explainability availability (real Grad-CAM when a model is loaded)
- severity availability (false until a segmentation model is connected)
- limitations list

History is recorded in SQLite. Images are not stored.

## POST /api/explain

`multipart/form-data` with field `image` and optional `target_class`.

Generates Grad-CAM when a compatible Keras model is loaded. Returns
`available: false` with a clear message otherwise.

## GET /api/history

Returns analysis history entries (id, timestamp, filename, crop, disease,
confidence, model, mode).

## DELETE /api/history

Clears all history records.

---

## Error responses

All errors return JSON:

```json
{ "detail": "human-readable message" }
```

Stack traces are never returned to clients.
