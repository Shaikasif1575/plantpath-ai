# PlantPath AI

**Explainable Deep Learning for Plant Disease Detection, Classification and Severity Assessment**

A B.Tech Computer Science research prototype: an AI-powered plant leaf analysis
platform with local ML inference, Grad-CAM explainability, disease knowledge
base, analysis history, and research documentation — built entirely with free
and open-source technologies.

---

## Table of Contents

- [Academic Disclaimer](#academic-disclaimer)
- [Project Overview](#project-overview)
- [Features](#features)
- [Research Motivation](#research-motivation)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Model](#ml-model)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [API Documentation](#api-documentation)
- [Grad-CAM](#grad-cam)
- [Severity Assessment](#severity-assessment)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Academic Disclaimer

> PlantPath AI is a research and educational prototype. Predictions are limited
> to the classes represented by the connected model and should not be treated as
> definitive agricultural diagnosis. Field performance may differ from
> controlled-dataset performance.

Additional honesty guarantees built into the product:

- **No invented metrics.** Accuracy / precision / recall / F1 are shown only
  when present in project evaluation files; otherwise "Not available".
- **No fabricated severity.** Severity is never derived from classification
  confidence. It is computed only from the connected segmentation model's
  masks (project-trained U-Nets; lesion labels are pseudo-labels — see
  `models/segmentation_audit.json`); without a loaded model the service
  returns `available: false`.
- **No paid / cloud AI APIs.** Inference runs locally (PyTorch).
- **Demo mode is visually obvious.** When no model file is connected,
  `DEMO_MODE=true` and the UI displays a DEMO MODE banner on every result.
- **Confidence ≠ certainty.** Confidence bands are labelled as relative model
  confidence with an explicit disclaimer.

---

## Project Overview

PlantPath AI lets users upload or capture a leaf image, validate its quality,
run local disease classification, view top predictions with confidence context,
inspect Grad-CAM attention maps (when available), and read verified disease
information — all through a professional research-prototype web interface.

**Main user flow:**

```
User → Open PlantPath AI → Scan/Upload Leaf → Image Validation
      → Local ML Inference → Disease Classification → Confidence + Top Predictions
      → Explainable AI (Grad-CAM) → Severity (segmentation masks)
      → Disease Information → Report → Analysis History
```

---

## Features

| Area | Features |
|---|---|
| **Scan Leaf** | Upload (JPG/JPEG/PNG/WEBP), drag & drop, browser camera capture (MediaDevices API), preview, retake, analyze |
| **Image Validation** | Format, size, dimension, corruption checks; quality warnings (dark, blurry, very small) — warnings never hard-reject |
| **Prediction** | Top-1 + top-3 probabilities, confidence band (low/medium/high, configurable thresholds), crop/disease parsing |
| **Explainable AI** | Real Grad-CAM for the connected PyTorch model; honest "unavailable" messaging otherwise |
| **Severity** | Connected U-Net pair (lesion + leaf) measures affected leaf area % from real masks; honest `available: false` if no segmentation model is loaded |
| **Disease Library** | 38 records, search, crop filter, healthy/diseased filter, detail panels with sources |
| **Model Dashboard** | Architecture, input size, parameters, sourced metrics (attributed or "Not available"), training history (only if present), verified class mapping with provenance, multi-model comparison slots |
| **Research** | Problem statement, gap, objectives, methodology, dataset, evaluation, XAI, lab-to-field gap, future work |
| **History** | SQLite-backed metadata records; clear history; no image storage; no filesystem path exposure |
| **Report** | Local HTML report download (no paid reporting API) with full disclaimer |
| **UI/UX** | Light/dark mode (persisted), responsive, hamburger menu, semantic HTML, ARIA labels, visible focus |
| **Security** | Extension + MIME validation, size limits, sanitized filenames, CORS, no stack traces to clients |

---

## Research Motivation

Plant diseases cause significant yield losses worldwide. Deep learning can
classify leaf images from photographs, but most student prototypes either
overclaim accuracy or hide limitations. PlantPath AI is built to demonstrate
the full ML product pipeline **while communicating scientific limits honestly**
— controlled datasets (PlantVillage) are not field performance, confidence is
not certainty, and severity requires segmentation, not classification.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Next.js 14 / TypeScript / Tailwind)              │
│  Home · Scan · Diseases · Model · Research · History · About│
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (REST, JSON / multipart)
┌──────────────────────────▼──────────────────────────────────┐
│  Backend (FastAPI)                                          │
│  api/ → services/ → ml/ → database/                         │
│  ┌────────────┐ ┌──────────────┐ ┌───────────────┐          │
│  │ predict    │ │ gradcam      │ │ severity      │          │
│  │ validation │ │ (real CAM)   │ │ (ext. point)  │          │
│  └────────────┘ └──────────────┘ └───────────────┘          │
│  SQLite: prediction_history, model_metadata                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ in-process
┌──────────────────────────▼──────────────────────────────────┐
│  ML Service (PyTorch, loaded ONCE at startup)               │
│  models/efficientnet_v2_s_best.pth + data/model_classes.json│
│  Verification gate: refuse to load unless the audit passes  │
└─────────────────────────────────────────────────────────────┘
```

**System pipeline (future-ready):**

```
Image → Leaf Validation (future) → OOD/Confidence Check (future)
      → Disease Classifier → Grad-CAM → Severity (segmentation) → Results
```

---

## Dataset

**PlantVillage** — a controlled benchmark of healthy and diseased leaf images
(Mohanty, Hughes & Salathé, 2016, *Frontiers in Plant Science*).

- **38 classes**, **14 crop species** (standard mapping in `data/model_classes.json`)
- Controlled backgrounds and lighting — **not equivalent to field imagery**

---

## ML Model

| Property | Value |
|---|---|
| Connected file | `models/efficientnet_v2_s_best.pth` (PyTorch checkpoint, ~82 MB) |
| Architecture | EfficientNetV2-S (torchvision) + Linear(38) head — 20,226,166 parameters |
| Source | Hugging Face `BiernyVR/crop-disease-classifier` (MIT license) |
| Input size | 224×224×3 |
| Preprocessing | RGB, bilinear resize, /255, ImageNet mean/std (author's own recipe) |
| Class mapping | `data/model_classes.json` — 38 PlantVillage classes, cross-checked against the checkpoint's embedded class list at load time |
| Reported metric | Validation accuracy 0.9988950… from the checkpoint's own `val_acc` key / model card — **attributed, controlled-dataset only** |
| Load policy | Once at application startup, behind a verification gate |

**Verification gate.** `tests/verify_new_model.py` probes the checkpoint before
the app will trust it: five distinct classes with non-degenerate probabilities
plus an in-domain sanity sample (the model author's own apple-scab image must
classify as `Apple___Apple_scab`). The gate writes `models/classifier_audit.json`;
the backend refuses to load any checkpoint whose audit is missing, failing, or
inconsistent with the loaded file. Its honest field probe (7 informal images:
4/7 correct) is recorded in the same file and shown on the Model dashboard —
this first probe is **not** a controlled field evaluation; see the structured
protocol below.

**Field evaluation (structured protocol, measured by this project).**
`tests/fetch_field_eval_set.py` collects in-the-wild photographs from Wikimedia
Commons and verifies each label mechanically from the uploader's file
description: crop keyword + disease keyword must appear in title/description,
leaf-tissue keywords are required (a leaf classifier is evaluated on leaves),
and electron-microscopy / non-leaf imagery is rejected. Provenance for every
image (description page, author, license, matched keywords) is recorded in
`data/field_eval/sources.json`. `tests/evaluate_field_set.py` then runs the
production `POST /api/predict` path over the set and writes per-image results
plus aggregates to `models/field_eval_results.json` and an attributed
`project_evaluation.field_evaluation` block in `data/model_metrics.json`.

| Metric (2026-10-13 run, 27 images / 7 classes) | Value |
|---|---|
| Exact class accuracy | **0.111** (3/27) |
| Crop-level accuracy | **0.259** (7/27) |
| Mean confidence | 0.708 |
| Low-confidence rate (< 0.55) | 0.185 |
| Best class | `Tomato___Late_blight` — recall 0.75, F1 0.55 |

Read this as an informal robustness probe, not a benchmark: small,
non-balanced convenience sample; labels are uploader descriptions; supports
per class are 1–6. The failure modes are consistent with the lab-to-field
gap — potato late blight is often predicted `Tomato___Late_blight` (same
pathogen, near-identical lesion appearance), and healthy tomato leaves are
often predicted as other healthy crops. The classifier's 0.9989 remains a
PlantVillage figure; field performance is what this protocol measures, and it
is far lower.

**Historical note:** the original TensorFlow/Keras demo pipeline
(`scripts/train_model.py`, `.keras` files) remains in the repository as the
project's training reference, and a MobileNetV2 loader is retained as a legacy
fallback, but the connected production model is the PyTorch checkpoint above.
No training, retraining, or fine-tuning was performed to produce it.

---

## Installation

### Prerequisites

- **Python 3.10–3.13** (the connected model runs on PyTorch CPU; TensorFlow is
  only needed for the legacy training script)
- **Node.js 18+**
- (Optional) Docker & Docker Compose

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
# For the connected PyTorch classifier:
pip install torch torchvision
```

### Frontend

```bash
cd frontend
npm install
```

---

## Configuration

Copy `.env.example` to `.env` (backend) and `frontend/.env.example` to
`frontend/.env.local` as needed.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/plantpath_model.keras` | Connected checkpoint (`.pth` or `.keras`); production uses `models/efficientnet_v2_s_best.pth` |
| `MODEL_CLASSES_PATH` | `data/model_classes.json` | Verified class-name mapping served by the API |
| `MODEL_AUDIT_PATH` | `models/classifier_audit.json` | Verification-gate output; load is refused without a passing audit |
| `CLASS_NAMES_PATH` | `data/class_names.json` | Legacy class mapping (fallback path) |
| `MODEL_METRICS_PATH` | `data/model_metrics.json` | Evaluation/training metrics (only sourced values are shown) |
| `DISEASES_PATH` | `data/diseases.json` | Disease knowledge base |
| `DEMO_MODE` | `false` | `true` only for labelled demonstrations; production is `false` |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size limit |
| `CORS_ORIGINS` | `http://localhost:3000,…` | Comma-separated allowed origins |
| `CONFIDENCE_LOW` | `0.55` | Low-confidence threshold (below → honest low-confidence message) |
| `CONFIDENCE_MEDIUM` | `0.75` | Medium/high band threshold |
| `SEGMENTATION_MODEL_PATH` | `models/segmentation` | Directory with `meta.json` + lesion/leaf U-Net checkpoints; unset ⇒ severity honestly unavailable |
| `NEXT_PUBLIC_API_URL` (frontend) | `http://localhost:8000` | Backend base URL |

---

## Running Locally

**Terminal 1 — backend:**

```bash
cd backend
.venv\Scripts\activate        # Windows
uvicorn app.main:app --reload --port 8000
```

`DEMO_MODE=false` is the production default (see `.env`); the backend refuses
to start predictions without a verified model.

**Terminal 2 — frontend:**

```bash
cd frontend
npm run build && npm run start   # production; or: npm run dev
```

Open **http://localhost:3000**

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

The navbar shows a green **LIVE MODEL** badge when the verified checkpoint is
connected; a DEMO MODE badge appears only when `DEMO_MODE=true`.

---

## Docker

```bash
docker compose up --build
```

- Frontend → http://localhost:3000
- Backend → http://localhost:8000 (Python 3.11 + PyTorch CPU included)

The backend image ships the connected checkpoint and runs with
`DEMO_MODE=false`. The container's prediction service still enforces the
verification gate at startup — if `models/classifier_audit.json` is missing or
inconsistent with the mounted checkpoint, predictions are refused (nothing is
simulated in its place). To mount a different checkpoint, place files in
`./models` and override `MODEL_PATH` / `MODEL_CLASSES_PATH` in the
environment.

---

## API Documentation

Interactive Swagger UI: **http://localhost:8000/docs**
(Aalso see `backend/API.md`)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model loaded, demo mode, XAI/severity availability |
| GET | `/api/model-info` | Architecture, metrics (only if present), class names, comparison slots |
| GET | `/api/crops` | Crop explorer generated from class mapping |
| GET | `/api/diseases` | Disease library (`crop`, `condition`, `search` filters) |
| GET | `/api/diseases/{id}` | Single disease record |
| POST | `/api/predict` | Multipart image → validation + prediction + XAI + severity status |
| POST | `/api/explain` | Multipart image (+ optional `target_class`) → Grad-CAM |
| GET | `/api/history` | Analysis history (metadata only) |
| DELETE | `/api/history` | Clear history |

---

## Grad-CAM

`POST /api/explain` computes gradient-weighted class activation maps on the
connected PyTorch model (target layer resolved from the architecture — the
last convolutional block, `features.7.0` for EfficientNetV2-S). Returns
original / heatmap / overlay images (base64 PNG).

> Highlighted regions indicate image areas that contributed strongly to the
> model prediction. This visualization is an interpretability aid and does not
> establish biological causation.

If no compatible model is loaded: `available: false` with a clear message —
**never fake heatmaps.**

---

## Severity Assessment

`SeverityService` implements:

```python
is_available()      # segmentation model connected?
analyze(image)      # lesion mask + leaf mask → affected pixel ratio
get_method_info()   # Diseased region area / Leaf region area × 100
```

**Current status: connected.** Two small U-Nets trained by this project on
CPU (PyTorch, `scripts/train_segmentation.py`) produce the masks:

| Model | Target | Training data | Label source | Val Dice / IoU |
|---|---|---|---|---|
| `models/segmentation/lesion_unet.pth` | lesion region | godoldol99/SAM2-based-plant-disease-lesion-segmentation-model-DATASET (Apache-2.0, 408 pairs) | **SAM2 model pseudo-labels** — not human annotation | 0.6845 / 0.5246 (best epoch 13 of 18) |
| `models/segmentation/leaf_unet.pth` | leaf vs background | LeafNet75/Leaf-Disease-Segmentation-masks (Apache-2.0, 588 pairs) | dataset-provided binary masks | 0.6868 / 0.5284 (best epoch 16 of 18) |

Both models are 1,942,577-parameter U-Nets trained on CPU (~24–27 min each).
Full per-epoch history and split sizes are in `models/segmentation/meta.json`;
dataset provenance is in `models/segmentation/dataset_sources.json`.

Severity = `lesion ∩ leaf` pixels / `leaf` pixels × 100, with thresholds
mild < 10%, moderate 10–40%, severe ≥ 40%. Validation metrics (Dice/IoU
against the same label source) and every provenance detail live in
`models/segmentation/meta.json` and `models/segmentation_audit.json`.

**Honesty contract:**
- Severity is *never* computed from classification confidence.
- The lesion labels are pseudo-labels, so validation Dice measures fit to the
  SAM2 labeler, not agreement with human annotators; severity values are
  unvalidated research estimates (stated in the API `note`, the UI, and the
  downloadable report).
- If the segmentation model fails to load or finds no leaf tissue, the API
  returns `available: false` with a reason instead of a number.
- The disease classifier was **not** trained, retrained or fine-tuned to
  produce severity — classifier weights are untouched.

Retraining (CPU, ~2 h):

```bash
python scripts/download_seg_data.py     # Apache-2.0 datasets (~400 + ~590 pairs)
python scripts/train_segmentation.py    # writes models/segmentation/{lesion,leaf}_unet.pth + meta.json
```

---

## Testing

**Backend (pytest):**

```bash
python -m pytest backend/tests/test_api.py -v
```

Covers health, model info (live contract: 38 classes, attributed metrics,
nulls where no evaluation ran), crops, diseases (list/filter/search/detail/404),
real prediction fields, in-domain probe prediction (author's sample must
classify as `Apple___Apple_scab`), Grad-CAM availability, forced
low-confidence branch, history CRUD, demo-mode honesty, a regression test
pinning API top-1 to the class-ordered probability vector, and connected
severity (coherent values, threshold-consistent level, honesty note).
**32 passed, 1 skipped** (33 total; the skip yields to the live model).

**E2E smoke test (API + rendered pages):**

```bash
python tests/e2e_smoke.py
```

Requires backend (:8000) and frontend (:3000) running. **74/74.**

**Browser journey test (Playwright + Chrome):**

```bash
python tests/browser_test.py
```

Walks the real user flow end-to-end — upload → analyze → Grad-CAM → report
download → history → model dashboard → dark mode → backend-down error state →
mobile viewport → accessibility → console cleanliness. **96/96**, screenshots
in `tests/browser_report/`.

**Model verification gate:**

```bash
python tests/verify_new_model.py
```

Probes the connected checkpoint and (re)writes `models/classifier_audit.json`.

**Frontend:** TypeScript-validated production build (`npm run build`).

---

## Project Structure

```
plantpath-ai/
├── frontend/
│   ├── app/                 # Home, scan, diseases, model, research, history, about
│   ├── components/          # Navbar, Footer, Capture, PredictionResult, GradCamViewer…
│   ├── lib/api.ts           # Centralized API client + types
│   ├── hooks/useTheme.ts    # Dark/light mode
│   ├── styles/globals.css
│   ├── Dockerfile
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + lifespan (model loads once)
│   │   ├── config.py        # Settings from env
│   │   ├── api/             # health, model, diseases, predict, explain, history
│   │   ├── services/        # prediction, disease, image validation
│   │   ├── ml/              # torch_classifier, gradcam_service, severity_service, segmentation, unet
│   │   ├── database/        # SQLite abstraction
│   │   ├── schemas/         # Pydantic models
│   │   └── utils/           # Safe filenames
│   ├── tests/test_api.py    # 33 pytest tests (live-model contract)
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── model_classes.json   # 38-class mapping served by the API (verified)
│   ├── class_names.json     # Legacy mapping (fallback)
│   ├── diseases.json        # Disease knowledge base w/ sources
│   ├── model_metrics.json   # Sourced metrics (nulls where no run produced them)
│   └── field_eval/          # Verified-label field set: images + sources.json (protocol v2)
├── models/
│   ├── efficientnet_v2_s_best.pth  # Connected checkpoint (~82 MB)
│   ├── segmentation/        # Trained severity U-Nets + meta.json (metrics/provenance)
│   ├── segmentation_audit.json # Segmentation training provenance + held-out metrics
│   ├── field_eval_results.json # Per-image field-eval predictions + aggregates
│   ├── classes.json         # Checkpoint's shipped class list
│   ├── infer.py             # Author's inference recipe (mirrored in loader)
│   └── classifier_audit.json # Verification-gate output
├── scripts/
│   ├── train_model.py       # Legacy training script (writes real metrics)
│   ├── download_seg_data.py # Downloads Apache-2.0 severity datasets
│   └── train_segmentation.py # Trains severity U-Nets (CPU, honest metrics)
├── tests/
│   ├── e2e_smoke.py         # End-to-end smoke test (API + pages)
│   ├── browser_test.py      # Playwright real-user journey
│   ├── verify_new_model.py  # Model verification gate
│   ├── fetch_field_eval_set.py # Wikimedia Commons field set (verified labels)
│   ├── evaluate_field_set.py # Field evaluation → attributed per-class metrics
│   ├── smoke_classifier.py  # In-process classifier smoke
│   └── probe_set/           # Verification images + sources.json
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Limitations

1. **Fixed class set** — only the trained classes can be recognised; unknown
   diseases are not detected without a future OOD module.
2. **Controlled dataset** — PlantVillage ≠ field imagery (lighting, occlusion,
   mixed infections, novel pathogens).
3. **Severity is model-derived and unvalidated** — classification confidence
   never indicates severity; the connected U-Net pair measures affected area
   from masks, but its lesion labels are SAM2 pseudo-labels, so values are
   research estimates, not agronomic diagnoses.
4. **Grad-CAM is not causation** — heatmaps are interpretability aids only.
5. **Controlled-dataset validation only** — the reported 0.9989 accuracy is
   PlantVillage validation performance carried in the checkpoint's own
   metadata; the structured field evaluation (27 verified-label Commons
   photographs, 7 classes) measured **0.111 exact / 0.259 crop accuracy**
   (`models/field_eval_results.json`), and an earlier informal 7-image probe
   scored 4/7 (`models/classifier_audit.json`). Not field-deployment
   validated.
6. **Leaf validation / OOD are extension points** — reported as
   `leaf_validation_available: false`, never faked. The 0.55 low-confidence
   threshold is a guardrail, not an out-of-distribution detector.

---

## Future Work

1. Human-annotated lesion masks → validate severity against real labels
2. Plant / non-plant classifier
3. Out-of-distribution detector
4. Larger, community-labelled field datasets beyond Wikimedia Commons
5. EfficientNet / ResNet comparison models
6. ONNX / TFLite deployment
7. Mobile application
8. Batch image analysis
9. Confusion matrix and class-wise metrics UI (data already supported)

---

## Technologies

| Layer | Stack |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Lucide React |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| ML | PyTorch (torchvision EfficientNetV2-S), NumPy, Pillow, OpenCV |
| Database | SQLite |
| Testing | pytest, HTTP smoke tests |
| Container | Docker, Docker Compose |

**No paid APIs. No proprietary AI. No cloud inference.**
