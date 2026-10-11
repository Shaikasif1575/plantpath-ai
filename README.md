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
  confidence. The service returns `available: false` until a real segmentation
  model is connected.
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
      → Explainable AI (Grad-CAM) → Severity (when model exists)
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
| **Severity** | Architecture ready for U-Net integration; returns `available: false` until a real model exists |
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
      → Disease Classifier → Grad-CAM → Severity (future) → Results
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
this is **not** a controlled field evaluation.

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
| `SEGMENTATION_MODEL_PATH` | `models/segmentation_model` | Severity segmentation model; unset ⇒ severity honestly unavailable |
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
analyze(image)      # mask → affected pixel ratio
get_method_info()   # Diseased Pixel Area / Total Leaf Area × 100
```

**Current status: not connected** — the UI shows "Severity assessment is
currently unavailable for this model." Severity is *never* computed from
classification confidence. To enable: place a segmentation model at
`models/segmentation_model` (path configurable via `SEGMENTATION_MODEL_PATH`).

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
low-confidence branch, history CRUD, and demo-mode honesty.
**31 passed, 1 skipped** (the skip yields to the live model).

**E2E smoke test (API + rendered pages):**

```bash
python tests/e2e_smoke.py
```

Requires backend (:8000) and frontend (:3000) running. **71/71.**

**Browser journey test (Playwright + Chrome):**

```bash
python tests/browser_test.py
```

Walks the real user flow end-to-end — upload → analyze → Grad-CAM → report
download → history → model dashboard → dark mode → backend-down error state →
mobile viewport → accessibility → console cleanliness. **93/93**, screenshots
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
│   │   ├── ml/              # torch_classifier, gradcam_service, severity_service
│   │   ├── database/        # SQLite abstraction
│   │   ├── schemas/         # Pydantic models
│   │   └── utils/           # Safe filenames
│   ├── tests/test_api.py    # 32 pytest tests (live-model contract)
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── model_classes.json   # 38-class mapping served by the API (verified)
│   ├── class_names.json     # Legacy mapping (fallback)
│   ├── diseases.json        # Disease knowledge base w/ sources
│   └── model_metrics.json   # Sourced metrics (nulls where no run produced them)
├── models/
│   ├── efficientnet_v2_s_best.pth  # Connected checkpoint (~82 MB)
│   ├── classes.json         # Checkpoint's shipped class list
│   ├── infer.py             # Author's inference recipe (mirrored in loader)
│   └── classifier_audit.json # Verification-gate output
├── scripts/train_model.py   # Legacy training script (writes real metrics)
├── tests/
│   ├── e2e_smoke.py         # End-to-end smoke test (API + pages)
│   ├── browser_test.py      # Playwright real-user journey
│   ├── verify_new_model.py  # Model verification gate
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
3. **No severity without segmentation** — classification confidence does not
   indicate disease severity.
4. **Grad-CAM is not causation** — heatmaps are interpretability aids only.
5. **Controlled-dataset validation only** — the reported 0.9989 accuracy is
   PlantVillage validation performance carried in the checkpoint's own
   metadata; an informal 7-image field probe scored 4/7 (recorded in
   `models/classifier_audit.json`). Not field-deployment validated.
6. **Leaf validation / OOD are extension points** — reported as
   `leaf_validation_available: false`, never faked. The 0.55 low-confidence
   threshold is a guardrail, not an out-of-distribution detector.

---

## Future Work

1. U-Net disease segmentation → severity % and masks
2. Plant / non-plant classifier
3. Out-of-distribution detector
4. Additional datasets (field-collected)
5. EfficientNet / ResNet comparison models
6. ONNX / TFLite deployment
7. Mobile application
8. Field-image robustness evaluation protocol
9. Batch image analysis
10. Confusion matrix and class-wise metrics UI (data already supported)

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
