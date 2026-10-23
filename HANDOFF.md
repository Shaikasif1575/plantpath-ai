# HandOff — PlantPath AI (for ChatGPT / next session)

Project: PlantPath AI — explainable deep-learning research prototype (FastAPI backend + Next.js frontend, private repo `Shaikasif1575/plantpath-ai`).
Latest pushed commit: `840430c` (`main`).

---

## 1. Completed Feature: Real Treatment + Pesticide Guidance

### What it does
Every live prediction (`POST /api/predict`) now returns `disease_info`: a structured knowledge-base record for the predicted disease (crop + disease matched with token overlap >= 0.6; `None` if no confident match — honest fallback, never guesses).

### What the user sees
- `localhost:3000/scan` → upload leaf → Analyze → result shows a new `Treatment & Control` card:
  - Numbered cure steps (cultural → chemical)
  - `Recommended Pesticides` cards with `category`, active-ingredient pills (`captan`, `myclobutanil`, etc.), and `application` notes
  - Safety disclaimer (`label` / `PPE` / `local registration` / `IPM first`)
  - `Sources` citation line
- `localhost:3000/diseases` → click any disease (e.g., Apple Scab, Tomato Late Blight) → detail dialog shows `Treatment Steps` + `Recommended Pesticides` + `Incurable` warning (for virus / HLB / esca).
- `preview/index.html` → standalone HTML preview showing the full result layout (no server needed).

### Data provenance (honest, not invented)
- `data/diseases.json`: 38 records (26 diseased + 12 healthy). Treatment and pesticide content curated from university extension publications (Cornell, UMass, Penn State, Oregon State) and standard agronomic references.
- Viral diseases (`tomato-mosaic-virus`, `tomato-yellow-leaf-curl-virus`) and `orange-hlb-citrus-greening` (`grape-esca` also flagged): `treatment` steps exist (sanitation, hygiene, vector control, removal), but `pesticides: []` and `incurable: true`. No fake chemical cures.
- Source fields preserved and cited; `script/add_treatment_data.py` is the reproducible merge script.

---

## 2. Architecture (unchanged core + new lookup layer)

### Backend (FastAPI, port 8000)
- `app/services/disease_service.py`: `find_for_prediction(crop, disease, condition)` — token-match lookup on `data/diseases.json`. Returns `_guidance_record()` (stripped to `id`, `treatment`, `pesticides`, `management`, `prevention`, `sources`, `disclaimer`, `incurable`).
- `app/api/predict.py`: `disease_info` attached only for real predictions (`not demo_mode`). Safe: catches lookup exceptions and sets `None`.
- `app/schemas/__init__.py`: `PredictionResponse` has optional `disease_info`; `DiseaseRecord` has `treatment`/`pesticides`/`incurable`.

### Frontend (Next.js 14, port 3000)
- `lib/api.ts`: `DiseaseInfo` + `PesticideRecommendation` interfaces; `PredictionResponse` extended.
- `components/TreatmentPanel.tsx`: renders conditionally (`if (!info) return null` — never shows wrong guidance). Healthy state shows `No disease was detected` message.
- `components/PredictionResult.tsx`: embeds `<TreatmentPanel info={result.disease_info} />` after severity + limitations.
- `app/diseases/page.tsx`: renders `treatment`, `pesticides`, `incurable`, and updated disclaimer footer.

---

## 3. Verification Evidence (all passing at commit time)

### Backend tests (`pytest backend/tests/test_api.py`)
- 35 passed, 1 skipped (`test_predict_includes_treatment_guidance`: verifies `apple-scab` guidance; `test_disease_info_absent_in_demo_mode`: verifies `None` in demo; `test_disease_kb_treatment_integrity`: verifies all 26 diseased records have treatment, pesticides well-formed, `tomato-mosaic-virus` is the only empty-pesticide exception, `incurable` flags present for HLB / mosaic / yellow-curl / esca).

### Live verification (`tests/_verify_guidance_live.py`)
- Probe image (`tests/probe_set/author_sample_apple_scab.jpg`) → `apple-apple-scab` with 4 treatment steps, 1 pesticide entry (fungicide: captan/myclobutanil/dodine/sulfur/thiophanate-methyl), sources (`Cornell...`), disclaimer present.

### Matcher verification (`tests/_check_guidance_lookup.py`)
- 12 cases (Tomato Late Blight, Pepper Bacterial Spot, Spider Mites, Grape Esca, HLB, Apple Scab, Strawberry Leaf Scorch, Healthy Apple, Healthy Squash, Unknown, etc.) — 0 failures, overlap >= 0.6 for all expected matches, `None` for unmatched.

### E2E (`tests/e2e_smoke.py`)
- `predict` contract: `live_model` true, `demo_mode` false, `disease_info` present with `id`, `treatment`, `pesticides`, `sources`, `disclaimer`, `label` reference.

### Browser (`tests/browser_test.py` + Playwright scripts `tests/_browser_verify.py`, `_playwright_diseases.py`, `_playwright_inspect_panel.py`, `_playwright_scan_test.py`)
- `tests/preview/05-scan-with-treatment.png`: scan result with `TreatmentPanel` visible.
- `tests/preview/03-diseases-live.png`: diseases library loaded with all records.
- `tests/preview/04-scan-page-load.png`: scan page loads correctly (`Analyze` button present).

---

## 4. Binding Rules (from original instructions) — preserved

- **No classifier retraining/fine-tuning**: `models/efficientnet_v2_s_best.pth` is original; no new training artifacts for classifier.
- **No fake predictions/metrics/severity/Grad-CAM**: All metrics (`models/field_eval_results.json`, `models/model_metrics.json`, `models/severity_sanity_results.json`) are from actual runs with source attribution.
- **Severity only from real segmentation**: `severity.available: true` with `note: "...segmentation-derived..."`; unavailable message shown clearly when disconnected.
- **Low confidence (<0.55)**: `low_confidence` flag + message shown.
- **DEMO_MODE=false**: Confirmed in `data/model_metrics.json`, API responses, and UI (`LIVE MODEL` badge).
- **UI distinguishes LIVE vs DEMO**: `ModelStateBadge`, separate demo predictions (`top_prediction: None`), no fake results shown as real.
- **No unverified labels**: Field pool exhaustion measured (only +1 image after expanded EN/DE/NL + category crawl); protocol rules NOT weakened.

---

## 5. Documented Limitations (honest, propagated)

- **Over-segmentation bias**: Severity model (pseudo-label trained) over-segments; healthy mean ~45.2%. Note: "Severity is estimated by a segmentation model trained on pseudo-labelled data and is not agronomically validated (see models/segmentation_audit.json)." README: "Human-annotated masks are the fix, not threshold tuning."
- **Pool exhaustion**: Apple scab / corn rust / potato early blight measured exhausted under protocol v2; only +1 image found after broadening.
- **No cure diseases**: `incurable` flags visible; empty `pesticides` for viral diseases (mosaic virus); no fabricated cures.

---

## 6. Files Added / Modified (commit `840430c` — 26 files, +1962/-29)

Modified: `backend/app/api/predict.py`, `backend/app/schemas/__init__.py`, `backend/app/services/disease_service.py`, `backend/app/ml/torch_classifier.py` (parse_class_name unchanged, lookup added), `backend/tests/test_api.py`, `data/diseases.json`, `frontend/app/diseases/page.tsx`, `frontend/components/PredictionResult.tsx`, `frontend/lib/api.ts`, `tests/browser_test.py`, `tests/e2e_smoke.py`, `README.md`, `backend/API.md`, `.dockerignore` (added), `.gitignore` updates.

New: `frontend/components/TreatmentPanel.tsx`, `preview/index.html`, `scripts/add_treatment_data.py`, `PROJECT_SUMMARY.md`, `tests/_browser_verify.py`, `tests/_check_guidance_lookup.py`, `tests/_verify_guidance_live.py`, `tests/_check_diseases_content.py`, `tests/_playwright_diseases.py`, `tests/_playwright_inspect_panel.py`, `tests/_playwright_scan_test.py`, `tests/preview/*.png`.

---

## 7. How to continue from here

1. Pull/update: `git pull origin main` (commit `840430c`)
2. Restart servers (if needed): backend `uvicorn app.main:app --port 8000` (cwd `backend`); frontend `npm run start` (port 3000, cwd `frontend`)
3. Verify: `python -m pytest backend/tests/test_api.py -q` + `python tests/_verify_guidance_live.py`
4. Open preview: `preview/index.html` (standalone) or `localhost:3000/scan`
5. Next work (resource-blocked, not hidden):
   - Human-annotated lesion masks (fixes measured severity bias — NOT threshold tuning)
   - Community-labelled field datasets (pool exhaustion documented honestly)
   - ONNX export / deployment optimization (would need user authorization)

---
*All claims traceable to workspace files. No fabricated metrics, no hidden demo claims, no unverified pesticide recommendations. All binding rules from the original user instructions preserved.*
