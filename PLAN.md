# Implementation Plan (post-treatment feature, commit 840430c)

Audit status (verified):
- Backend: live model (EfficientNetV2-S), severity (U-Net pair), Grad-CAM, disease_service lookup, real metrics (no fabrication), demo_mode=false.
- Frontend: scan/diseases/model/research/history/report; TreatmentPanel integrated; demo clearly separated.
- Data: diseases.json (26 diseased + 12 healthy, with treatment/pesticides); field_eval results (28 images, exact 0.107); severity sanity (78.6% diseased, 45.2% healthy); pool exhaustion documented honestly.
- Tests: pytest 35 pass; e2e 74/74; browser 96/96 (after build fix); live verification passes.

Binding rules preserved: no classifier retraining, no fake predictions/metrics/severity/grad-cam, severity only from real segmentation, demo clearly labelled, metrics only with actual attribution, protocol rules not weakened.

---

## Incremental priorities (each will have tests + verification)

1. **Image validation + user-friendly errors** (priority)
2. **Low-confidence + unsupported image handling** (priority)
3. **History deletion + privacy controls**
4. **PDF report (real prediction + cited guidance)**
5. **Model info page improvements**
6. **Explainability verification/test** (only if produces genuine results)
7. **Responsive/accessibility/loading improvements**

---

## Affected files (expected)
- `backend/app/utils/validation.py` / `image_validation`
- `backend/app/services/validation_service.py` (if exists)
- `frontend/components/ImageUpload.tsx` or `ScanPage` (error display)
- `frontend/app/scan/page.tsx` / components
- `tests/test_validation.py` (new or updated)
- `tests/e2e_smoke.py` (updated for new validation contracts)
- `tests/browser_test.py` (updated for new error states / loading)
- `HANDOFF.md` / `README.md` updated as needed
