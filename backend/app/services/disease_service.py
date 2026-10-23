"""Disease knowledge-base service."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("plantpath.services.diseases")

TREATMENT_DISCLAIMER = (
    "Treatment and pesticide guidance is educational reference data distilled "
    "from agricultural extension publications — not a prescription. Always "
    "follow the product label (the legal document), including dosage, "
    "pre-harvest interval and PPE requirements; confirm the product is "
    "registered for your crop and disease in your region; and consult your "
    "local agricultural extension or plant-protection authority before "
    "spraying. Use integrated pest management: resistant varieties, "
    "sanitation, crop rotation and monitoring come before any chemical "
    "application."
)


class DiseaseService:
    def __init__(self, diseases_path: Path) -> None:
        self._path = diseases_path
        self._records: List[Dict[str, Any]] = []
        self._meta: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._records = list(data.get("diseases", []))
                    self._meta = {k: v for k, v in data.items() if k != "diseases"}
                elif isinstance(data, list):
                    self._records = data
                    self._meta = {}
                else:
                    self._records = []
            else:
                self._records = []
                logger.warning("Diseases file not found: %s", self._path)
        except Exception as exc:
            logger.exception("Failed to load diseases")
            self._records = []
            self._meta = {"error": str(exc)}

    def list_diseases(
        self,
        crop: Optional[str] = None,
        condition: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records = self._records
        if crop:
            crop_l = crop.lower()
            records = [r for r in records if str(r.get("crop", "")).lower() == crop_l]
        if condition:
            cond_l = condition.lower()
            records = [r for r in records if str(r.get("condition", "")).lower() == cond_l]
        if search:
            q = search.lower().strip()
            def matches(r: Dict[str, Any]) -> bool:
                fields = [
                    r.get("crop", ""),
                    r.get("disease", ""),
                    r.get("scientific_name", ""),
                    r.get("description", ""),
                    " ".join(r.get("symptoms", [])),
                    " ".join(r.get("visual_signs", [])),
                ]
                return any(q in str(f).lower() for f in fields)
            records = [r for r in records if matches(r)]
        return records

    def get_disease(self, disease_id: str) -> Optional[Dict[str, Any]]:
        for r in self._records:
            if r.get("id") == disease_id:
                return r
        return None

    def find_for_prediction(
        self, crop: str, disease: str, condition: str
    ) -> Optional[Dict[str, Any]]:
        """Map a classifier output (crop/disease strings) to a knowledge record.

        Conservative by design: candidates must share the crop and condition,
        and a diseased prediction must overlap the record's disease name by
        token similarity. When nothing matches well enough, return None —
        showing no guidance is honest; showing the wrong disease's guidance
        is not.
        """
        crop_l = (crop or "").strip().lower()
        cond_l = (condition or "").strip().lower()
        candidates = [
            r
            for r in self._records
            if str(r.get("crop", "")).strip().lower() == crop_l
            and str(r.get("condition", "")).strip().lower() == cond_l
        ]
        if not candidates:
            return None
        if cond_l == "healthy":
            return self._guidance_record(candidates[0])

        pred_tokens = self._tokens(disease)
        if not pred_tokens:
            return None
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for r in candidates:
            rec_tokens = self._tokens(str(r.get("disease", "")))
            if not rec_tokens:
                continue
            overlap = len(pred_tokens & rec_tokens) / min(len(pred_tokens), len(rec_tokens))
            if best is None or overlap > best[0]:
                best = (overlap, r)
        if best is not None and best[0] >= 0.6:
            return self._guidance_record(best[1])
        return None

    @staticmethod
    def _tokens(text: str) -> set:
        import re

        stop = {"the", "of", "a", "an", "and", "on", "in"}
        return {
            t for t in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if t not in stop
        }

    @staticmethod
    def _guidance_record(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": r.get("id"),
            "crop": r.get("crop"),
            "disease": r.get("disease"),
            "scientific_name": r.get("scientific_name"),
            "condition": r.get("condition"),
            "description": r.get("description"),
            "treatment": list(r.get("treatment", []) or []),
            "pesticides": list(r.get("pesticides", []) or []),
            "management": list(r.get("management", []) or []),
            "prevention": list(r.get("prevention", []) or []),
            "sources": list(r.get("sources", []) or []),
            "disclaimer": TREATMENT_DISCLAIMER,
        }
        if r.get("incurable"):
            out["incurable"] = True
        return out

    def get_crops(self, class_names: List[str]) -> List[Dict[str, Any]]:
        """Build crop explorer info from class names."""
        from collections import defaultdict

        crop_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"name": "", "class_count": 0, "healthy_available": False, "diseases": []}
        )
        for cn in class_names:
            if "___" in cn:
                crop_part, disease_part = cn.split("___", 1)
            else:
                crop_part, disease_part = cn, cn
            crop_name = crop_part.replace(",", "").replace("_", " ").strip().title()
            entry = crop_map[crop_name]
            entry["name"] = crop_name
            entry["class_count"] += 1
            disease_clean = disease_part.replace("_", " ").strip()
            if disease_clean.lower() == "healthy":
                entry["healthy_available"] = True
            else:
                entry["diseases"].append(disease_clean)
        result = sorted(crop_map.values(), key=lambda x: x["name"])
        return result

    def all_class_names(self) -> List[str]:
        """Every class in the reference library (not the model's class set)."""
        names = set()
        for r in self._records:
            crop = (r.get("crop") or "").strip()
            if crop:
                names.add(crop.replace(",", "").replace(" ", "_"))
        return sorted(names)

    def get_meta(self) -> Dict[str, Any]:
        return dict(self._meta)
