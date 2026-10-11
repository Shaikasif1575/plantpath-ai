"""Disease knowledge-base service."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plantpath.services.diseases")


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
