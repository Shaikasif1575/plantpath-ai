"""Application configuration loaded from environment variables."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Anchor every path at the project root regardless of the process cwd.
# config.py lives at <project>/backend/app/config.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_PROJECT_ROOT / ".env"), extra="ignore")

    # Paths
    model_path: str = "models/plantpath_model.keras"
    class_names_path: str = "data/class_names.json"
    model_classes_path: str = "data/model_classes.json"
    model_audit_path: str = "models/classifier_audit.json"
    preprocessing: str = "imagenet_normalized"
    segmentation_model_path: str = "models/segmentation"
    model_metrics_path: str = "data/model_metrics.json"
    diseases_path: str = "data/diseases.json"
    db_path: str = "backend/app/database/plantpath.db"

    # Runtime
    demo_mode: bool = False
    max_upload_size_mb: int = 10
    confidence_low: float = 0.55
    confidence_medium: float = 0.75
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Model
    model_name: str = "PlantPath AI"
    model_version: str = "1.0.0"

    @property
    def cors_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def resolved_model_path(self) -> Path:
        return _abs(self.model_path)

    @property
    def resolved_class_names_path(self) -> Path:
        return _abs(self.class_names_path)

    @property
    def resolved_model_classes_path(self) -> Path:
        return _abs(self.model_classes_path)

    @property
    def resolved_model_audit_path(self) -> Path:
        return _abs(self.model_audit_path)

    @property
    def resolved_segmentation_model_path(self) -> Path:
        return _abs(self.segmentation_model_path)

    @property
    def resolved_metrics_path(self) -> Path:
        return _abs(self.model_metrics_path)

    @property
    def resolved_diseases_path(self) -> Path:
        return _abs(self.diseases_path)

    @property
    def resolved_db_path(self) -> Path:
        return _abs(self.db_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_json(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return None
