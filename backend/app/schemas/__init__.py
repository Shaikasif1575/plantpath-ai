"""Pydantic response/request schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str
    crop: str
    disease: str
    probability: float = Field(ge=0.0, le=1.0)
    rank: int


class QualityWarning(BaseModel):
    code: str
    message: str
    severity: str = "info"  # info | warning


class ImageValidation(BaseModel):
    valid: bool
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    mode: Optional[str] = None
    file_size_bytes: Optional[int] = None
    warnings: List[QualityWarning] = Field(default_factory=list)
    error: Optional[str] = None


class GradCamResult(BaseModel):
    available: bool
    message: Optional[str] = None
    original_image: Optional[str] = None
    heatmap: Optional[str] = None
    overlay: Optional[str] = None
    target_class: Optional[str] = None
    target_layer: Optional[str] = None


class SeverityResult(BaseModel):
    available: bool
    message: Optional[str] = None
    reason: Optional[str] = None
    affected_area_percent: Optional[float] = None
    severity_level: Optional[str] = None
    method: Optional[str] = None
    original_image: Optional[str] = None
    mask_image: Optional[str] = None
    note: Optional[str] = None


class PredictionResponse(BaseModel):
    status: str
    demo_mode: bool
    leaf_validation_available: bool
    validation: ImageValidation
    crop: str
    disease: str
    condition: str  # healthy | diseased
    detected_disease: Optional[str]
    top_prediction: PredictionItem
    top_predictions: List[PredictionItem]
    confidence_band: str  # low | medium | high
    confidence_note: str
    model: Dict[str, Any]
    explainability: GradCamResult
    severity: SeverityResult
    limitations: List[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    target_class: Optional[str] = None


class DiseaseRecord(BaseModel):
    id: str
    crop: str
    disease: str
    scientific_name: str
    condition: str
    description: str
    symptoms: List[str] = Field(default_factory=list)
    visual_signs: List[str] = Field(default_factory=list)
    prevention: List[str] = Field(default_factory=list)
    management: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class CropInfo(BaseModel):
    name: str
    class_count: int
    healthy_available: bool
    diseases: List[str] = Field(default_factory=list)


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    filename: str
    crop: str
    disease: str
    confidence: float
    model: str
    mode: str
    condition: Optional[str] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    demo_mode: bool
    model_name: str
    model_version: str
    gradcam_available: bool
    severity_available: bool
    leaf_validation_available: bool
    class_count: Optional[int] = None
    timestamp: str
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    name: str = "PlantPath AI"
    version: str = "1.0.0"
    loaded: bool = False
    demo_mode: bool = False
    model_format: Optional[str] = None
    architecture: Optional[str] = None
    framework: Optional[str] = None
    input_size: Optional[List[int]] = None
    input_channels: Optional[int] = None
    output_classes: Optional[int] = None
    num_classes: Optional[int] = None
    parameters: Optional[int] = None
    model_file: Optional[str] = None
    classes: List[str] = Field(default_factory=list)
    class_names: List[str] = Field(default_factory=list)
    preprocessing: Optional[str] = None
    preprocessing_key: Optional[str] = None
    class_mapping_provenance: Optional[str] = None
    gradcam_available: bool = False
    severity: Dict[str, Any] = Field(default_factory=dict)
    segmentation_available: bool = False
    segmentation_status: str = "unavailable"
    low_confidence_threshold: float = 0.55
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    comparison: List[Dict[str, Any]] = Field(default_factory=list)
