"""Disease knowledge-base endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.main import AppState

router = APIRouter()


@router.get("/diseases")
async def list_diseases(
    crop: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    ds = AppState.disease_service
    if ds is None:
        return {"items": [], "total": 0, "meta": {}}
    items = ds.list_diseases(crop=crop, condition=condition, search=search)
    return {"items": items, "total": len(items), "meta": ds.get_meta()}


@router.get("/diseases/{disease_id}")
async def get_disease(disease_id: str):
    ds = AppState.disease_service
    if ds is None:
        raise HTTPException(status_code=404, detail="Disease library not loaded.")
    record = ds.get_disease(disease_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Disease '{disease_id}' not found.")
    return record
