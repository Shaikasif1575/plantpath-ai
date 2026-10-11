"""Analysis history endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.main import AppState

router = APIRouter()


@router.get("/history")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = AppState.database
    if db is None:
        return {"items": [], "total": 0}
    items = db.list_predictions(limit=limit, offset=offset)
    total = db.count_predictions()
    return {"items": items, "total": total}


@router.delete("/history")
async def clear_history():
    db = AppState.database
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    deleted = db.clear_history()
    return {"status": "ok", "deleted": deleted}
