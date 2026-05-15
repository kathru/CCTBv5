"""Positions endpoints — read-only."""
from fastapi import APIRouter, Request
from src.persistence.repositories.positions import PositionRepository

router = APIRouter(prefix="/api/positions", tags=["positions"])


@router.get("/open")
async def get_open_positions(request: Request) -> list[dict]:
    repo = PositionRepository(request.app.state.db)
    rows = await repo.get_open()
    return [_serialize(r) for r in rows]


@router.get("/closed")
async def get_closed_positions(request: Request, limit: int = 50) -> list[dict]:
    repo = PositionRepository(request.app.state.db)
    rows = await repo.get_closed(limit=limit)
    return [_serialize(r) for r in rows]


def _serialize(row: dict) -> dict:
    return {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
            for k, v in row.items()}
