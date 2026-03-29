from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models import ApiLog
from app.schemas import HealthResponse, ConfigRead, ConfigUpdate, ApiLogRead
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(select(func.count()).select_from(ApiLog))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", version="1.0.0", database=db_status)


@router.get("/logs", response_model=list[ApiLogRead])
async def get_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    result = await db.execute(
        select(ApiLog).order_by(ApiLog.timestamp.desc()).offset(offset).limit(page_size)
    )
    return result.scalars().all()


@router.get("/config", response_model=ConfigRead)
async def get_config():
    return ConfigRead(
        LLM_PROVIDER=settings.LLM_PROVIDER,
        LLM_MODEL=settings.LLM_MODEL,
        ANTHROPIC_MODEL=settings.ANTHROPIC_MODEL,
        LLM_TEMPERATURE=settings.LLM_TEMPERATURE,
        LLM_MAX_TOKENS=settings.LLM_MAX_TOKENS,
        RESEARCH_SOURCES=settings.RESEARCH_SOURCES,
        LOG_LEVEL=settings.LOG_LEVEL,
        BACKEND_PORT=settings.BACKEND_PORT,
        FRONTEND_PORT=settings.FRONTEND_PORT,
        CORS_ORIGINS=settings.CORS_ORIGINS,
    )


@router.post("/config", response_model=ConfigRead)
async def update_config(payload: ConfigUpdate):
    if payload.LLM_PROVIDER is not None:
        settings.LLM_PROVIDER = payload.LLM_PROVIDER
    if payload.LLM_MODEL is not None:
        settings.LLM_MODEL = payload.LLM_MODEL
    if payload.ANTHROPIC_MODEL is not None:
        settings.ANTHROPIC_MODEL = payload.ANTHROPIC_MODEL
    if payload.LLM_TEMPERATURE is not None:
        settings.LLM_TEMPERATURE = payload.LLM_TEMPERATURE
    if payload.LLM_MAX_TOKENS is not None:
        settings.LLM_MAX_TOKENS = payload.LLM_MAX_TOKENS
    if payload.RESEARCH_SOURCES is not None:
        settings.RESEARCH_SOURCES = payload.RESEARCH_SOURCES
    return await get_config()
