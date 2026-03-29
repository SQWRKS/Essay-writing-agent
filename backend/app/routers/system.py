from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models import ApiLog
from app.schemas import HealthResponse, ConfigRead, ConfigUpdate, ApiLogRead
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["system"])

QUALITY_MODE_TO_ANTHROPIC_MODEL = {
    "quality": "claude-opus-4-6",
    "balanced": "claude-sonnet-4-6",
}

QUALITY_MODE_PROFILES = {
    "quality": {
        "LLM_MAX_TOKENS": 4096,
        "MAX_REVISION_ATTEMPTS": 3,
        "SECTION_SCORE_TARGET": 0.9,
        "COHERENCE_SCORE_TARGET": 0.9,
        "MIN_REVISION_DELTA": 0.015,
        "MAX_SECTION_REVISION_MINUTES": 90,
        "MAX_COHERENCE_REVISION_ROUNDS": 3,
    },
    "balanced": {
        "LLM_MAX_TOKENS": 3072,
        "MAX_REVISION_ATTEMPTS": 2,
        "SECTION_SCORE_TARGET": 0.88,
        "COHERENCE_SCORE_TARGET": 0.88,
        "MIN_REVISION_DELTA": 0.02,
        "MAX_SECTION_REVISION_MINUTES": 60,
        "MAX_COHERENCE_REVISION_ROUNDS": 2,
    },
}


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
        QUALITY_MODE=settings.QUALITY_MODE,
        LLM_PROVIDER=settings.LLM_PROVIDER,
        LLM_MODEL=settings.LLM_MODEL,
        ANTHROPIC_MODEL=settings.ANTHROPIC_MODEL,
        LLM_TEMPERATURE=settings.LLM_TEMPERATURE,
        LLM_MAX_TOKENS=settings.LLM_MAX_TOKENS,
        RESEARCH_SOURCES=settings.RESEARCH_SOURCES,
        WEB_SEARCH_ENABLED=settings.WEB_SEARCH_ENABLED,
        LOG_LEVEL=settings.LOG_LEVEL,
        BACKEND_PORT=settings.BACKEND_PORT,
        FRONTEND_PORT=settings.FRONTEND_PORT,
        CORS_ORIGINS=settings.CORS_ORIGINS,
        MAX_REVISION_ATTEMPTS=settings.MAX_REVISION_ATTEMPTS,
        SECTION_SCORE_TARGET=settings.SECTION_SCORE_TARGET,
        COHERENCE_SCORE_TARGET=settings.COHERENCE_SCORE_TARGET,
        MIN_REVISION_DELTA=settings.MIN_REVISION_DELTA,
        MAX_SECTION_REVISION_MINUTES=settings.MAX_SECTION_REVISION_MINUTES,
        MAX_COHERENCE_REVISION_ROUNDS=settings.MAX_COHERENCE_REVISION_ROUNDS,
        REVIEW_MIN_SCORE=settings.REVIEW_MIN_SCORE,
        GROUNDING_MIN_SCORE=settings.GROUNDING_MIN_SCORE,
        COHERENCE_MIN_SCORE=settings.COHERENCE_MIN_SCORE,
    )


@router.post("/config", response_model=ConfigRead)
async def update_config(payload: ConfigUpdate):
    if payload.QUALITY_MODE is not None:
        requested_mode = payload.QUALITY_MODE.strip().lower()
        if requested_mode not in QUALITY_MODE_TO_ANTHROPIC_MODEL:
            requested_mode = "quality"
        settings.QUALITY_MODE = requested_mode

        # Apply mode profile defaults first; explicit payload fields may override below.
        profile = QUALITY_MODE_PROFILES.get(requested_mode, {})
        settings.LLM_MAX_TOKENS = int(profile.get("LLM_MAX_TOKENS", settings.LLM_MAX_TOKENS))
        settings.MAX_REVISION_ATTEMPTS = int(profile.get("MAX_REVISION_ATTEMPTS", settings.MAX_REVISION_ATTEMPTS))
        settings.SECTION_SCORE_TARGET = float(profile.get("SECTION_SCORE_TARGET", settings.SECTION_SCORE_TARGET))
        settings.COHERENCE_SCORE_TARGET = float(profile.get("COHERENCE_SCORE_TARGET", settings.COHERENCE_SCORE_TARGET))
        settings.MIN_REVISION_DELTA = float(profile.get("MIN_REVISION_DELTA", settings.MIN_REVISION_DELTA))
        settings.MAX_SECTION_REVISION_MINUTES = int(profile.get("MAX_SECTION_REVISION_MINUTES", settings.MAX_SECTION_REVISION_MINUTES))
        settings.MAX_COHERENCE_REVISION_ROUNDS = int(profile.get("MAX_COHERENCE_REVISION_ROUNDS", settings.MAX_COHERENCE_REVISION_ROUNDS))

        if settings.LLM_PROVIDER.lower() == "anthropic":
            settings.ANTHROPIC_MODEL = QUALITY_MODE_TO_ANTHROPIC_MODEL[requested_mode]

    if payload.LLM_PROVIDER is not None:
        settings.LLM_PROVIDER = payload.LLM_PROVIDER
        if settings.LLM_PROVIDER.lower() == "anthropic":
            settings.ANTHROPIC_MODEL = QUALITY_MODE_TO_ANTHROPIC_MODEL.get(
                settings.QUALITY_MODE,
                settings.ANTHROPIC_MODEL,
            )
    if payload.LLM_MODEL is not None:
        settings.LLM_MODEL = payload.LLM_MODEL
    if payload.ANTHROPIC_MODEL is not None:
        settings.ANTHROPIC_MODEL = payload.ANTHROPIC_MODEL
        if payload.ANTHROPIC_MODEL == QUALITY_MODE_TO_ANTHROPIC_MODEL["quality"]:
            settings.QUALITY_MODE = "quality"
        elif payload.ANTHROPIC_MODEL == QUALITY_MODE_TO_ANTHROPIC_MODEL["balanced"]:
            settings.QUALITY_MODE = "balanced"
    if payload.LLM_TEMPERATURE is not None:
        settings.LLM_TEMPERATURE = payload.LLM_TEMPERATURE
    if payload.LLM_MAX_TOKENS is not None:
        settings.LLM_MAX_TOKENS = payload.LLM_MAX_TOKENS
    if payload.RESEARCH_SOURCES is not None:
        settings.RESEARCH_SOURCES = payload.RESEARCH_SOURCES
    if payload.WEB_SEARCH_ENABLED is not None:
        settings.WEB_SEARCH_ENABLED = payload.WEB_SEARCH_ENABLED
    if payload.MAX_REVISION_ATTEMPTS is not None:
        settings.MAX_REVISION_ATTEMPTS = payload.MAX_REVISION_ATTEMPTS
    if payload.SECTION_SCORE_TARGET is not None:
        settings.SECTION_SCORE_TARGET = payload.SECTION_SCORE_TARGET
    if payload.COHERENCE_SCORE_TARGET is not None:
        settings.COHERENCE_SCORE_TARGET = payload.COHERENCE_SCORE_TARGET
    if payload.MIN_REVISION_DELTA is not None:
        settings.MIN_REVISION_DELTA = payload.MIN_REVISION_DELTA
    if payload.MAX_SECTION_REVISION_MINUTES is not None:
        settings.MAX_SECTION_REVISION_MINUTES = payload.MAX_SECTION_REVISION_MINUTES
    if payload.MAX_COHERENCE_REVISION_ROUNDS is not None:
        settings.MAX_COHERENCE_REVISION_ROUNDS = payload.MAX_COHERENCE_REVISION_ROUNDS
    if payload.REVIEW_MIN_SCORE is not None:
        settings.REVIEW_MIN_SCORE = payload.REVIEW_MIN_SCORE
    if payload.GROUNDING_MIN_SCORE is not None:
        settings.GROUNDING_MIN_SCORE = payload.GROUNDING_MIN_SCORE
    if payload.COHERENCE_MIN_SCORE is not None:
        settings.COHERENCE_MIN_SCORE = payload.COHERENCE_MIN_SCORE
    return await get_config()
