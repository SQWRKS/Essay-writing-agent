from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
from pathlib import Path
import json


CORE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CORE_DIR.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

ENV_FILES = (
    str(BACKEND_DIR / ".env"),
    str(PROJECT_ROOT / ".env"),
)


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./essay_agent.db"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # Select the active provider: "anthropic" (default) or "openai"
    LLM_PROVIDER: str = "anthropic"

    # OpenAI settings
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # Anthropic settings
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-opus-4-6"

    # Shared LLM settings
    QUALITY_MODE: str = "quality"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    RESEARCH_SOURCES: List[str] = ["arxiv", "semantic_scholar", "web"]
    LOG_LEVEL: str = "INFO"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 3000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    # Timeout (seconds) for a single LLM API request before it is cancelled
    LLM_REQUEST_TIMEOUT: int = 180

    # Maximum number of revision loops per section (writer → reviewer → writer)
    MAX_REVISION_ATTEMPTS: int = 3
    SECTION_SCORE_TARGET: float = 0.9
    COHERENCE_SCORE_TARGET: float = 0.9
    MIN_REVISION_DELTA: float = 0.015
    MAX_SECTION_REVISION_MINUTES: int = 90
    MAX_COHERENCE_REVISION_ROUNDS: int = 3
    REVIEW_MIN_SCORE: float = 0.75
    GROUNDING_MIN_SCORE: float = 0.7
    COHERENCE_MIN_SCORE: float = 0.72

    @field_validator("RESEARCH_SOURCES", "CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_list_setting(cls, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.startswith("["):
                try:
                    decoded = json.loads(trimmed)
                    if isinstance(decoded, list):
                        return [str(item).strip() for item in decoded if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in trimmed.split(",") if item.strip()]
        return value

    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore", enable_decoding=False)


settings = Settings()
