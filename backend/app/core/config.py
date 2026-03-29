from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./essay_agent.db"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # Select the active provider: "openai" (default) or "anthropic"
    LLM_PROVIDER: str = "openai"

    # OpenAI settings
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # Anthropic settings
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-latest"

    # Shared LLM settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    RESEARCH_SOURCES: List[str] = ["arxiv", "semantic_scholar", "web"]
    LOG_LEVEL: str = "INFO"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 3000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    # Maximum number of revision loops per section (writer → reviewer → writer)
    MAX_REVISION_ATTEMPTS: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
