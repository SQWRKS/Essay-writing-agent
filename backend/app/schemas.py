from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ProjectCreate(BaseModel):
    title: str
    topic: str


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    topic: Optional[str] = None
    status: Optional[str] = None
    content: Optional[str] = None


class ProjectRead(BaseModel):
    id: str
    title: str
    topic: str
    status: str
    created_at: datetime
    updated_at: datetime
    content: Optional[str] = None

    model_config = {"from_attributes": True}


class TaskCreate(BaseModel):
    agent_name: str
    input_data: Optional[str] = None
    dependencies: Optional[str] = None


class TaskRead(BaseModel):
    id: str
    project_id: str
    agent_name: str
    status: str
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: Optional[str] = None

    model_config = {"from_attributes": True}


class AgentStateRead(BaseModel):
    id: int
    project_id: str
    agent_name: str
    status: str
    last_output: Optional[str] = None
    updated_at: datetime

    model_config = {"from_attributes": True}


class ApiLogRead(BaseModel):
    id: int
    endpoint: str
    method: str
    agent_name: Optional[str] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    timestamp: datetime
    metadata_json: Optional[str] = None

    model_config = {"from_attributes": True}


class ConfigRead(BaseModel):
    QUALITY_MODE: str
    LLM_PROVIDER: str
    LLM_MODEL: str
    ANTHROPIC_MODEL: str
    LLM_TEMPERATURE: float
    LLM_MAX_TOKENS: int
    RESEARCH_SOURCES: List[str]
    WEB_SEARCH_ENABLED: bool
    LOG_LEVEL: str
    BACKEND_PORT: int
    FRONTEND_PORT: int
    CORS_ORIGINS: List[str]
    MAX_REVISION_ATTEMPTS: int
    SECTION_SCORE_TARGET: float
    COHERENCE_SCORE_TARGET: float
    MIN_REVISION_DELTA: float
    MAX_SECTION_REVISION_MINUTES: int
    MAX_COHERENCE_REVISION_ROUNDS: int
    REVIEW_MIN_SCORE: float
    GROUNDING_MIN_SCORE: float
    COHERENCE_MIN_SCORE: float


class ConfigUpdate(BaseModel):
    QUALITY_MODE: Optional[str] = None
    LLM_PROVIDER: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    ANTHROPIC_MODEL: Optional[str] = None
    LLM_TEMPERATURE: Optional[float] = None
    LLM_MAX_TOKENS: Optional[int] = None
    RESEARCH_SOURCES: Optional[List[str]] = None
    WEB_SEARCH_ENABLED: Optional[bool] = None
    MAX_REVISION_ATTEMPTS: Optional[int] = None
    SECTION_SCORE_TARGET: Optional[float] = None
    COHERENCE_SCORE_TARGET: Optional[float] = None
    MIN_REVISION_DELTA: Optional[float] = None
    MAX_SECTION_REVISION_MINUTES: Optional[int] = None
    MAX_COHERENCE_REVISION_ROUNDS: Optional[int] = None
    REVIEW_MIN_SCORE: Optional[float] = None
    GROUNDING_MIN_SCORE: Optional[float] = None
    COHERENCE_MIN_SCORE: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class RunAgentRequest(BaseModel):
    agent_name: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
