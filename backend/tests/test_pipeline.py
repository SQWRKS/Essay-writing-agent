"""
Pipeline-level integration test.

Asserts two end-to-end invariants in a single run:
  1. Thesis metadata is generated and stored in project content.
  2. The strict reviewer rewrite loop fires for at least one section.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import ApiLog, Project, Task  # noqa: F401 – needed for table creation
from app.orchestration.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# In-memory test DB (self-contained; mirrors the setup used in test_agents.py)
# ---------------------------------------------------------------------------

_PIPELINE_DB_URL = "sqlite+aiosqlite:///:memory:"
_pipeline_engine = create_async_engine(_PIPELINE_DB_URL, echo=False)
_PipelineSession = async_sessionmaker(_pipeline_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _setup_pipeline_db():
    async with _pipeline_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _pipeline_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db():
    async with _PipelineSession() as session:
        yield session


@pytest_asyncio.fixture
async def test_project(db):
    now = datetime.now(timezone.utc)
    project = Project(
        id=str(uuid.uuid4()),
        title="Pipeline Test Project",
        topic="machine learning",
        status="pending",
        created_at=now,
        updated_at=now,
    )
    db.add(project)
    await db.commit()
    return project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REJECT_RESULT = {
    "score": 0.35,
    "feedback": "Test-forced rejection: content lacks evidence and citations.",
    "suggestions": ["Add inline citations.", "Include quantitative data."],
    "approved": False,
    "metrics": {
        "repeated_phrase_ratio": 0.0,
        "citation_count": 0,
        "generic_phrase_count": 2,
        "quantitative_signal_count": 0,
        "domain_keyword_hits": 0,
    },
}


def _make_selective_reviewer(real_execute):
    """
    Return a patched reviewer execute method that rejects the *first* review
    call for every section and defers subsequent calls to the real implementation.
    This guarantees the rewrite loop is triggered without depending on LLM access.
    """
    call_counts: dict[str, int] = {}

    async def selective_execute(self, input_data, project_id, db):
        section = input_data.get("section", "unknown")
        call_counts[section] = call_counts.get(section, 0) + 1
        if call_counts[section] == 1:
            return dict(_REJECT_RESULT)
        return await real_execute(self, input_data, project_id, db)

    return selective_execute


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_thesis_metadata_and_reviewer_rewrite(db, test_project):
    """
    End-to-end pipeline test on 'machine learning'.

    Invariant 1 – Thesis metadata:
        content["metadata"]["thesis"] must be present and non-trivial after the
        pipeline runs, confirming the ThesisAgent step executed and persisted its
        result.

    Invariant 2 – Reviewer rewrite loop:
        At least one section must have generated more than one writer task,
        confirming that a reviewer rejection triggered the revision path inside
        WorkerPool.execute_project_pipeline.
    """
    from app.agents.reviewer import ReviewerAgent

    real_execute = ReviewerAgent.execute
    patched_execute = _make_selective_reviewer(real_execute)

    pool = WorkerPool()
    with patch.object(ReviewerAgent, "execute", patched_execute):
        await pool.execute_project_pipeline(test_project.id, "machine learning", db)

    # Reload project from DB
    result = await db.execute(select(Project).where(Project.id == test_project.id))
    project = result.scalar_one_or_none()
    assert project is not None, "Project must exist after pipeline run"
    assert project.status == "completed", f"Pipeline ended with status: {project.status}"
    assert project.content, "Project content must be non-empty after pipeline run"

    content = json.loads(project.content)

    # -- Invariant 1: thesis metadata -------------------------------------------
    assert "metadata" in content, "content must contain a 'metadata' key"
    assert "thesis" in content["metadata"], (
        "Pipeline must produce and store a thesis statement in content['metadata']['thesis']"
    )
    thesis_text = content["metadata"]["thesis"]
    assert isinstance(thesis_text, str) and len(thesis_text.split()) >= 8, (
        f"Thesis must be a meaningful sentence (>=8 words); got: {thesis_text!r}"
    )

    # -- Invariant 2: reviewer rewrite loop -------------------------------------
    tasks_result = await db.execute(
        select(Task).where(
            and_(Task.project_id == test_project.id, Task.agent_name == "writer")
        )
    )
    writer_tasks = tasks_result.scalars().all()
    section_count = len(content.get("sections", {}))

    assert section_count > 0, "Pipeline must produce at least one section"
    assert len(writer_tasks) > section_count, (
        f"Reviewer rewrite loop must fire: expected more than {section_count} writer tasks, "
        f"got {len(writer_tasks)}"
    )
