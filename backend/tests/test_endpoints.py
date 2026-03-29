import pytest
import json
from sqlalchemy import select


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_config(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "QUALITY_MODE" in data
    assert "LLM_MODEL" in data
    assert "MAX_REVISION_ATTEMPTS" in data
    assert "SECTION_SCORE_TARGET" in data
    assert "MIN_REVISION_DELTA" in data
    assert "MAX_SECTION_REVISION_MINUTES" in data
    assert "REVIEW_MIN_SCORE" in data
    assert "GROUNDING_MIN_SCORE" in data
    assert "COHERENCE_MIN_SCORE" in data


@pytest.mark.asyncio
async def test_update_config(client):
    resp = await client.post(
        "/api/config",
        json={
            "QUALITY_MODE": "balanced",
            "LLM_PROVIDER": "anthropic",
            "LLM_TEMPERATURE": 0.5,
            "MAX_REVISION_ATTEMPTS": 4,
            "SECTION_SCORE_TARGET": 0.95,
            "MIN_REVISION_DELTA": 0.02,
            "MAX_SECTION_REVISION_MINUTES": 180,
            "REVIEW_MIN_SCORE": 0.8,
            "GROUNDING_MIN_SCORE": 0.72,
            "COHERENCE_MIN_SCORE": 0.76,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["QUALITY_MODE"] == "balanced"
    assert resp.json()["LLM_PROVIDER"] == "anthropic"
    assert resp.json()["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"
    assert resp.json()["LLM_TEMPERATURE"] == 0.5
    assert resp.json()["MAX_REVISION_ATTEMPTS"] == 4
    assert resp.json()["SECTION_SCORE_TARGET"] == 0.95
    assert resp.json()["MIN_REVISION_DELTA"] == 0.02
    assert resp.json()["MAX_SECTION_REVISION_MINUTES"] == 180
    assert resp.json()["REVIEW_MIN_SCORE"] == 0.8
    assert resp.json()["GROUNDING_MIN_SCORE"] == 0.72
    assert resp.json()["COHERENCE_MIN_SCORE"] == 0.76


@pytest.mark.asyncio
async def test_quality_mode_switches_anthropic_model(client):
    quality_resp = await client.post(
        "/api/config",
        json={"LLM_PROVIDER": "anthropic", "QUALITY_MODE": "quality"},
    )
    assert quality_resp.status_code == 200
    assert quality_resp.json()["QUALITY_MODE"] == "quality"
    assert quality_resp.json()["ANTHROPIC_MODEL"] == "claude-opus-4-6"

    balanced_resp = await client.post(
        "/api/config",
        json={"LLM_PROVIDER": "anthropic", "QUALITY_MODE": "balanced"},
    )
    assert balanced_resp.status_code == 200
    assert balanced_resp.json()["QUALITY_MODE"] == "balanced"
    assert balanced_resp.json()["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_quality_mode_balanced_applies_cost_profile_defaults(client):
    resp = await client.post(
        "/api/config",
        json={"LLM_PROVIDER": "anthropic", "QUALITY_MODE": "balanced"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["QUALITY_MODE"] == "balanced"
    assert data["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"
    assert data["MAX_REVISION_ATTEMPTS"] == 2
    assert data["SECTION_SCORE_TARGET"] == 0.88
    assert data["COHERENCE_SCORE_TARGET"] == 0.88
    assert data["MIN_REVISION_DELTA"] == 0.02
    assert data["MAX_SECTION_REVISION_MINUTES"] == 60
    assert data["MAX_COHERENCE_REVISION_ROUNDS"] == 2
    assert data["LLM_MAX_TOKENS"] == 3072


@pytest.mark.asyncio
async def test_quality_mode_profile_allows_explicit_field_overrides(client):
    resp = await client.post(
        "/api/config",
        json={
            "LLM_PROVIDER": "anthropic",
            "QUALITY_MODE": "balanced",
            "MAX_REVISION_ATTEMPTS": 5,
            "SECTION_SCORE_TARGET": 0.91,
            "LLM_MAX_TOKENS": 2500,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["QUALITY_MODE"] == "balanced"
    assert data["MAX_REVISION_ATTEMPTS"] == 5
    assert data["SECTION_SCORE_TARGET"] == 0.91
    assert data["LLM_MAX_TOKENS"] == 2500


@pytest.mark.asyncio
async def test_create_project(client):
    resp = await client.post("/projects", json={"title": "Test Project", "topic": "machine learning"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Test Project"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_list_projects(client):
    await client.post("/projects", json={"title": "List Test", "topic": "AI"})
    resp = await client.get("/projects")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_project(client):
    create_resp = await client.post("/projects", json={"title": "Detail Test", "topic": "NLP"})
    project_id = create_resp.json()["id"]
    resp = await client.get(f"/projects/{project_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == project_id
    assert "agent_states" in data
    assert "tasks" in data


@pytest.mark.asyncio
async def test_get_project_not_found(client):
    resp = await client.get("/projects/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_planner_agent(client):
    create_resp = await client.post("/projects", json={"title": "Planner Test", "topic": "deep learning"})
    project_id = create_resp.json()["id"]
    resp = await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "planner", "input_data": {"topic": "deep learning"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "output" in data


@pytest.mark.asyncio
async def test_run_unknown_agent(client):
    create_resp = await client.post("/projects", json={"title": "Bad Agent", "topic": "test"})
    project_id = create_resp.json()["id"]
    resp = await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "nonexistent_agent", "input_data": {}},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_project_tasks(client):
    create_resp = await client.post("/projects", json={"title": "Tasks Test", "topic": "robotics"})
    project_id = create_resp.json()["id"]
    await client.post(
        f"/projects/{project_id}/run-agent",
        json={"agent_name": "planner", "input_data": {"topic": "robotics"}},
    )
    resp = await client.get(f"/projects/{project_id}/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) > 0


@pytest.mark.asyncio
async def test_pause_project(client):
    create_resp = await client.post("/projects", json={"title": "Pause Test", "topic": "ethics"})
    project_id = create_resp.json()["id"]

    pause_resp = await client.post(f"/projects/{project_id}/pause")
    assert pause_resp.status_code == 202

    project_resp = await client.get(f"/projects/{project_id}")
    assert project_resp.status_code == 200
    assert project_resp.json()["status"] == "paused"


@pytest.mark.asyncio
async def test_delete_project(client):
    create_resp = await client.post("/projects", json={"title": "Delete Test", "topic": "history"})
    project_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/projects/{project_id}")
    assert delete_resp.status_code == 200

    get_resp = await client.get(f"/projects/{project_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_running_project_conflict(client):
    create_resp = await client.post("/projects", json={"title": "Running Delete", "topic": "systems"})
    project_id = create_resp.json()["id"]

    run_resp = await client.post(f"/projects/{project_id}/run")
    assert run_resp.status_code == 202

    delete_resp = await client.delete(f"/projects/{project_id}")
    assert delete_resp.status_code == 409


@pytest.mark.asyncio
async def test_pipeline_omits_sections_marked_excluded(db_session, monkeypatch):
    from app import agents as agents_module
    from app.models import Project
    from app.orchestration.worker_pool import WorkerPool

    class FakePlanner:
        async def execute(self, input_data, project_id, db):
            topic = input_data.get("topic", "test topic")
            return {
                "sections": [
                    {
                        "key": "introduction",
                        "title": "Introduction",
                        "description": "Intro",
                        "research_queries": [f"{topic} overview"],
                        "word_count_target": 180,
                        "thesis_goal": "State the thesis.",
                        "must_cover": ["Context"],
                        "evidence_requirements": ["Use one citation"],
                        "writing_directive": "Be concise.",
                        "include": True,
                        "subheading_hints": ["Context"],
                    },
                    {
                        "key": "results",
                        "title": "Results",
                        "description": "Results",
                        "research_queries": [f"{topic} findings"],
                        "word_count_target": 220,
                        "thesis_goal": "Present findings.",
                        "must_cover": ["Main finding"],
                        "evidence_requirements": ["Empirical evidence"],
                        "writing_directive": "Be specific.",
                        "include": False,
                        "subheading_hints": ["Primary Findings"],
                    },
                ],
                "research_queries": [f"{topic} overview", f"{topic} findings"],
                "estimated_total_words": 400,
            }

    class FakeResearch:
        async def execute(self, input_data, project_id, db):
            return {
                "sources": [
                    {
                        "title": "Synthetic Source",
                        "source": "web",
                        "year": 2024,
                        "abstract": "Evidence-backed synthetic abstract.",
                        "relevance_score": 0.9,
                        "combined_quality_score": 0.9,
                        "verification_score": 0.9,
                    }
                ],
                "total_found": 1,
                "summary": "Synthetic research summary.",
                "source_breakdown": {"web": 1},
            }

    class FakeVerification:
        async def execute(self, input_data, project_id, db):
            sources = input_data.get("sources", [])
            return {
                "verified_sources": sources,
                "verification_summary": {"verified": len(sources)},
            }

    class FakeWriter:
        async def execute(self, input_data, project_id, db):
            section = input_data.get("section", "introduction")
            return {
                "section": section,
                "content": f"## Context\nThis is generated content for {section} with citation [1].",
                "word_count": 20,
                "subheadings": [{"title": "Context", "content": "Generated subsection."}],
            }

    class FakeGrounding:
        async def execute(self, input_data, project_id, db):
            return {
                "score": 0.9,
                "approved": True,
                "issues": [],
                "unsupported_claim_count": 0,
            }

    class FakeReviewer:
        async def execute(self, input_data, project_id, db):
            return {
                "score": 0.92,
                "approved": True,
                "feedback": "Good",
                "suggestions": [],
                "strengths": ["Grounded"],
                "blocking_issues": [],
                "category_scores": {"coverage": 0.9},
                "citation_count": 1,
            }

    class FakeCoherence:
        async def execute(self, input_data, project_id, db):
            sections = input_data.get("sections", {})
            return {
                "score": 0.91,
                "approved": True,
                "feedback": "Coherent",
                "issues": [],
                "suggestions": [],
                "flagged_sections": [],
                "repeated_opening_sections": [],
                "section_summaries": {k: {"opening": "ok"} for k in sections.keys()},
                "topic_coverage": 0.7,
            }

    class FakeCitation:
        async def execute(self, input_data, project_id, db):
            return {
                "formatted_citations": ["[1] Synthetic Source (2024)"],
                "bibliography": "[1] Synthetic Source (2024).",
            }

    class FakeFigure:
        async def execute(self, input_data, project_id, db):
            return {"figures": []}

    fake_registry = {
        **agents_module.AGENT_REGISTRY,
        "planner": FakePlanner,
        "research": FakeResearch,
        "verification": FakeVerification,
        "writer": FakeWriter,
        "grounding": FakeGrounding,
        "reviewer": FakeReviewer,
        "coherence": FakeCoherence,
        "citation": FakeCitation,
        "figure": FakeFigure,
    }
    monkeypatch.setattr(agents_module, "AGENT_REGISTRY", fake_registry)

    project = Project(title="Adaptive Test", topic="conceptual ethics", status="pending")
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)

    pool = WorkerPool()
    await pool.execute_project_pipeline(project.id, project.topic, db_session)

    result = await db_session.execute(select(Project).where(Project.id == project.id))
    stored_project = result.scalar_one()
    assert stored_project.status == "completed"

    content_blob = stored_project.content
    parsed_content = json.loads(content_blob) if isinstance(content_blob, str) else (content_blob or {})
    section_keys = set((parsed_content.get("sections") or {}).keys())
    assert "introduction" in section_keys
    assert "results" not in section_keys
