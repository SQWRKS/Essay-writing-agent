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


# ---------------------------------------------------------------------------
# Fine-tune settings tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_project_without_settings_is_backward_compatible(client):
    """Creating without settings must work identically to before."""
    resp = await client.post("/projects", json={"title": "No Settings", "topic": "quantum computing"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "No Settings"
    assert data["status"] == "pending"
    assert data.get("settings_json") is None


@pytest.mark.asyncio
async def test_create_project_with_partial_settings(client):
    """Only the supplied settings fields are stored; omitted fields stay null."""
    resp = await client.post(
        "/projects",
        json={
            "title": "Partial Settings",
            "topic": "climate change",
            "settings": {"word_count_target": 3000},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    settings_json = data.get("settings_json")
    assert settings_json is not None
    import json as _json
    stored = _json.loads(settings_json)
    assert stored["word_count_target"] == 3000
    assert "rubric" not in stored
    assert "writing_style" not in stored
    assert "context_text" not in stored


@pytest.mark.asyncio
async def test_create_project_with_all_settings(client):
    """All fine-tune settings are persisted correctly."""
    resp = await client.post(
        "/projects",
        json={
            "title": "All Settings",
            "topic": "renewable energy",
            "settings": {
                "word_count_target": 5000,
                "writing_style": "argumentative",
                "context_text": "Focus on solar and wind power.",
                "rubric": "30% analysis, 30% evidence, 40% structure",
            },
        },
    )
    assert resp.status_code == 201
    import json as _json
    stored = _json.loads(resp.json()["settings_json"])
    assert stored["word_count_target"] == 5000
    assert stored["writing_style"] == "argumentative"
    assert "solar" in stored["context_text"]
    assert "analysis" in stored["rubric"]


@pytest.mark.asyncio
async def test_word_count_target_validation(client):
    """word_count_target below minimum (100) is rejected by schema."""
    resp = await client.post(
        "/projects",
        json={"title": "Bad WC", "topic": "test", "settings": {"word_count_target": 50}},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_word_count_target_max_validation(client):
    """word_count_target above maximum (50000) is rejected by schema."""
    resp = await client.post(
        "/projects",
        json={"title": "Big WC", "topic": "test", "settings": {"word_count_target": 99999}},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_context_txt(client):
    """Upload a plain-text context file and verify it is merged into settings."""
    create_resp = await client.post(
        "/projects", json={"title": "File Upload Test", "topic": "machine vision"}
    )
    project_id = create_resp.json()["id"]

    txt_content = b"Machine vision relies on convolutional neural networks for feature extraction."
    from httpx import AsyncClient
    resp = await client.post(
        f"/projects/{project_id}/upload-context",
        files={"file": ("context.txt", txt_content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["characters_extracted"] == len(txt_content)

    # Verify context_text was stored in project settings_json
    import json as _json
    from sqlalchemy import select
    proj_resp = await client.get(f"/projects/{project_id}")
    project_data = proj_resp.json()
    settings = _json.loads(project_data["settings_json"])
    assert "convolutional" in settings["context_text"]


@pytest.mark.asyncio
async def test_upload_context_unsupported_type(client):
    """Uploading an unsupported file type returns 400."""
    create_resp = await client.post(
        "/projects", json={"title": "Bad File", "topic": "test topic"}
    )
    project_id = create_resp.json()["id"]
    resp = await client.post(
        f"/projects/{project_id}/upload-context",
        files={"file": ("notes.csv", b"a,b,c", "text/csv")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_context_merges_with_existing(client):
    """File upload appends to existing context_text rather than overwriting it."""
    create_resp = await client.post(
        "/projects",
        json={
            "title": "Merge Test",
            "topic": "ai safety",
            "settings": {"context_text": "Prior context."},
        },
    )
    project_id = create_resp.json()["id"]

    resp = await client.post(
        f"/projects/{project_id}/upload-context",
        files={"file": ("extra.txt", b"Additional file content.", "text/plain")},
    )
    assert resp.status_code == 200

    import json as _json
    proj_resp = await client.get(f"/projects/{project_id}")
    settings = _json.loads(proj_resp.json()["settings_json"])
    assert "Prior context." in settings["context_text"]
    assert "Additional file content." in settings["context_text"]


@pytest.mark.asyncio
async def test_planner_scales_word_count(client):
    """When word_count_target is set, section word counts are scaled proportionally."""
    from app.agents.planner import PlannerAgent
    agent = PlannerAgent()
    result = agent._template_plan("renewable energy", word_count_target=700)
    total = sum(s["word_count_target"] for s in result["sections"])
    assert abs(total - 700) <= 100  # allow rounding tolerance


@pytest.mark.asyncio
async def test_planner_no_scaling_without_word_count():
    """Without word_count_target, section word counts are the template defaults."""
    from app.agents.planner import PlannerAgent, SECTION_TEMPLATES
    agent = PlannerAgent()
    result = agent._template_plan("renewable energy")
    default_total = sum(t["word_count_target"] for t in SECTION_TEMPLATES)
    actual_total = sum(s["word_count_target"] for s in result["sections"])
    assert actual_total == default_total


@pytest.mark.asyncio
async def test_planner_writing_style_injected():
    """writing_style hint appears in section writing_directives."""
    from app.agents.planner import PlannerAgent
    agent = PlannerAgent()
    result = agent._template_plan("climate policy", writing_style="argumentative")
    for section in result["sections"]:
        assert "argumentative" in section.get("writing_directive", "").lower()


@pytest.mark.asyncio
async def test_reviewer_accepts_rubric():
    """Reviewer heuristic path accepts rubric without errors."""
    from app.agents.reviewer import ReviewerAgent
    agent = ReviewerAgent()
    result = agent._heuristic_review(
        section="introduction",
        content="This is a test. However, limitations exist. The evidence suggests improvements.",
        expected_word_count=20,
        rubric="30% critical analysis, 30% evidence, 40% structure",
    )
    assert "score" in result
    assert isinstance(result["score"], float)


@pytest.mark.asyncio
async def test_pipeline_receives_project_settings(db_session, monkeypatch):
    """Settings (word_count_target, writing_style, rubric) reach the relevant agents."""
    from app import agents as agents_module
    from app.models import Project
    from app.orchestration.worker_pool import WorkerPool

    received_inputs: dict = {}

    class CapturePlanner:
        async def execute(self, input_data, project_id, db):
            received_inputs["planner"] = dict(input_data)
            from app.agents.planner import PlannerAgent
            return PlannerAgent()._template_plan(
                input_data.get("topic", "test"),
                word_count_target=input_data.get("word_count_target"),
                writing_style=input_data.get("writing_style", ""),
            )

    class FakeResearch:
        async def execute(self, input_data, project_id, db):
            return {
                "sources": [{"title": "S1", "source": "web", "year": 2024,
                              "abstract": "Abstract.", "relevance_score": 0.9,
                              "combined_quality_score": 0.9, "verification_score": 0.9}],
                "total_found": 1, "summary": "Summary.", "source_breakdown": {"web": 1},
            }

    class FakeVerification:
        async def execute(self, input_data, project_id, db):
            return {"verified_sources": input_data.get("sources", []),
                    "verification_summary": {"verified": 1}}

    class CaptureWriter:
        async def execute(self, input_data, project_id, db):
            received_inputs.setdefault("writer", []).append(dict(input_data))
            section = input_data.get("section", "introduction")
            return {"section": section,
                    "content": f"Content for {section} with citation [1].",
                    "word_count": 20, "subheadings": []}

    class FakeGrounding:
        async def execute(self, input_data, project_id, db):
            return {"score": 0.9, "approved": True, "issues": [], "unsupported_claim_count": 0}

    class CaptureReviewer:
        async def execute(self, input_data, project_id, db):
            received_inputs.setdefault("reviewer", []).append(dict(input_data))
            return {"score": 0.95, "approved": True, "feedback": "Good",
                    "suggestions": [], "strengths": [], "blocking_issues": [],
                    "category_scores": {"coverage": 0.95}, "citation_count": 1}

    class FakeCoherence:
        async def execute(self, input_data, project_id, db):
            sections = input_data.get("sections", {})
            return {"score": 0.92, "approved": True, "feedback": "Coherent",
                    "issues": [], "suggestions": [], "flagged_sections": [],
                    "repeated_opening_sections": [],
                    "section_summaries": {k: {"opening": "ok"} for k in sections},
                    "topic_coverage": 0.7}

    class FakeCitation:
        async def execute(self, input_data, project_id, db):
            return {"formatted_citations": ["[1] S1 (2024)"], "bibliography": "[1] S1 (2024)."}

    class FakeFigure:
        async def execute(self, input_data, project_id, db):
            return {"figures": []}

    fake_registry = {
        **agents_module.AGENT_REGISTRY,
        "planner": CapturePlanner,
        "research": FakeResearch,
        "verification": FakeVerification,
        "writer": CaptureWriter,
        "grounding": FakeGrounding,
        "reviewer": CaptureReviewer,
        "coherence": FakeCoherence,
        "citation": FakeCitation,
        "figure": FakeFigure,
    }
    monkeypatch.setattr(agents_module, "AGENT_REGISTRY", fake_registry)

    project = Project(title="Settings Passthrough", topic="quantum computing", status="pending")
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)

    project_settings = {
        "word_count_target": 2800,
        "writing_style": "argumentative",
        "rubric": "50% argument quality, 50% evidence",
    }
    pool = WorkerPool()
    await pool.execute_project_pipeline(project.id, project.topic, db_session,
                                        project_settings=project_settings)

    # Planner received both settings fields
    assert received_inputs["planner"]["word_count_target"] == 2800
    assert received_inputs["planner"]["writing_style"] == "argumentative"

    # Writer received writing_style
    writer_calls = received_inputs.get("writer", [])
    assert writer_calls, "Writer was never called"
    assert all(c.get("writing_style") == "argumentative" for c in writer_calls)

    # Reviewer received rubric
    reviewer_calls = received_inputs.get("reviewer", [])
    assert reviewer_calls, "Reviewer was never called"
    assert all(c.get("rubric") == "50% argument quality, 50% evidence" for c in reviewer_calls)
