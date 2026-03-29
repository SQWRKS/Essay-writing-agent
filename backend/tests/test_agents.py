import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.database import Base
from app.models import Project, AgentState, ApiLog  # noqa: F401

AGENT_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
agent_engine = create_async_engine(AGENT_TEST_DB_URL, echo=False)
AgentTestSession = async_sessionmaker(agent_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_agent_db():
    async with agent_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with agent_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db():
    async with AgentTestSession() as session:
        yield session


@pytest_asyncio.fixture
async def test_project(db):
    import uuid
    from datetime import datetime, timezone
    project = Project(
        id=str(uuid.uuid4()),
        title="Agent Test Project",
        topic="machine learning",
        status="running",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(project)
    await db.commit()
    return project


@pytest.mark.asyncio
async def test_planner_agent(db, test_project):
    from app.agents.planner import PlannerAgent
    agent = PlannerAgent()
    result = await agent.execute({"topic": "machine learning"}, test_project.id, db)
    assert "sections" in result
    assert "research_queries" in result
    assert "estimated_total_words" in result
    assert len(result["sections"]) > 0
    assert "thesis_goal" in result["sections"][0]
    assert "must_cover" in result["sections"][0]
    assert "evidence_requirements" in result["sections"][0]


@pytest.mark.asyncio
async def test_research_agent(db, test_project):
    from app.agents.research import ResearchAgent
    agent = ResearchAgent()
    result = await agent.execute(
        {"queries": ["machine learning overview"], "sources": ["web"]},
        test_project.id, db
    )
    assert "sources" in result
    assert "total_found" in result
    assert isinstance(result["sources"], list)
    if result["sources"]:
        assert "relevance_score" in result["sources"][0]
        assert "ranking_features" in result["sources"][0]


def test_research_ranking_prefers_query_aligned_sources():
    from app.agents.research import ResearchAgent

    agent = ResearchAgent()
    ranked = agent._rank_sources(
        [
            {
                "title": "Clinical Retrieval-Augmented Generation for Oncology",
                "authors": ["Smith"],
                "year": 2024,
                "abstract": "Retrieval-augmented generation improved grounded oncology question answering with evidence-linked citations.",
                "doi": "10.1234/onc-rag",
                "source": "semantic_scholar",
            },
            {
                "title": "General Neural Network Survey",
                "authors": ["Jones"],
                "year": 2024,
                "abstract": "An overview of neural architectures and optimization strategies.",
                "doi": "10.1234/general-nn",
                "source": "web",
            },
        ],
        topic="clinical retrieval augmented generation",
        queries=["evidence grounded clinical question answering"],
    )

    assert ranked[0]["title"] == "Clinical Retrieval-Augmented Generation for Oncology"
    assert ranked[0]["relevance_score"] > ranked[1]["relevance_score"]
    assert ranked[0]["match_reasons"]


@pytest.mark.asyncio
async def test_verification_agent(db, test_project):
    from app.agents.verification import VerificationAgent
    sources = [
        {"title": "Test Paper", "authors": ["Smith, J."], "year": 2023,
         "abstract": "Test abstract.", "url": "https://example.com", "doi": "10.1234/test"},
        {"title": "", "authors": [], "year": None, "abstract": "", "url": "", "doi": "bad-doi"},
    ]
    agent = VerificationAgent()
    result = await agent.execute({"sources": sources}, test_project.id, db)
    assert "verified_sources" in result
    assert "rejected_sources" in result
    assert "verification_score" in result
    assert "verification_summary" in result
    assert result["verified_sources"][0]["credibility_label"] in {"medium", "high"}
    assert result["rejected_sources"][0]["credibility_label"] == "low"


@pytest.mark.asyncio
async def test_writer_agent(db, test_project, monkeypatch):
    from app.agents.writer import WriterAgent
    import app.agents.writer as writer_module

    async def fake_completion(*args, **kwargs):
        return "Machine learning systems improve diagnostic triage accuracy when evidence retrieval quality is high [1]."

    monkeypatch.setattr(writer_module, "is_llm_available", lambda: True)
    monkeypatch.setattr(writer_module, "timed_chat_completion", fake_completion)

    agent = WriterAgent()
    result = await agent.execute(
        {"section": "introduction", "topic": "machine learning", "word_count": 200, "research_data": {}},
        test_project.id, db
    )
    assert "section" in result
    assert "content" in result
    assert "word_count" in result
    assert result["section"] == "introduction"
    assert len(result["content"]) > 0


@pytest.mark.asyncio
async def test_writer_agent_fails_without_llm(db, test_project, monkeypatch):
    from app.agents.writer import WriterAgent
    import app.agents.writer as writer_module

    monkeypatch.setattr(writer_module, "is_llm_available", lambda: False)

    agent = WriterAgent()
    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        await agent.execute(
            {"section": "introduction", "topic": "machine learning", "word_count": 200, "research_data": {}},
            test_project.id,
            db,
        )


def test_grounding_agent_flags_unsupported_claims():
    from app.agents.grounding import GroundingAgent

    agent = GroundingAgent()
    result = agent._heuristic_grounding(
        content=(
            "This system improves every benchmark in medicine and law without any trade-offs. "
            "It therefore outperforms all prior approaches universally."
        ),
        evidence_pack=[
            {"title": "Clinical Retrieval Study", "abstract_excerpt": "Grounded retrieval improved answer faithfulness in a constrained clinical QA benchmark."},
        ],
        revision_attempt=0,
    )

    assert result["approved"] is False
    assert result["unsupported_claim_count"] >= 1
    assert result["issues"]


def test_grounding_agent_accepts_cited_evidence_aligned_claims():
    from app.agents.grounding import GroundingAgent

    agent = GroundingAgent()
    result = agent._heuristic_grounding(
        content=(
            "Clinical retrieval-augmented generation improved answer faithfulness in oncology question answering [1]. "
            "The evidence suggests gains were strongest when retrieval quality was high and domain terms were well represented in the corpus [1]."
        ),
        evidence_pack=[
            {
                "title": "Clinical Retrieval-Augmented Generation for Oncology",
                "abstract_excerpt": "Retrieval quality improved grounded oncology question answering and answer faithfulness.",
            },
        ],
        revision_attempt=0,
    )

    assert result["approved"] is True
    assert result["supported_claim_count"] >= 1
    assert result["score"] >= 0.7


def test_coherence_agent_flags_repetition_and_weak_sections():
    from app.agents.coherence import CoherenceAgent

    agent = CoherenceAgent()
    result = agent._heuristic_coherence(
        topic="clinical retrieval augmented generation",
        sections={
            "introduction": "This paper examines clinical retrieval augmented generation. This paper examines clinical retrieval augmented generation in detail.",
            "discussion": "The findings suggest retrieval quality matters for grounded oncology question answering.",
            "conclusion": "This paper examines clinical retrieval augmented generation. Future work should continue.",
        },
        quality_sections={
            "introduction": {"approved": True, "score": 0.82},
            "discussion": {"approved": False, "score": 0.61},
            "conclusion": {"approved": True, "score": 0.78},
        },
    )

    assert result["approved"] is False
    assert result["issues"]
    assert result["flagged_sections"] == ["discussion"]


def test_coherence_agent_accepts_aligned_sections():
    from app.agents.coherence import CoherenceAgent

    agent = CoherenceAgent()
    result = agent._heuristic_coherence(
        topic="clinical retrieval augmented generation",
        sections={
            "introduction": "Clinical retrieval augmented generation matters because grounded oncology question answering depends on evidence quality.",
            "discussion": "The discussion argues that retrieval quality, evidence coverage, and citation discipline jointly improve grounded oncology question answering.",
            "conclusion": "In conclusion, the essay argues that clinical retrieval augmented generation is most effective when retrieval quality and evidence coverage remain central.",
        },
        quality_sections={
            "introduction": {"approved": True, "score": 0.84},
            "discussion": {"approved": True, "score": 0.81},
            "conclusion": {"approved": True, "score": 0.8},
        },
    )

    assert result["approved"] is True
    assert result["score"] >= 0.72
    assert result["topic_coverage"] >= 0.35


def test_writer_template_fallback_is_not_repetitive():
    from app.agents.writer import WriterAgent
    agent = WriterAgent()

    result = agent._template_write(
        section="introduction",
        topic="sustainable agriculture",
        word_count_target=220,
        section_plan={
            "thesis_goal": "Argue that resilient agricultural systems depend on integrated soil and water management.",
            "must_cover": ["soil health", "water efficiency", "yield stability"],
            "evidence_requirements": ["cite context-setting sources", "state at least one limitation"],
            "writing_directive": "Move from context to thesis quickly.",
        },
        feedback="Make the claims more specific and tie them to the evidence pack.",
        research_data={
            "section_queries": ["yield optimization", "soil health", "water efficiency"],
            "research_summary": "Recent studies converge on integrated soil and irrigation management as a major driver of resilient crop output.",
            "evidence_pack": [
                {
                    "title": "Soil Carbon and Yield Stability",
                    "year": 2021,
                    "source": "semantic_scholar",
                    "abstract_excerpt": "Longitudinal field data show that farms with higher soil carbon had lower inter-annual yield volatility under drought conditions.",
                },
                {
                    "title": "Drip Irrigation Meta-analysis",
                    "year": 2020,
                    "source": "web",
                    "abstract_excerpt": "A meta-analysis across 40 studies reported improved water productivity in drip systems relative to flood irrigation.",
                },
            ],
        },
    )

    content = result["content"]
    assert "Evidence highlights:" in content
    assert "[1]" in content
    assert "[2]" in content
    assert "Section objective:" not in content
    assert "Revision guidance to address:" not in content
    assert "Writing directive:" not in content
    assert content.count("This paper examines sustainable agriculture") <= 1
    assert result["word_count"] >= 160


def test_planner_normalize_plan_filters_excluded_sections():
    from app.agents.planner import PlannerAgent

    agent = PlannerAgent()
    normalized = agent._normalize_plan({
        "sections": [
            {
                "key": "introduction",
                "title": "Introduction",
                "include": True,
                "research_queries": ["topic context"],
                "word_count_target": 350,
            },
            {
                "key": "results",
                "title": "Results",
                "include": False,
                "research_queries": ["empirical findings"],
                "word_count_target": 500,
            },
        ],
        "research_queries": ["topic context", "empirical findings"],
        "estimated_total_words": 850,
    })

    keys = [section["key"] for section in normalized["sections"]]
    assert "introduction" in keys
    assert "results" not in keys


def test_writer_extracts_light_subheadings_from_markdown():
    from app.agents.writer import WriterAgent

    agent = WriterAgent()
    extracted = agent._extract_subheadings(
        "## Primary Finding\nEvidence-backed paragraph.\n\n## Limitation\nSome uncertainty remains.",
    )
    assert len(extracted) == 2
    assert extracted[0]["title"] == "Primary Finding"
    assert "Evidence-backed" in extracted[0]["content"]


def test_reviewer_heuristic_rejects_generic_ungrounded_content():
    from app.agents.reviewer import ReviewerAgent

    agent = ReviewerAgent()
    result = agent._heuristic_review(
        section="literature_review",
        content=(
            "A substantial body of literature exists on machine learning. "
            "Researchers have explored multiple dimensions. "
            "The analysis of machine learning is important."
        ),
        expected_word_count=350,
        evidence_pack=[
            {"title": "Transformer Models in Clinical NLP", "year": 2024, "source": "semantic_scholar"},
            {"title": "Benchmarking Retrieval-Augmented Generation", "year": 2023, "source": "web"},
        ],
        revision_attempt=0,
    )

    assert result["approved"] is False
    assert result["score"] < 0.75
    assert result["blocking_issues"]
    assert any("cite" in suggestion.lower() or "evidence" in suggestion.lower() for suggestion in result["suggestions"])


def test_reviewer_rejects_meta_instructional_text():
    from app.agents.reviewer import ReviewerAgent

    agent = ReviewerAgent()
    content = (
        "Section objective: explain the topic and map the argument. "
        "This section must address source quality and methods. "
        "Writing directive: include evidence requirements and revision guidance to address weak claims."
    )
    result = agent._heuristic_review(
        section="introduction",
        content=content,
        expected_word_count=220,
        evidence_pack=[
            {"title": "Source Quality Study", "year": 2024, "source": "semantic_scholar"},
        ],
        revision_attempt=0,
    )

    assert result["approved"] is False
    assert any("instructional/meta-writing" in issue.lower() for issue in result["blocking_issues"])


def test_reviewer_heuristic_rewards_grounded_specific_content():
    from app.agents.reviewer import ReviewerAgent

    agent = ReviewerAgent()
    content = (
        "Transformer-based retrieval systems improved answer grounding in biomedical question answering, with benchmark gains reported in 2023 and 2024 [1][2]. "
        "The strongest studies linked those gains to better evidence retrieval rather than to larger generation models alone, which suggests that retrieval quality is a primary driver of answer fidelity [1].\n\n"
        "However, the evidence also suggests that performance varies by corpus coverage, annotation quality, and domain drift, so claims should be interpreted cautiously. "
        "Several evaluation papers note that improvements on narrow benchmark datasets do not always transfer to clinical deployment settings, particularly when terminology and document structure differ from the training distribution [2].\n\n"
        "A major limitation is that many comparisons still rely on short-form factoid questions instead of longitudinal diagnostic reasoning, which leaves uncertainty about how well the reported gains generalize to high-stakes workflows. "
        "Taken together, the section is grounded in cited evidence, makes a specific comparative claim, and clearly identifies where the current literature remains incomplete."
    )
    result = agent._heuristic_review(
        section="discussion",
        content=content,
        expected_word_count=180,
        evidence_pack=[
            {"title": "Biomedical Retrieval-Augmented Generation", "year": 2024, "source": "semantic_scholar"},
            {"title": "Clinical QA Benchmark Study", "year": 2023, "source": "web"},
        ],
        revision_attempt=1,
    )

    assert result["approved"] is True
    assert result["score"] >= 0.75
    assert result["citation_count"] >= 2
    assert result["category_scores"]["grounding"] >= 0.75


@pytest.mark.asyncio
async def test_reviewer_agent(db, test_project):
    from app.agents.reviewer import ReviewerAgent
    agent = ReviewerAgent()
    content = "This is a test section about machine learning. " * 20
    result = await agent.execute(
        {"section": "introduction", "content": content},
        test_project.id, db
    )
    assert "score" in result
    assert "feedback" in result
    assert "suggestions" in result
    assert "approved" in result
    assert 0.0 <= result["score"] <= 1.0
    assert "category_scores" in result


def test_reviewer_respects_grounding_summary():
    from app.agents.reviewer import ReviewerAgent

    agent = ReviewerAgent()
    result = agent._heuristic_review(
        section="results",
        content=(
            "The model improves answer fidelity across all domains [1]. "
            "However, results vary by corpus quality and coverage."
        ),
        expected_word_count=120,
        evidence_pack=[{"title": "Answer Fidelity Study", "year": 2024, "source": "web"}],
        grounding_summary={
            "score": 0.45,
            "unsupported_claim_count": 1,
            "issues": ["Some claim-like sentences lack both citations and evidence alignment."],
        },
        revision_attempt=0,
    )

    assert result["approved"] is False
    assert result["grounding_score"] == 0.45
    assert any("grounding" in issue.lower() or "claim" in issue.lower() for issue in result["blocking_issues"])


@pytest.mark.asyncio
async def test_citation_agent(db, test_project):
    from app.agents.citation import CitationAgent
    sources = [
        {"title": "Test Paper", "authors": ["Smith, J.", "Doe, A."], "year": 2023,
         "url": "https://arxiv.org/test", "doi": "10.1234/test"},
    ]
    agent = CitationAgent()
    result = await agent.execute({"sources": sources, "style": "harvard"}, test_project.id, db)
    assert "formatted_citations" in result
    assert "bibliography" in result
    assert "in_text_citations" in result
    assert len(result["formatted_citations"]) == 1

    result_ieee = await agent.execute({"sources": sources, "style": "ieee"}, test_project.id, db)
    assert "[1]" in result_ieee["formatted_citations"][0]


@pytest.mark.asyncio
async def test_figure_agent(db, test_project):
    from app.agents.figure import FigureAgent
    agent = FigureAgent()
    result = await agent.execute(
        {"topic": "machine learning", "section": "results", "data": {}},
        test_project.id, db
    )
    assert "figures" in result
    assert isinstance(result["figures"], list)


# ---------------------------------------------------------------------------
# New tests for quality / token-efficiency improvements
# ---------------------------------------------------------------------------

def test_truncate_text_helper():
    from app.agents.llm_client import truncate_text

    assert truncate_text("hello world", 5) == "hello…"
    assert truncate_text("short", 100) == "short"
    assert truncate_text("", 10) == ""
    assert truncate_text("exactly", 7) == "exactly"


def test_quality_max_tokens_quality_mode(monkeypatch):
    from app.core.config import settings
    import app.agents.llm_client as llm_module

    monkeypatch.setattr(settings, "QUALITY_MODE", "quality")
    monkeypatch.setattr(settings, "LLM_MAX_TOKENS", 4096)
    result = llm_module.quality_max_tokens()
    assert result == 4096


def test_quality_max_tokens_balanced_mode(monkeypatch):
    from app.core.config import settings
    import app.agents.llm_client as llm_module

    monkeypatch.setattr(settings, "QUALITY_MODE", "balanced")
    monkeypatch.setattr(settings, "LLM_MAX_TOKENS", 4096)
    result = llm_module.quality_max_tokens()
    assert result <= 2048


def test_find_cache_split_returns_zero_for_short_prompt():
    from app.agents.llm_client import _find_cache_split

    short = "Write an essay.\nBe concise."
    assert _find_cache_split(short) == 0


def test_find_cache_split_splits_long_prompt():
    from app.agents.llm_client import _find_cache_split

    context = "Background context. " * 30      # ~600 chars
    # Task must be at least min_trailing (200 chars) from the end
    task = "Now write the introduction section. " * 10  # ~360 chars
    prompt = context + "\n\n" + task
    split = _find_cache_split(prompt)
    assert split > 0
    assert prompt[split:].strip()  # trailing part is non-empty


def test_research_agent_is_generic_query():
    from app.agents.research import ResearchAgent

    agent = ResearchAgent()
    assert agent._is_generic_query("overview") is True
    assert agent._is_generic_query("methods survey") is True
    assert agent._is_generic_query("transformer attention mechanisms NLP 2024") is False


def test_coherence_agent_llm_fallback_on_error(monkeypatch):
    """_llm_coherence should return the heuristic result on LLM errors."""
    from app.agents.coherence import CoherenceAgent
    import app.agents.coherence as coh_module

    async def fake_timed(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(coh_module, "timed_chat_completion", fake_timed)
    monkeypatch.setattr(coh_module, "is_llm_available", lambda: True)

    agent = CoherenceAgent()
    heuristic = {
        "score": 0.8,
        "approved": True,
        "feedback": "ok",
        "issues": [],
        "suggestions": [],
        "flagged_sections": [],
        "repeated_opening_sections": [],
    }

    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        agent._llm_coherence("test topic", {"introduction": "content"}, heuristic, "pid", None)
    )
    assert result == heuristic


@pytest.mark.asyncio
async def test_coherence_agent_uses_llm_when_available(db, test_project, monkeypatch):
    from app.agents.coherence import CoherenceAgent
    import app.agents.coherence as coh_module
    import json

    llm_response = json.dumps({
        "score": 0.85,
        "approved": True,
        "feedback": "Good overall coherence.",
        "issues": [],
        "suggestions": [],
    })

    async def fake_timed(*args, **kwargs):
        return llm_response

    monkeypatch.setattr(coh_module, "timed_chat_completion", fake_timed)
    monkeypatch.setattr(coh_module, "is_llm_available", lambda: True)

    agent = CoherenceAgent()
    result = await agent.execute(
        {
            "topic": "machine learning",
            "sections": {
                "introduction": "Machine learning methods are advancing rapidly with transformer models.",
                "conclusion": "In conclusion, machine learning, especially transformers, continue to transform the field.",
            },
            "quality_sections": {
                "introduction": {"approved": True, "score": 0.85},
                "conclusion": {"approved": True, "score": 0.82},
            },
        },
        test_project.id,
        db,
    )
    assert "score" in result
    assert "approved" in result
    assert "issues" in result
    # Blended score should incorporate both LLM and heuristic
    assert 0.0 <= result["score"] <= 1.0


def test_build_section_evidence_includes_recency_bonus():
    """Newer sources should score higher than older identical sources."""
    from app.orchestration.worker_pool import WorkerPool
    import time as _time

    pool = WorkerPool()
    current_year = _time.gmtime().tm_year

    sources = [
        {
            "title": "Neural Attention Mechanisms for NLP",
            "abstract": "attention mechanisms neural language processing overview",
            "year": current_year - 1,   # recent
            "source": "arxiv",
            "relevance_score": 0.5,
            "combined_quality_score": 0.5,
            "verification_score": 0.6,
        },
        {
            "title": "Neural Attention Mechanisms for NLP",
            "abstract": "attention mechanisms neural language processing overview",
            "year": current_year - 12,  # older
            "source": "arxiv",
            "relevance_score": 0.5,
            "combined_quality_score": 0.5,
            "verification_score": 0.6,
        },
    ]
    section_info = {
        "key": "literature_review",
        "title": "Literature Review",
        "description": "survey of neural attention mechanisms",
        "research_queries": ["neural attention NLP"],
        "must_cover": ["attention mechanisms"],
    }
    evidence = pool._build_section_evidence(section_info, sources)
    assert evidence, "Expected at least one evidence item"
    assert evidence[0]["year"] == current_year - 1, "Recent source should rank first"

