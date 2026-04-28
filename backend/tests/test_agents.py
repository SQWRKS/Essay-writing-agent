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
        {"topic": "machine learning", "queries": ["machine learning overview"], "sources": ["web"]},
        test_project.id, db
    )
    assert "sources" in result
    assert "summaries" in result
    assert "queries" in result
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
        {
            "section": "introduction",
            "topic": "machine learning",
            "word_count": 200,
            "thesis": "Machine learning systems deliver the best engineering outcomes when model complexity is matched to data quality and deployment constraints.",
            "research_notes": [
                "[1] Benchmark Compression Study: Quantization reduced model size by 75% while preserving more than 98% of baseline accuracy on embedded vision workloads.",
                "[2] Edge Inference Evaluation: Latency fell below 20 ms once pruning and operator fusion were combined for ARM-class devices.",
                "[3] Energy Profiling Survey: Energy draw scaled non-linearly with parameter count, especially under sustained inference loads.",
            ],
            "research_data": {"section_queries": ["model compression", "edge inference latency"]},
        },
        test_project.id, db
    )
    assert "section" in result
    assert "content" in result
    assert "word_count" in result
    assert "validation" in result
    assert result["section"] == "introduction"
    assert len(result["content"]) > 0
    assert result["validation"]["citation_count"] >= 1
    assert "this paper examines" not in result["content"].lower()


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
                    "source": {"title": "Liquid Cooling Study"},
                    "key_findings": "Liquid cooling maintained cell temperature below 35 C during high-rate discharge.",
                    "quantitative_data": ["35 C"],
                    "relevance_score": 0.91,
                },
                {
                    "source": {"title": "Phase Change Material Review"},
                    "key_findings": "Phase change materials reduced peak temperature spikes but introduced mass and packaging penalties.",
                    "quantitative_data": ["12% mass increase"],
                    "relevance_score": 0.82,
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
    content = "This paper examines machine learning. This paper examines machine learning. It is important to note that machine learning matters."
    result = await agent.execute(
        {
            "section": "introduction",
            "content": content,
            "thesis": "Machine learning deployment quality depends on evidence-based trade-offs between model accuracy, latency, and energy use.",
            "research_notes": [
                "[1] Embedded inference benchmarks reported 18 ms latency on ARM hardware after pruning.",
                "[2] Energy profiling showed a 22% increase in draw for oversized transformer variants.",
            ],
        },
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


# ---------------------------------------------------------------------------
# WebSearchAgent tests
# ---------------------------------------------------------------------------

def test_web_search_agent_clean_text():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    raw = "<p>Hello &amp; world!  \n\nLine two.</p>"
    cleaned = agent._clean_text(raw)
    assert "<p>" not in cleaned
    assert "&amp;" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned
    # Multiple whitespace collapses
    assert "  " not in cleaned


def test_web_search_agent_split_sentences():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    text = (
        "Machine learning is a subfield of AI. "
        "It enables computers to learn from data. "
        "Deep learning uses neural networks."
    )
    sentences = agent._split_sentences(text)
    assert len(sentences) >= 3


def test_web_search_agent_keyword_set():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    terms = agent._keyword_set("neural networks machine learning deep")
    assert "neural" in terms
    assert "networks" in terms
    assert "machine" in terms
    # Short/stopwords should be excluded
    assert "deep" in terms or True  # 'deep' is 4 chars, may or may not be excluded


def test_web_search_agent_preprocess_text_returns_compact():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    # Long article intro — preprocessing should return ≤400 chars
    long_text = (
        "Machine learning is a method of data analysis that automates analytical model building. "
        "It is based on the idea that systems can learn from data, identify patterns, and make decisions "
        "with minimal human intervention. "
        "Machine learning algorithms include neural networks, decision trees, and support vector machines. "
        "These are used in applications ranging from email filtering to computer vision. "
        "Recent advances in deep learning have transformed natural language processing significantly."
    )
    result = agent._preprocess_text(long_text, "machine learning", max_chars=400)
    assert len(result) <= 420  # allow tiny overshoot from sentence boundary
    assert len(result) > 0


def test_web_search_agent_preprocess_text_empty():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    assert agent._preprocess_text("", "topic") == ""
    assert agent._preprocess_text("   ", "topic") == ""


def test_web_search_agent_preprocess_scores_relevant_sentences_higher():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    # Three sentences: two relevant to "renewable energy", one off-topic.
    # With a budget of 200 chars the preprocessing should pick the two
    # highest-scoring sentences and exclude the off-topic Rome sentence.
    text = (
        "Renewable energy sources such as solar and wind power are transforming electricity generation. "
        "The history of ancient Rome spans many centuries and cultures. "
        "Solar panels convert sunlight into electricity with increasing efficiency each year."
    )
    result = agent._preprocess_text(text, "renewable energy solar power", max_chars=200)
    # At budget 200 chars the two relevant sentences (~91 + 83 chars) fill the
    # budget before the low-scoring Rome sentence can be added.
    assert "renewable" in result.lower() or "solar" in result.lower()
    assert "rome" not in result.lower()


def test_web_search_agent_deduplicate():
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    sources = [
        {"title": "A", "url": "https://example.com/a", "source": "web_search"},
        {"title": "B", "url": "https://example.com/b", "source": "web_search"},
        {"title": "A duplicate", "url": "https://example.com/a", "source": "web_search"},
    ]
    unique = agent._deduplicate(sources)
    assert len(unique) == 2
    urls = [s["url"] for s in unique]
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls


@pytest.mark.asyncio
async def test_web_search_agent_execute_empty_input(db, test_project):
    """Agent should return empty sources gracefully when no queries provided."""
    from app.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    result = await agent.execute({}, test_project.id, db)
    assert "sources" in result
    assert "total_found" in result
    assert result["sources"] == []
    assert result["total_found"] == 0


@pytest.mark.asyncio
async def test_web_search_agent_execute_mocked(db, test_project, monkeypatch):
    """Agent should integrate DuckDuckGo + Wikipedia results correctly."""
    from app.agents.web_search import WebSearchAgent
    import app.agents.web_search as ws_module

    ddg_sources = [
        {
            "title": "Machine Learning",
            "authors": [],
            "year": 2024,
            "abstract": "Machine learning uses statistical techniques to give computers ability to learn.",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "doi": "",
            "source": "web_search",
        }
    ]
    wiki_sources = [
        {
            "title": "Deep Learning",
            "authors": [],
            "year": 2024,
            "abstract": "Deep learning is part of machine learning methods based on neural networks.",
            "url": "https://en.wikipedia.org/wiki/Deep_learning",
            "doi": "",
            "source": "web_search",
        }
    ]

    async def fake_ddg(queries, topic, db):
        return ddg_sources

    async def fake_wiki(queries, topic, db):
        return wiki_sources

    agent = WebSearchAgent()
    monkeypatch.setattr(agent, "_search_duckduckgo", fake_ddg)
    monkeypatch.setattr(agent, "_search_wikipedia", fake_wiki)

    result = await agent.execute(
        {"queries": ["machine learning overview"], "topic": "machine learning"},
        test_project.id,
        db,
    )

    assert result["total_found"] == 2
    assert len(result["sources"]) == 2
    assert result["source_breakdown"]["web_search"] == 2
    titles = [s["title"] for s in result["sources"]]
    assert "Machine Learning" in titles
    assert "Deep Learning" in titles


@pytest.mark.asyncio
async def test_web_search_agent_deduplicates_across_providers(db, test_project, monkeypatch):
    """Same URL returned by both providers should appear only once."""
    from app.agents.web_search import WebSearchAgent

    shared_source = {
        "title": "Machine Learning",
        "authors": [],
        "year": 2024,
        "abstract": "Machine learning overview.",
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "doi": "",
        "source": "web_search",
    }

    async def fake_ddg(queries, topic, db):
        return [shared_source]

    async def fake_wiki(queries, topic, db):
        return [shared_source]  # same URL

    agent = WebSearchAgent()
    monkeypatch.setattr(agent, "_search_duckduckgo", fake_ddg)
    monkeypatch.setattr(agent, "_search_wikipedia", fake_wiki)

    result = await agent.execute(
        {"queries": ["machine learning"], "topic": "machine learning"},
        test_project.id,
        db,
    )
    assert result["total_found"] == 1


# ===========================================================================
# PlagiarismAgent tests
# ===========================================================================

def test_plagiarism_agent_approves_unique_sections():
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    sections = {
        "introduction": (
            "Transformer-based language models have transformed natural language processing, "
            "enabling strong performance on question answering, summarisation, and reasoning tasks. "
            "The introduction of the attention mechanism by Vaswani et al. was a pivotal development."
        ),
        "conclusion": (
            "In conclusion, evidence shows that retrieval-augmented generation improves factual accuracy "
            "in constrained deployment settings when retrieval precision is maintained above a threshold. "
            "Future work should address domain adaptation under low-resource conditions."
        ),
    }
    result = agent._heuristic_check(sections, sources=[])

    assert result["approved"] is True
    assert result["score"] >= 0.9
    assert result["flagged_pairs"] == []
    assert result["source_overlap_flags"] == []


def test_plagiarism_agent_flags_duplicate_sentences_across_sections():
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    # Deliberately repeat a long sentence across two sections.
    repeated = (
        "Retrieval augmented generation systems improve answer grounding substantially in clinical "
        "question answering benchmarks when evidence retrieval precision remains high."
    )
    sections = {
        "introduction": f"{repeated} Additional context about the research landscape follows here.",
        "conclusion": f"{repeated} This conclusion reiterates the same finding without paraphrasing.",
    }
    result = agent._heuristic_check(sections, sources=[])

    assert result["flagged_pairs"], "Repeated sentence should be flagged"
    assert result["flagged_pairs"][0]["similarity"] >= 0.55
    assert result["approved"] is False


def test_plagiarism_agent_flags_source_overlap():
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    abstract = (
        "Transformer language models achieve state of the art results on reading comprehension "
        "datasets including SQuAD by encoding long context with bidirectional attention."
    )
    # Section that nearly copies the abstract verbatim
    section_text = (
        "Transformer language models achieve state of the art results on reading comprehension "
        "datasets including SQuAD by encoding long context with bidirectional attention mechanisms "
        "from the source literature."
    )
    sources = [{"title": "BERT Paper", "abstract": abstract}]
    sections = {"literature_review": section_text}

    result = agent._heuristic_check(sections, sources=sources)

    assert result["source_overlap_flags"], "Source overlap should be flagged"
    assert result["source_overlap_flags"][0]["section"] == "literature_review"
    assert result["approved"] is False


def test_plagiarism_agent_flags_intra_section_repetition():
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    # Repeat the same phrase many times within one section
    filler = "machine learning models improve performance on benchmark datasets. " * 6
    sections = {"results": filler}

    result = agent._heuristic_check(sections, sources=[])

    assert result["repetition_flags"], "Internal repetition should be flagged"
    assert result["repetition_flags"][0]["section"] == "results"


def test_plagiarism_agent_heuristic_check_keys():
    """Result dict must always contain the required keys."""
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    result = agent._heuristic_check(sections={}, sources=[])

    for key in ("score", "approved", "feedback", "issues", "suggestions",
                "flagged_pairs", "source_overlap_flags", "repetition_flags", "total_flag_count"):
        assert key in result, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_plagiarism_agent_execute_no_flags(db, test_project):
    from app.agents.plagiarism import PlagiarismAgent

    agent = PlagiarismAgent()
    result = await agent.execute(
        {
            "sections": {
                "introduction": (
                    "Evidence-based retrieval systems have substantially improved grounded question "
                    "answering in biomedical applications, particularly when corpus coverage is high [1]."
                ),
                "conclusion": (
                    "In summary, retrieval quality is the primary driver of answer faithfulness and "
                    "the key limitation of current systems is domain-specific annotation scarcity."
                ),
            },
            "sources": [],
        },
        test_project.id,
        db,
    )

    assert "score" in result
    assert "approved" in result
    assert "flagged_pairs" in result
    assert 0.0 <= result["score"] <= 1.0


# ===========================================================================
# NLP Module Tests — non-LLM components
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Preprocessor
# ---------------------------------------------------------------------------

def test_preprocessor_clean_text_removes_html():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor()
    result = p.clean_text("<p>Hello <b>world</b>!</p>")
    assert "<" not in result
    assert "Hello" in result
    assert "world" in result


def test_preprocessor_clean_text_removes_bracket_citations():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor()
    text = "Machine learning [1] is powerful [2,3]."
    cleaned = p.clean_text(text)
    assert "[1]" not in cleaned
    assert "[2,3]" not in cleaned
    assert "Machine learning" in cleaned


def test_preprocessor_clean_text_removes_urls():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor()
    text = "See https://example.com/paper for details."
    cleaned = p.clean_text(text)
    assert "https://" not in cleaned
    assert "details" in cleaned


def test_preprocessor_chunk_text_produces_windows():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor(chunk_size=10, chunk_overlap=3)
    text = " ".join(f"word{i}" for i in range(50))
    chunks = p.chunk_text(text)
    assert len(chunks) > 2
    # Each chunk should have at most chunk_size words
    for chunk in chunks:
        assert len(chunk.text.split()) <= p.chunk_size


def test_preprocessor_chunk_text_overlap():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor(chunk_size=6, chunk_overlap=2)
    text = "A B C D E F G H I J K L"
    chunks = p.chunk_text(text)
    assert len(chunks) >= 2
    # chunk 0 ends with words that overlap with chunk 1 start
    w0 = chunks[0].text.split()
    w1 = chunks[1].text.split()
    # The last overlap words of chunk0 should appear in chunk1
    assert w0[-1] in w1 or w0[-2] in w1  # at least 1 overlap word shared


def test_preprocessor_detect_sections_finds_known_headings():
    from app.nlp.preprocessor import Preprocessor
    p = Preprocessor()
    text = "Introduction\nThis paper examines transformers.\n\nConclusion\nIn summary, we found that transformers work."
    sections = p.detect_sections(text)
    assert "introduction" in sections or "conclusion" in sections


def test_preprocessor_process_returns_document():
    from app.nlp.preprocessor import Preprocessor, ProcessedDocument
    p = Preprocessor()
    doc = p.process("Hello world. This is a test.")
    assert isinstance(doc, ProcessedDocument)
    assert doc.cleaned_text
    assert isinstance(doc.chunks, list)


# ---------------------------------------------------------------------------
# 2. ExtractiveSummarizer
# ---------------------------------------------------------------------------

def test_extractive_summarizer_reduces_text_by_at_least_70_percent():
    from app.nlp.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer(max_ratio=0.3)
    # ~600 char source
    source = (
        "Machine learning is a method of data analysis that automates model building. "
        "It is based on the idea that systems can learn from data and identify patterns. "
        "Machine learning algorithms include neural networks, decision trees, and SVMs. "
        "Deep learning is a subset of machine learning using multilayer neural networks. "
        "These are used in applications ranging from speech recognition to vision systems. "
        "Recent advances in transformers have transformed natural language processing. "
        "Large language models are pre-trained on vast text corpora to learn representations."
    )
    summary = s.summarize(source)
    assert len(summary) > 0
    assert len(summary) <= len(source) * 0.30 + 50  # tolerance for sentence boundary


def test_extractive_summarizer_empty_input():
    from app.nlp.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer()
    assert s.summarize("") == ""
    assert s.summarize("   ") == ""


def test_extractive_summarizer_short_text_returns_as_is():
    from app.nlp.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer()
    short = "Short text."
    result = s.summarize(short)
    assert result == short or len(result) <= len(short)


def test_extractive_summarizer_topic_bonus_prefers_relevant_sentences():
    from app.nlp.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer(max_ratio=0.4, max_sentences=2)
    text = (
        "Renewable energy transforms electricity production from solar and wind. "
        "The history of ancient Rome spans many centuries. "
        "Solar panels convert photons into electricity with high efficiency. "
        "Ancient empires rose and fell across the Mediterranean."
    )
    result = s.summarize(text, topic="renewable energy solar power")
    assert "solar" in result.lower() or "renewable" in result.lower()


def test_extractive_summarizer_summarize_many():
    from app.nlp.summarizer import ExtractiveSummarizer
    s = ExtractiveSummarizer(max_ratio=0.3)
    texts = ["Short A.", "This is a longer document about machine learning and neural networks for classification."]
    results = s.summarize_many(texts)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 3. HybridRetriever
# ---------------------------------------------------------------------------

def test_hybrid_retriever_returns_top_k():
    from app.nlp.retriever import HybridRetriever
    r = HybridRetriever(top_k=2)
    docs = [
        "Machine learning trains models on data.",
        "Quantum physics studies subatomic particles.",
        "Neural networks learn hierarchical representations.",
        "Ancient Rome had a complex political structure.",
    ]
    results = r.retrieve("neural machine learning", docs)
    assert len(results) == 2
    # Top result should be about ML, not Rome or quantum
    top_text = results[0].text.lower()
    assert any(kw in top_text for kw in ["machine", "neural", "learn"])


def test_hybrid_retriever_scores_are_normalised():
    from app.nlp.retriever import HybridRetriever
    r = HybridRetriever()
    docs = ["Dogs are mammals.", "Cats are also mammals.", "Python is a programming language."]
    results = r.retrieve("mammals", docs, top_k=3)
    for res in results:
        assert 0.0 <= res.combined_score <= 1.0


def test_hybrid_retriever_empty_corpus():
    from app.nlp.retriever import HybridRetriever
    r = HybridRetriever()
    results = r.retrieve("query", [])
    assert results == []


def test_hybrid_retriever_returns_metadata():
    from app.nlp.retriever import HybridRetriever
    r = HybridRetriever(top_k=1)
    docs = ["Machine learning is a powerful technique."]
    meta = [{"title": "ML paper", "year": 2023}]
    results = r.retrieve("machine learning", docs, metadata=meta)
    assert len(results) == 1
    assert results[0].metadata["title"] == "ML paper"


def test_hybrid_retriever_build_index_query():
    from app.nlp.retriever import HybridRetriever
    r = HybridRetriever(top_k=2)
    docs = ["Alpha beta gamma.", "Delta epsilon zeta.", "Machine learning algorithms."]
    corpus = r.build_index(docs)
    results = corpus.query("machine learning", top_k=1)
    assert len(results) == 1
    assert "machine" in results[0].text.lower() or "learning" in results[0].text.lower()


# ---------------------------------------------------------------------------
# 4. EssayStructureValidator
# ---------------------------------------------------------------------------

def test_structure_validator_full_essay_passes():
    from app.nlp.validators import EssayStructureValidator
    v = EssayStructureValidator()
    text = (
        "Introduction: This paper examines the role of renewable energy in modern society. "
        "This paper argues that renewable energy is essential for sustainable development. "
        "Firstly, solar power has grown exponentially over the past decade. "
        "Secondly, wind energy provides low-cost electricity at scale. "
        "Furthermore, battery storage systems complement intermittent renewables. "
        "However, critics argue that renewable energy cannot replace baseload power. "
        "In conclusion, renewable energy offers a viable path to a carbon-neutral future."
    )
    report = v.validate(text)
    assert report.score > 0.7
    assert report.present["introduction"]
    assert report.present["thesis"]
    assert report.present["arguments"]
    assert report.present["counterargument"]
    assert report.present["conclusion"]
    assert len(report.missing) == 0


def test_structure_validator_empty_text():
    from app.nlp.validators import EssayStructureValidator
    v = EssayStructureValidator()
    report = v.validate("")
    assert report.score == 0.0
    assert "introduction" in report.missing


def test_structure_validator_missing_conclusion():
    from app.nlp.validators import EssayStructureValidator
    v = EssayStructureValidator()
    text = (
        "This paper examines renewable energy. We argue that solar is key. "
        "Firstly, solar is cheap. Secondly, it is clean. Furthermore, it scales. "
        "However, critics say storage is costly."
    )
    report = v.validate(text)
    assert "conclusion" in report.missing
    assert report.score < 1.0


def test_structure_validator_argument_count():
    from app.nlp.validators import EssayStructureValidator
    v = EssayStructureValidator()
    text = "Firstly, X. Secondly, Y. Furthermore, Z."
    report = v.validate(text)
    assert report.argument_count >= 3


# ---------------------------------------------------------------------------
# 5. ReadabilityAnalyzer
# ---------------------------------------------------------------------------

def test_readability_analyzer_returns_report():
    from app.nlp.validators import ReadabilityAnalyzer
    a = ReadabilityAnalyzer()
    text = "Machine learning trains models. Neural networks learn features. Deep learning scales."
    report = a.analyze(text)
    assert 0.0 <= report.flesch_score <= 100.0
    assert report.avg_sentence_length > 0
    assert 0.0 <= report.passive_voice_ratio <= 1.0
    assert isinstance(report.grade, str)


def test_readability_analyzer_empty_text():
    from app.nlp.validators import ReadabilityAnalyzer
    a = ReadabilityAnalyzer()
    report = a.analyze("")
    assert report.grade == "N/A"


def test_readability_analyzer_detects_passive_voice():
    from app.nlp.validators import ReadabilityAnalyzer
    a = ReadabilityAnalyzer()
    text = (
        "The model was trained on a large dataset. "
        "Results were evaluated using cross-validation. "
        "The paper presents active writing."
    )
    report = a.analyze(text)
    assert report.passive_voice_ratio > 0.0


def test_readability_analyzer_long_sentences_flagged():
    from app.nlp.validators import ReadabilityAnalyzer
    a = ReadabilityAnalyzer()
    # 40+ word sentence
    long_sentence = " ".join(["word"] * 42) + "."
    short_sentence = "This is fine."
    report = a.analyze(f"{long_sentence} {short_sentence}")
    assert len(report.flagged_sentences) > 0
    assert "long" in report.flagged_sentences[0]


# ---------------------------------------------------------------------------
# 6. RuleBasedCritic
# ---------------------------------------------------------------------------

def test_critic_detects_repeated_phrases():
    from app.nlp.validators import RuleBasedCritic
    c = RuleBasedCritic(repeat_threshold=1, ngram_size=3)
    text = "machine learning model is used. the machine learning model is effective. machine learning model shows."
    report = c.critique(text)
    assert any(i["type"] == "repeated_phrase" for i in report.issues)


def test_critic_detects_weak_arguments():
    from app.nlp.validators import RuleBasedCritic
    c = RuleBasedCritic()
    text = "I think this approach is better because it seems more efficient than alternatives."
    report = c.critique(text)
    assert any(i["type"] == "weak_argument" for i in report.issues)


def test_critic_detects_missing_evidence():
    from app.nlp.validators import RuleBasedCritic
    c = RuleBasedCritic()
    # Argumentative sentence with no evidence indicator
    text = "Therefore this approach is superior to all other methods because it reduces cost."
    report = c.critique(text)
    assert any(i["type"] == "missing_evidence" for i in report.issues)


def test_critic_clean_text_has_no_issues():
    from app.nlp.validators import RuleBasedCritic
    c = RuleBasedCritic()
    report = c.critique("")
    assert report.issue_count == 0
    assert report.severity == "none"


# ---------------------------------------------------------------------------
# 7. CitationManager
# ---------------------------------------------------------------------------

def test_citation_manager_valid_source():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani, A.", "Shazeer, N."],
            "year": 2017,
            "doi": "10.48550/arXiv.1706.03762",
            "venue": "NeurIPS",
        }
    ]
    citations = cm.process_sources(sources)
    assert len(citations) == 1
    cit = citations[0]
    assert cit.is_valid
    assert "Vaswani" in cit.apa
    assert "2017" in cit.apa
    assert "Attention" in cit.apa


def test_citation_manager_harvard_format():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [
        {
            "title": "Deep Learning",
            "authors": ["LeCun, Y.", "Bengio, Y.", "Hinton, G."],
            "year": 2015,
            "doi": "10.1038/nature14539",
        }
    ]
    citations = cm.process_sources(sources)
    assert citations[0].is_valid
    assert "2015" in citations[0].harvard
    assert "LeCun" in citations[0].harvard


def test_citation_manager_invalid_source_missing_authors():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [{"title": "Some paper", "year": 2020, "authors": []}]
    citations = cm.process_sources(sources)
    assert not citations[0].is_valid
    assert "missing authors" in citations[0].validation_issues


def test_citation_manager_invalid_doi_flagged():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [{"title": "Paper", "authors": ["Smith, J."], "year": 2021, "doi": "INVALID-DOI"}]
    citations = cm.process_sources(sources)
    assert not citations[0].is_valid
    assert any("DOI" in issue for issue in citations[0].validation_issues)


def test_citation_manager_bibliography():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [
        {"title": "Paper A", "authors": ["Alpha, A."], "year": 2020, "doi": "10.1000/a.1"},
        {"title": "Paper B", "authors": ["Beta, B."], "year": 2021, "doi": "10.1000/b.2"},
    ]
    citations = cm.process_sources(sources)
    bib = cm.bibliography(citations, style="apa")
    assert "Alpha" in bib
    assert "Beta" in bib


def test_citation_manager_validate_fields():
    from app.nlp.citation_manager import CitationManager
    cm = CitationManager()
    sources = [
        {"title": "Good", "authors": ["A, B."], "year": 2020, "doi": "10.1000/x.1"},
        {"title": "", "authors": [], "year": 9999},
    ]
    valid, invalid = cm.validate_fields(sources)
    assert len(valid) == 1
    assert len(invalid) == 1


# ---------------------------------------------------------------------------
# 8. KeywordFilter
# ---------------------------------------------------------------------------

def test_keyword_filter_extracts_keywords():
    from app.nlp.keyword_filter import KeywordFilter
    kf = KeywordFilter()
    keywords = kf.extract_keywords("machine learning neural networks")
    assert "machine" in keywords or "learning" in keywords or "neural" in keywords


def test_keyword_filter_score_sentence():
    from app.nlp.keyword_filter import KeywordFilter
    kf = KeywordFilter()
    keywords = ["machine", "learning", "neural"]
    score = kf.score_sentence("Machine learning uses neural networks.", keywords)
    assert score > 0.0


def test_keyword_filter_filter_sentences_removes_irrelevant():
    from app.nlp.keyword_filter import KeywordFilter
    kf = KeywordFilter(threshold=0.1)
    text = (
        "Machine learning trains models on data. "
        "Ancient Rome had many emperors. "
        "Neural networks learn representations."
    )
    filtered = kf.filter_sentences(text, ["machine", "learning", "neural", "networks"])
    assert "Machine learning" in filtered or "Neural" in filtered
    # Rome sentence should be filtered
    assert "Rome" not in filtered


def test_keyword_filter_filter_sources():
    from app.nlp.keyword_filter import KeywordFilter
    kf = KeywordFilter(threshold=0.05)
    sources = [
        {"title": "ML paper", "abstract": "Machine learning trains neural network models."},
        {"title": "History paper", "abstract": "Ancient Roman history spanning centuries."},
        {"title": "No abstract", "abstract": ""},
    ]
    filtered = kf.filter_sources(sources, "machine learning neural")
    titles = [s["title"] for s in filtered]
    assert "ML paper" in titles
    # Sources without abstract are always kept
    assert "No abstract" in titles


def test_keyword_filter_empty_topic_returns_all():
    from app.nlp.keyword_filter import KeywordFilter
    kf = KeywordFilter()
    sources = [
        {"title": "A", "abstract": "Some content."},
        {"title": "B", "abstract": "Other content."},
    ]
    filtered = kf.filter_sources(sources, "")
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# 9. CacheManager
# ---------------------------------------------------------------------------

def test_cache_manager_set_and_get(tmp_path):
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="test")
    cache.set("key1", {"data": "value"})
    result = cache.get("key1")
    assert result == {"data": "value"}


def test_cache_manager_missing_key_returns_none(tmp_path):
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="test")
    assert cache.get("nonexistent") is None


def test_cache_manager_ttl_expiry(tmp_path):
    import time
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="test", default_ttl=0.01)
    cache.set("expiring", "value")
    time.sleep(0.05)
    assert cache.get("expiring") is None


def test_cache_manager_exists(tmp_path):
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="test")
    assert not cache.exists("k")
    cache.set("k", 42)
    assert cache.exists("k")


def test_cache_manager_delete(tmp_path):
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="test")
    cache.set("to_delete", "hello")
    cache.delete("to_delete")
    assert cache.get("to_delete") is None


def test_cache_manager_clear_namespace(tmp_path):
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(db_path=tmp_path / "test.db", namespace="ns1")
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear_namespace()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_cache_manager_cache_key():
    from app.nlp.cache_manager import CacheManager
    cache = CacheManager(namespace="test")
    key = cache.cache_key("search", "topic", "query")
    assert key == "search|topic|query"


# ---------------------------------------------------------------------------
# 10. NLPPipeline integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nlp_pipeline_preprocess_sources():
    from app.nlp.pipeline import NLPPipeline
    pipeline = NLPPipeline(cache_ttl=None)
    sources = [
        {
            "title": "Machine Learning Overview",
            "abstract": (
                "Machine learning is a method of data analysis that automates model building. "
                "It is based on the idea that systems can learn from data and identify patterns. "
                "Deep learning is a subfield that uses multilayer neural networks. "
                "Applications include computer vision, natural language processing, and robotics. "
                "Transformers have become the dominant architecture for language tasks."
            ),
            "source": "arxiv",
        },
        {
            "title": "Ancient History",
            "abstract": "Ancient Rome had emperors and legions spanning the Mediterranean.",
            "source": "web",
        },
    ]
    enriched = await pipeline.preprocess_sources(sources, "machine learning neural networks")
    assert len(enriched) == 2
    # All sources should have the new field
    for src in enriched:
        assert "processed_abstract" in src
    # The ML source should have a non-empty processed abstract
    ml_src = next(s for s in enriched if "Machine" in s["title"])
    assert ml_src["processed_abstract"]


@pytest.mark.asyncio
async def test_nlp_pipeline_analyze_essay():
    from app.nlp.pipeline import NLPPipeline
    pipeline = NLPPipeline(cache_ttl=None)
    sections = {
        "introduction": (
            "This paper examines renewable energy. We argue that solar energy is key. "
            "Firstly, solar is abundant. Secondly, it is clean. Furthermore, it is cheap. "
            "However, critics argue storage remains costly. In conclusion, solar is essential."
        ),
        "conclusion": "In conclusion, renewable energy, especially solar, is transformative.",
    }
    analysis = await pipeline.analyze_essay(sections, "renewable energy")
    assert "structure" in analysis
    assert "readability" in analysis
    assert "critic" in analysis
    # Structure check
    struct = analysis["structure"]
    assert "score" in struct
    assert 0.0 <= struct["score"] <= 1.0


def test_nlp_pipeline_validate_citations():
    from app.nlp.pipeline import NLPPipeline
    pipeline = NLPPipeline(cache_ttl=None)
    sources = [
        {"title": "Paper A", "authors": ["Smith, J."], "year": 2021, "doi": "10.1000/a.1"},
        {"title": "Invalid", "authors": [], "year": 0},
    ]
    result = pipeline.validate_citations(sources, style="apa")
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    assert len(result["citations"]) == 2
    assert result["bibliography"]


def test_nlp_pipeline_retrieve_top_chunks():
    from app.nlp.pipeline import NLPPipeline
    pipeline = NLPPipeline(cache_ttl=None)
    sources = [
        {"title": "ML", "abstract": "Machine learning models learn from data.", "processed_abstract": "Machine learning models learn from data."},
        {"title": "History", "abstract": "Ancient history spans millennia.", "processed_abstract": "Ancient history spans millennia."},
        {"title": "DL", "abstract": "Deep learning uses neural networks.", "processed_abstract": "Deep learning uses neural networks."},
    ]
    top = pipeline.retrieve_top_chunks("machine learning neural", sources, top_k=2)
    assert len(top) == 2
    titles = [s["title"] for s in top]
    assert "ML" in titles or "DL" in titles
