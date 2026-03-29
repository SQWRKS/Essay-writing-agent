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
    assert len(result["queries"]) >= 5
    assert len(result["summaries"]) == len(result["sources"])
    assert {"source", "key_findings", "quantitative_data", "relevance_score"}.issubset(result["summaries"][0].keys())


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


@pytest.mark.asyncio
async def test_writer_agent(db, test_project):
    from app.agents.writer import WriterAgent
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


def test_writer_grounded_fallback_is_validated():
    from app.agents.writer import WriterAgent
    agent = WriterAgent()

    payload = agent._normalize_inputs(
        {
            "section": "introduction",
            "topic": "sustainable agriculture",
            "thesis": "Sustainable agriculture becomes more resilient when water management and soil carbon interventions are deployed together.",
            "research_notes": [
                "[1] Soil Carbon and Yield Stability: Longitudinal field data show that farms with higher soil carbon had lower inter-annual yield volatility under drought conditions.",
                "[2] Drip Irrigation Meta-analysis: A meta-analysis across 40 studies reported improved water productivity in drip systems relative to flood irrigation.",
                "[3] Irrigation Scheduling Trials: Sensor-guided irrigation reduced peak seasonal water demand by 18% without depressing output.",
            ],
            "word_count": 220,
        }
    )
    content = agent.clean_output(agent._grounded_write(payload))
    validation = agent.validate_output(content, payload)

    assert "[1]" in content
    assert "[2]" in content
    assert "this paper examines" not in content.lower()
    assert validation["valid"] is True
    assert validation["repeated_phrase_ratio"] < 0.08


@pytest.mark.asyncio
async def test_thesis_agent(db, test_project):
    from app.agents.thesis import ThesisAgent
    agent = ThesisAgent()
    result = await agent.execute(
        {
            "topic": "battery thermal management",
            "research_summaries": [
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
        test_project.id,
        db,
    )
    assert "thesis" in result
    assert len(result["thesis"].split()) > 8
    assert "supporting_claims" in result


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
    assert result["approved"] is False
    assert result["metrics"]["generic_phrase_count"] >= 1


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
