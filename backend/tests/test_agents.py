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
    from datetime import datetime
    project = Project(
        id=str(uuid.uuid4()),
        title="Agent Test Project",
        topic="machine learning",
        status="running",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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
        {"queries": ["machine learning overview"], "sources": ["web"]},
        test_project.id, db
    )
    assert "sources" in result
    assert "total_found" in result
    assert isinstance(result["sources"], list)


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
        {"section": "introduction", "topic": "machine learning", "word_count": 200, "research_data": {}},
        test_project.id, db
    )
    assert "section" in result
    assert "content" in result
    assert "word_count" in result
    assert result["section"] == "introduction"
    assert len(result["content"]) > 0


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
