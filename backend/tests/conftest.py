import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import Project, Task, AgentState, Output, ApiLog  # noqa: F401

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# StaticPool forces all sessions to reuse a single in-memory connection so
# tables created by setup_test_db are visible to every subsequent session.
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

# A separate, isolated in-memory engine for background pipeline tasks fired by
# the run_pipeline HTTP endpoint.  It has its own tables, but the project rows
# created in `test_engine` don't exist here.  So `execute_project_pipeline`
# looks up the project, finds nothing, and returns immediately — without
# touching `test_engine`'s shared connection.
bg_test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
BgTestSessionLocal = async_sessionmaker(bg_test_engine, class_=AsyncSession, expire_on_commit=False)

import app.database as _app_db  # noqa: E402
_app_db._bg_session_factory = BgTestSessionLocal


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    # Create tables in both the main test engine and the background-task engine.
    for eng in (test_engine, bg_test_engine):
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    for eng in (test_engine, bg_test_engine):
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(autouse=True)
async def cancel_leaked_tasks():
    """Cancel asyncio tasks that a test spawned but didn't await (e.g.
    pipeline fire-and-forget tasks).  Prevents them from bleeding into the
    next test's event loop and corrupting the shared StaticPool connection.
    """
    pre_test = set(asyncio.all_tasks())
    yield
    leaked = asyncio.all_tasks() - pre_test - {asyncio.current_task()}
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)


@pytest_asyncio.fixture
async def db_session():
    async with TestSessionLocal() as session:
        yield session


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
