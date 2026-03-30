from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine_kwargs = {"echo": False, "pool_pre_ping": True}
if settings.DATABASE_URL.startswith("sqlite+aiosqlite"):
    engine_kwargs["poolclass"] = NullPool
    # Give SQLite time to wait on write locks instead of immediately failing.
    engine_kwargs["connect_args"] = {"timeout": 60}

engine = create_async_engine(settings.DATABASE_URL, **engine_kwargs)


if settings.DATABASE_URL.startswith("sqlite+aiosqlite"):
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()


AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        from app import models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        # Safe migration: add settings_json to existing SQLite databases.
        # create_all only adds new tables, not new columns on existing ones.
        if settings.DATABASE_URL.startswith("sqlite"):
            from sqlalchemy import text
            try:
                await conn.execute(text("ALTER TABLE projects ADD COLUMN settings_json TEXT"))
            except Exception:
                pass  # Column already exists — no action needed.
