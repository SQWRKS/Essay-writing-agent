import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging_config import logger
from app.database import init_db, AsyncSessionLocal


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Essay Writing Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = (time.monotonic() - start) * 1000

    try:
        async with AsyncSessionLocal() as db:
            from app.models import ApiLog
            from datetime import datetime, timezone
            log = ApiLog(
                endpoint=str(request.url.path),
                method=request.method,
                agent_name=None,
                duration_ms=duration,
                status_code=response.status_code,
                timestamp=datetime.now(timezone.utc),
            )
            db.add(log)
            await db.commit()
    except Exception:
        pass

    return response


from app.routers import projects, system, events  # noqa: E402

app.include_router(projects.router)
app.include_router(system.router)
app.include_router(events.router)

# Mount static files for figures
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BACKEND_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
