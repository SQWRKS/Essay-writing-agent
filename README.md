# Essay Writing Agent

An AI-powered, multi-agent academic writing system that autonomously generates university-level final-year engineering projects — dissertations, technical reports, literature reviews, and more.

## Overview

The system orchestrates a pipeline of specialised agents that plan, research, write, review, and cite a full academic document from a single topic input. It exposes everything through a REST + SSE API and a modern React dashboard.

```
User Input: "In-hub motors in Formula Student vehicles"
       ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│   Planner    │→  │  Researcher  │→  │   Verification   │
│              │   │ (arXiv / SS) │   │  (DOI / metadata)│
└──────────────┘   └──────────────┘   └──────────────────┘
                                               ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│   Citation   │←  │   Reviewer   │←  │     Writer       │
│  Manager     │   │ (QA + loops) │   │  (per section)   │
└──────────────┘   └──────────────┘   └──────────────────┘
       ↓
┌──────────────┐
│   Figures    │   → TXT / PDF / DOCX export
└──────────────┘
```

---

## Quick Start

### Option 1 — Single command (development)

```bash
# 1. Copy and configure environment variables
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY (optional — template fallbacks work without it)

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Install Node dependencies (first time only)
cd frontend && npm install && cd ..

# 4. Start everything
python run_app.py
```

`run_app.py` waits for both services to become reachable before opening the browser.
If a healthy backend or frontend is already running on the configured port, it reuses that service instead of failing on an address-in-use error.

Keep the terminal open while using the app. Press `Ctrl+C` only when you want to stop the services started by the launcher.

Frontend: http://localhost:3000  
Backend API health: http://localhost:8000/api/health

### Option 2 — Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

Frontend → http://localhost:3000  
Backend API → http://localhost:8000

---

## Architecture

```
essay-writing-agent/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, CORS, middleware
│   │   ├── database.py           # Async SQLAlchemy + aiosqlite
│   │   ├── models.py             # ORM models
│   │   ├── schemas.py            # Pydantic v2 schemas
│   │   ├── core/
│   │   │   ├── config.py         # Settings (pydantic-settings)
│   │   │   ├── logging_config.py # Structured logging
│   │   │   └── sse.py            # SSE manager (asyncio queues)
│   │   ├── agents/               # 7 independent agents
│   │   │   ├── base.py
│   │   │   ├── planner.py
│   │   │   ├── research.py
│   │   │   ├── verification.py
│   │   │   ├── writer.py
│   │   │   ├── reviewer.py
│   │   │   ├── citation.py
│   │   │   └── figure.py
│   │   ├── orchestration/
│   │   │   ├── task_graph.py     # DAG topological sort
│   │   │   └── worker_pool.py    # Async pipeline execution
│   │   ├── routers/
│   │   │   ├── projects.py       # /projects endpoints
│   │   │   ├── events.py         # SSE /projects/{id}/events
│   │   │   └── system.py         # /api/health, /api/logs, /api/config
│   │   └── export/
│   │       ├── txt_exporter.py
│   │       ├── pdf_exporter.py   # reportlab
│   │       └── docx_exporter.py  # python-docx
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_endpoints.py
│   │   └── test_agents.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/client.js         # Axios instance + API functions
│   │   ├── hooks/                # useProjects, useSSE, useConfig
│   │   ├── components/           # Layout, Sidebar, StatusBadge, …
│   │   └── pages/                # Dashboard, ProjectView, …
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── run_app.py                    # Single-command startup
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | OpenAI key — optional; template fallbacks work without it |
| `LLM_MODEL` | `gpt-4o-mini` | Model name passed to the OpenAI API |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per completion |
| `RESEARCH_SOURCES` | `arxiv,semantic_scholar,web` | Enabled research backends |
| `DATABASE_URL` | `sqlite+aiosqlite:///./essay_agent.db` | SQLAlchemy async DB URL |
| `BACKEND_PORT` | `8000` | Backend server port |
| `FRONTEND_PORT` | `3000` | Frontend dev-server port |
| `CORS_ORIGINS` | `http://localhost:3000,…` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `BROWSER_OPEN_DELAY` | `3` | Delay before opening the browser after both services are ready |
| `STARTUP_TIMEOUT` | `60` | Max time to wait for backend and frontend readiness |
| `HEALTH_CHECK_INTERVAL` | `5` | Interval for launcher process health checks |
| `SHUTDOWN_TIMEOUT` | `15` | Time to wait for graceful shutdown before force-killing child processes |

---

## API Reference

### Projects

| Method | Path | Description |
|---|---|---|
| `POST` | `/projects` | Create a new project |
| `GET` | `/projects` | List all projects |
| `GET` | `/projects/{id}` | Get project with agent states & tasks |
| `POST` | `/projects/{id}/run` | Start full pipeline (async) |
| `POST` | `/projects/{id}/run-agent` | Run a single agent |
| `GET` | `/projects/{id}/tasks` | List all tasks for a project |
| `GET` | `/projects/{id}/export?format=txt\|pdf\|docx` | Download export |
| `GET` | `/projects/{id}/events` | SSE real-time event stream |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/logs` | Paginated API log entries |
| `GET` | `/api/config` | Current runtime configuration |
| `POST` | `/api/config` | Update runtime configuration |

---

## Multi-Agent Pipeline

| Agent | Responsibility |
|---|---|
| **Planner** | Breaks topic into sections, generates research queries |
| **Research** | Queries arXiv / Semantic Scholar, returns source metadata |
| **Verification** | Validates DOIs, metadata completeness, credibility score |
| **Writer** | Produces academic text per section (LLM or template) |
| **Reviewer** | Scores writing quality, returns feedback loops |
| **Citation** | Formats Harvard / IEEE references, bibliography |
| **Figure** | Generates matplotlib charts, saves PNG assets |

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

The backend test suite currently contains 18 passing tests, including regression coverage for the writer fallback path.

---

## Frontend Pages

| Route | Page | Description |
|---|---|---|
| `/` | Dashboard | Create projects, view history |
| `/projects/:id` | Project View | Pipeline status, task list, SSE live updates |
| `/projects/:id/agents` | Agent Monitor | Filterable task table with JSON output viewer |
| `/projects/:id/editor` | Document Editor | Section editor, citations, re-run writer |
| `/projects/:id/export` | Export Panel | TXT / PDF / DOCX download |
| `/api-monitor` | API Monitor | Live log table, summary stats, pagination |

---

## Persistence

SQLite database tables: `projects`, `tasks`, `agent_states`, `outputs`, `api_logs`.  
Data persists across restarts. Change `DATABASE_URL` in `.env` for PostgreSQL in production.
