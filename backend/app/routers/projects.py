import asyncio
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Project, Task, AgentState
from app.schemas import ProjectCreate, ProjectRead, ProjectUpdate, TaskRead, RunAgentRequest, ContentUpdate
from app.orchestration.worker_pool import WorkerPool
from app.core.sse import sse_manager

router = APIRouter(prefix="/projects", tags=["projects"])

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_DIR = os.path.join(BACKEND_DIR, "exports")


async def _load_project_or_404(project_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("", response_model=ProjectRead, status_code=201)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)):
    settings_str: Optional[str] = None
    if payload.settings is not None:
        settings_str = payload.settings.model_dump_json(exclude_none=True)

    project = Project(
        id=str(uuid.uuid4()),
        title=payload.title,
        topic=payload.topic,
        status="pending",
        settings_json=settings_str,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=list[ProjectRead])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    return result.scalars().all()


@router.get("/{project_id}", response_model=dict)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    agent_result = await db.execute(select(AgentState).where(AgentState.project_id == project_id))
    agent_states = agent_result.scalars().all()

    task_result = await db.execute(select(Task).where(Task.project_id == project_id))
    tasks = task_result.scalars().all()

    return {
        "id": project.id,
        "title": project.title,
        "topic": project.topic,
        "status": project.status,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "content": project.content,
        "settings_json": project.settings_json,
        "agent_states": [
            {"id": s.id, "agent_name": s.agent_name, "status": s.status, "updated_at": s.updated_at}
            for s in agent_states
        ],
        "tasks": [
            {
                "id": t.id,
                "agent_name": t.agent_name,
                "status": t.status,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
            }
            for t in tasks
        ],
    }


@router.put("/{project_id}/content")
async def update_project_content(
    project_id: str,
    payload: ContentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Persist manually edited document content back to the database."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.content = payload.model_dump_json(exclude_none=False)
    project.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(project)
    return {"id": project.id, "updated_at": project.updated_at}


@router.post("/{project_id}/run-agent")
async def run_agent(project_id: str, payload: RunAgentRequest, db: AsyncSession = Depends(get_db)):
    from app.agents import AGENT_REGISTRY

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    agent_cls = AGENT_REGISTRY.get(payload.agent_name)
    if not agent_cls:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {payload.agent_name}")

    task = Task(
        id=str(uuid.uuid4()),
        project_id=project_id,
        agent_name=payload.agent_name,
        status="running",
        input_data=json.dumps(payload.input_data),
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
    )
    db.add(task)
    await db.flush()

    try:
        agent = agent_cls()
        output = await agent.execute(payload.input_data, project_id, db)
        task.status = "completed"
        task.output_data = json.dumps(output)
        task.completed_at = datetime.now(timezone.utc)
        await db.commit()
        return {"task_id": task.id, "status": "completed", "output": output}
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.completed_at = datetime.now(timezone.utc)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/tasks", response_model=list[TaskRead])
async def get_project_tasks(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Task).where(Task.project_id == project_id).order_by(Task.created_at))
    return result.scalars().all()


@router.get("/{project_id}/export")
async def export_project(
    project_id: str,
    export_format: str = Query("txt", pattern="^(txt|pdf|docx)$", alias="format"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # Allow export for any non-pending project so users can retrieve partial results
    if project.status == "pending":
        raise HTTPException(
            status_code=400,
            detail="Project pipeline has not been started yet. Run the pipeline first.",
        )

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if export_format == "txt":
        from app.export.txt_exporter import export_project_txt
        file_path = export_project_txt(project, EXPORT_DIR)
        media_type = "text/plain"
    elif export_format == "pdf":
        from app.export.pdf_exporter import export_project_pdf
        file_path = export_project_pdf(project, EXPORT_DIR)
        media_type = "application/pdf"
    else:
        from app.export.docx_exporter import export_project_docx
        file_path = export_project_docx(project, EXPORT_DIR)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    return FileResponse(path=file_path, media_type=media_type, filename=os.path.basename(file_path))


@router.post("/{project_id}/upload-context", status_code=200)
async def upload_context_file(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a context file (.txt, .docx, or .pdf) for a project.

    The extracted text is merged into the project's ``settings_json`` under
    the ``context_text`` key.  Any text previously stored there is preserved
    (new content is appended with a blank-line separator).
    """
    project = await _load_project_or_404(project_id, db)

    content_bytes = await file.read()
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".txt":
            extracted = content_bytes.decode("utf-8", errors="replace")
        elif ext == ".docx":
            extracted = _extract_docx_text(content_bytes)
        elif ext == ".pdf":
            extracted = _extract_pdf_text(content_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Accepted formats: .txt, .docx, .pdf",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to extract text from file: {exc}") from exc

    # Merge into existing settings (preserve all other fields)
    existing: dict = {}
    if project.settings_json:
        try:
            existing = json.loads(project.settings_json)
        except Exception:
            pass

    prior_ctx = (existing.get("context_text") or "").strip()
    existing["context_text"] = (prior_ctx + "\n\n" + extracted.strip()).strip() if prior_ctx else extracted.strip()

    project.settings_json = json.dumps(existing)
    project.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "message": "Context file uploaded and text extracted successfully.",
        "characters_extracted": len(extracted),
        "project_id": project_id,
    }


def _extract_docx_text(content_bytes: bytes) -> str:
    """Extract plain text from a DOCX byte payload."""
    try:
        from docx import Document  # python-docx is in requirements.txt
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for DOCX extraction. Install it with: pip install python-docx"
        ) from exc
    doc = Document(io.BytesIO(content_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_pdf_text(content_bytes: bytes) -> str:
    """Extract plain text from a PDF byte payload using pypdf.

    pypdf is listed in requirements.txt.  If it is missing (e.g. in a minimal
    test environment without PDF support), a clear RuntimeError is raised.
    """
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF extraction. Install it with: pip install pypdf"
        ) from exc
    reader = pypdf.PdfReader(io.BytesIO(content_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


@router.post("/{project_id}/run", status_code=202)
async def run_pipeline(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await _load_project_or_404(project_id, db)
    if project.status == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")

    project.status = "running"
    project.updated_at = datetime.now(timezone.utc)
    await db.commit()

    topic = project.topic
    project_settings: dict = {}
    if project.settings_json:
        try:
            project_settings = json.loads(project.settings_json)
        except Exception:
            pass

    async def run_bg():
        import app.database as _app_db
        factory = _app_db._bg_session_factory or _app_db.AsyncSessionLocal
        async with factory() as session:
            pool = WorkerPool()
            await pool.execute_project_pipeline(project_id, topic, session, project_settings=project_settings)

    asyncio.create_task(run_bg())
    return {"message": "Pipeline started", "project_id": project_id}


@router.post("/{project_id}/pause", status_code=202)
async def pause_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await _load_project_or_404(project_id, db)

    if project.status == "completed":
        raise HTTPException(status_code=409, detail="Completed projects cannot be paused")
    if project.status == "failed":
        raise HTTPException(status_code=409, detail="Failed projects cannot be paused")
    if project.status == "paused":
        return {"message": "Project already paused", "project_id": project_id}

    project.status = "paused"
    project.updated_at = datetime.now(timezone.utc)
    await db.commit()

    await sse_manager.publish(project_id, "pipeline_pause_requested", {"project_id": project_id})
    return {"message": "Pause requested", "project_id": project_id}


@router.delete("/{project_id}")
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await _load_project_or_404(project_id, db)

    if project.status == "running":
        raise HTTPException(status_code=409, detail="Pause the project before deleting it")

    await db.delete(project)
    await db.commit()

    await sse_manager.publish(project_id, "project_deleted", {"project_id": project_id})
    return {"message": "Project deleted", "project_id": project_id}
