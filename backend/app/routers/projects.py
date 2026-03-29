import json
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Project, Task, AgentState
from app.schemas import ProjectCreate, ProjectRead, ProjectUpdate, TaskRead, RunAgentRequest
from app.orchestration.worker_pool import WorkerPool

router = APIRouter(prefix="/projects", tags=["projects"])

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_DIR = os.path.join(BACKEND_DIR, "exports")


@router.post("", response_model=ProjectRead, status_code=201)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)):
    project = Project(
        id=str(uuid.uuid4()),
        title=payload.title,
        topic=payload.topic,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
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
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
    )
    db.add(task)
    await db.flush()

    try:
        agent = agent_cls()
        output = await agent.execute(payload.input_data, project_id, db)
        task.status = "completed"
        task.output_data = json.dumps(output)
        task.completed_at = datetime.utcnow()
        await db.commit()
        return {"task_id": task.id, "status": "completed", "output": output}
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.completed_at = datetime.utcnow()
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
    if project.status != "completed":
        raise HTTPException(status_code=400, detail="Project is not yet completed")

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


@router.post("/{project_id}/run", status_code=202)
async def run_pipeline(project_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")

    topic = project.topic

    async def run_bg():
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            pool = WorkerPool()
            await pool.execute_project_pipeline(project_id, topic, session)

    background_tasks.add_task(run_bg)
    return {"message": "Pipeline started", "project_id": project_id}
