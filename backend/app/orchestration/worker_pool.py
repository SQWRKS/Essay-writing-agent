import json
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.orchestration.task_graph import TaskGraph
from app.core.sse import sse_manager
from app.core.config import settings


class WorkerPool:
    async def execute_project_pipeline(self, project_id: str, topic: str, db: AsyncSession):
        from app.models import Project, Task
        from app.agents import AGENT_REGISTRY

        # Update project status
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return
        project.status = "running"
        project.updated_at = datetime.utcnow()
        await db.commit()

        await sse_manager.publish(project_id, "pipeline_start", {"project_id": project_id, "topic": topic})

        graph = TaskGraph()
        content: dict = {"sections": {}, "metadata": {"topic": topic, "sources": [], "citations": []}}

        try:
            # --- PLANNER ---
            planner_task = await self._create_task(db, project_id, "planner", {"topic": topic}, [])
            graph.add_task(planner_task.id, "planner")
            await db.commit()

            plan_result = await self._run_task(planner_task, db, project_id, {"topic": topic})
            graph.mark_completed(planner_task.id)
            plan = plan_result or {}

            # --- RESEARCH ---
            queries = plan.get("research_queries", [f"{topic} overview", f"{topic} methods"])
            research_input = {"queries": queries[:6], "sources": settings.RESEARCH_SOURCES}
            research_task = await self._create_task(db, project_id, "research", research_input, [planner_task.id])
            graph.add_task(research_task.id, "research", [planner_task.id])
            await db.commit()

            research_result = await self._run_task(research_task, db, project_id, research_input)
            graph.mark_completed(research_task.id)

            # --- VERIFICATION ---
            verify_input = {"sources": research_result.get("sources", [])}
            verify_task = await self._create_task(db, project_id, "verification", verify_input, [research_task.id])
            graph.add_task(verify_task.id, "verification", [research_task.id])
            await db.commit()

            verify_result = await self._run_task(verify_task, db, project_id, verify_input)
            graph.mark_completed(verify_task.id)
            verified_sources = verify_result.get("verified_sources", research_result.get("sources", []))
            content["metadata"]["sources"] = verified_sources

            # --- WRITER + REVIEWER per section ---
            sections = plan.get("sections", [])
            if not sections:
                sections = [{"key": "introduction", "title": "Introduction", "word_count_target": 500}]

            writer_task_ids = []
            for section_info in sections:
                sec_key = section_info.get("key", "introduction")
                writer_input = {
                    "section": sec_key,
                    "topic": topic,
                    "word_count": section_info.get("word_count_target", 500),
                    "research_data": {"sources": verified_sources},
                }
                writer_task = await self._create_task(db, project_id, "writer", writer_input, [verify_task.id])
                graph.add_task(writer_task.id, "writer", [verify_task.id])
                await db.commit()

                write_result = await self._run_task(writer_task, db, project_id, writer_input)
                graph.mark_completed(writer_task.id)
                written_content = write_result.get("content", "")

                # Reviewer
                review_input = {"section": sec_key, "content": written_content}
                reviewer_task = await self._create_task(db, project_id, "reviewer", review_input, [writer_task.id])
                graph.add_task(reviewer_task.id, "reviewer", [writer_task.id])
                await db.commit()

                review_result = await self._run_task(reviewer_task, db, project_id, review_input)
                graph.mark_completed(reviewer_task.id)

                if not review_result.get("approved", True):
                    # One revision attempt
                    revised_input = {**writer_input, "feedback": review_result.get("feedback", "")}
                    writer_task2 = await self._create_task(db, project_id, "writer", revised_input, [reviewer_task.id])
                    graph.add_task(writer_task2.id, "writer", [reviewer_task.id])
                    await db.commit()
                    write_result = await self._run_task(writer_task2, db, project_id, revised_input)
                    graph.mark_completed(writer_task2.id)
                    written_content = write_result.get("content", written_content)

                content["sections"][sec_key] = written_content
                writer_task_ids.append(writer_task.id)

            # --- CITATION ---
            citation_input = {"sources": verified_sources, "style": "harvard"}
            citation_task = await self._create_task(db, project_id, "citation", citation_input, writer_task_ids)
            graph.add_task(citation_task.id, "citation", writer_task_ids)
            await db.commit()

            citation_result = await self._run_task(citation_task, db, project_id, citation_input)
            graph.mark_completed(citation_task.id)
            content["metadata"]["citations"] = citation_result.get("formatted_citations", [])
            content["metadata"]["bibliography"] = citation_result.get("bibliography", "")

            # --- FIGURE ---
            figure_input = {"topic": topic, "section": "results", "data": {}}
            figure_task = await self._create_task(db, project_id, "figure", figure_input, [citation_task.id])
            graph.add_task(figure_task.id, "figure", [citation_task.id])
            await db.commit()

            figure_result = await self._run_task(figure_task, db, project_id, figure_input)
            graph.mark_completed(figure_task.id)
            content["metadata"]["figures"] = figure_result.get("figures", [])

            # Update project content and status
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.content = json.dumps(content)
                project.status = "completed"
                project.updated_at = datetime.utcnow()
            await db.commit()

            await sse_manager.publish(project_id, "pipeline_complete", {"project_id": project_id, "status": "completed"})

        except Exception as e:
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.status = "failed"
                project.updated_at = datetime.utcnow()
            await db.commit()
            await sse_manager.publish(project_id, "pipeline_error", {"project_id": project_id, "error": str(e)})
            raise

    async def _create_task(self, db, project_id: str, agent_name: str, input_data: dict, dep_ids: list):
        from app.models import Task
        task = Task(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_name=agent_name,
            status="pending",
            input_data=json.dumps(input_data),
            dependencies=json.dumps(dep_ids),
            created_at=datetime.utcnow(),
        )
        db.add(task)
        await db.flush()
        return task

    async def _run_task(self, task, db, project_id: str, input_data: dict) -> dict:
        from app.models import Task
        from app.agents import AGENT_REGISTRY
        from sqlalchemy import select as sel

        result = await db.execute(sel(Task).where(Task.id == task.id))
        db_task = result.scalar_one_or_none()
        if db_task:
            db_task.status = "running"
            db_task.started_at = datetime.utcnow()
        await db.flush()

        await sse_manager.publish(project_id, "task_update", {
            "task_id": task.id,
            "status": "running",
            "agent": task.agent_name,
        })

        try:
            agent_cls = AGENT_REGISTRY.get(task.agent_name)
            if not agent_cls:
                raise ValueError(f"Unknown agent: {task.agent_name}")
            agent = agent_cls()
            output = await agent.execute(input_data, project_id, db)

            result2 = await db.execute(sel(Task).where(Task.id == task.id))
            db_task2 = result2.scalar_one_or_none()
            if db_task2:
                db_task2.status = "completed"
                db_task2.output_data = json.dumps(output)
                db_task2.completed_at = datetime.utcnow()
            await db.flush()

            await sse_manager.publish(project_id, "task_update", {
                "task_id": task.id,
                "status": "completed",
                "agent": task.agent_name,
            })
            return output
        except Exception as e:
            result3 = await db.execute(sel(Task).where(Task.id == task.id))
            db_task3 = result3.scalar_one_or_none()
            if db_task3:
                db_task3.status = "failed"
                db_task3.error = str(e)
                db_task3.completed_at = datetime.utcnow()
            await db.flush()
            await sse_manager.publish(project_id, "task_update", {
                "task_id": task.id,
                "status": "failed",
                "agent": task.agent_name,
                "error": str(e),
            })
            raise
