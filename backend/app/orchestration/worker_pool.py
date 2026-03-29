import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.orchestration.task_graph import TaskGraph
from app.core.sse import sse_manager
from app.core.config import settings


class WorkerPool:
    def _derive_figure_data(self, sources: list[dict], source_breakdown: dict | None = None) -> dict:
        source_breakdown = source_breakdown or {}
        if not source_breakdown:
            for src in sources:
                name = src.get("source", "unknown")
                source_breakdown[name] = source_breakdown.get(name, 0) + 1

        categories = list(source_breakdown.keys())[:6] or ["No Data"]
        values = [source_breakdown[c] for c in categories] if source_breakdown else [1]

        year_counts: dict[int, int] = {}
        for src in sources:
            year = src.get("year")
            if isinstance(year, int) and 1900 <= year <= 2100:
                year_counts[year] = year_counts.get(year, 0) + 1

        if year_counts:
            years = sorted(year_counts.keys())
            trend = [year_counts[y] for y in years]
        else:
            years = list(range(2019, 2025))
            trend = [1, 1, 2, 2, 3, 3]

        return {
            "categories": categories,
            "values": values,
            "years": years,
            "trend": trend,
            "trend_label": "Publications",
        }

    async def execute_project_pipeline(self, project_id: str, topic: str, db: AsyncSession):
        from app.models import Project, Task
        from app.agents import AGENT_REGISTRY

        # Update project status
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return
        project.status = "running"
        project.updated_at = datetime.now(timezone.utc)
        await db.commit()

        await sse_manager.publish(project_id, "pipeline_start", {"project_id": project_id, "topic": topic})

        graph = TaskGraph()
        content: dict = {
            "sections": {},
            "metadata": {
                "topic": topic,
                "sources": [],
                "citations": [],
                "research": {},
                "figures": [],
            },
        }

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

            content["metadata"]["research"] = {
                "queries": queries,
                "source_breakdown": research_result.get("source_breakdown", {}),
                "total_found": research_result.get("total_found", 0),
                "summary": research_result.get("summary", ""),
            }

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
                section_evidence = self._build_section_evidence(section_info, verified_sources)
                writer_input = {
                    "section": sec_key,
                    "topic": topic,
                    "word_count": section_info.get("word_count_target", 500),
                    "research_data": {
                        "sources": verified_sources,
                        "section_queries": section_info.get("research_queries", []),
                        "evidence_pack": section_evidence,
                        "research_summary": research_result.get("summary", ""),
                    },
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
                    # Revision loop: attempt up to settings.MAX_REVISION_ATTEMPTS revisions
                    for _attempt in range(settings.MAX_REVISION_ATTEMPTS):
                        revised_input = {**writer_input, "feedback": review_result.get("feedback", "")}
                        writer_task2 = await self._create_task(db, project_id, "writer", revised_input, [reviewer_task.id])
                        graph.add_task(writer_task2.id, "writer", [reviewer_task.id])
                        await db.commit()
                        write_result = await self._run_task(writer_task2, db, project_id, revised_input)
                        graph.mark_completed(writer_task2.id)
                        written_content = write_result.get("content", written_content)
                        # Check again after revision
                        re_review_input = {"section": sec_key, "content": written_content}
                        re_review_task = await self._create_task(db, project_id, "reviewer", re_review_input, [writer_task2.id])
                        graph.add_task(re_review_task.id, "reviewer", [writer_task2.id])
                        await db.commit()
                        review_result = await self._run_task(re_review_task, db, project_id, re_review_input)
                        graph.mark_completed(re_review_task.id)
                        if review_result.get("approved", True):
                            break

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
            figure_input = {
                "topic": topic,
                "section": "results",
                "data": self._derive_figure_data(
                    verified_sources,
                    content["metadata"]["research"].get("source_breakdown", {}),
                ),
            }
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
                project.updated_at = datetime.now(timezone.utc)
            await db.commit()

            await sse_manager.publish(project_id, "pipeline_complete", {"project_id": project_id, "status": "completed"})

        except Exception as e:
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.status = "failed"
                project.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await sse_manager.publish(project_id, "pipeline_error", {"project_id": project_id, "error": str(e)})
            raise

    def _build_section_evidence(self, section_info: dict, sources: list[dict]) -> list[dict]:
        """Select top evidence items for a section using simple term overlap + source relevance."""
        if not sources:
            return []

        terms = set()
        for field in [section_info.get("key", ""), section_info.get("title", ""), section_info.get("description", "")]:
            terms.update({tok.lower() for tok in str(field).split() if len(tok) > 3})
        for query in section_info.get("research_queries", [])[:4]:
            terms.update({tok.lower() for tok in str(query).split() if len(tok) > 3})

        scored = []
        for src in sources:
            title = (src.get("title") or "").lower()
            abstract = (src.get("abstract") or "").lower()
            blob = f"{title} {abstract}"
            overlap = sum(1 for term in terms if term in blob)
            base = float(src.get("relevance_score") or 0.0)
            score = base + min(1.0, overlap / 4.0)
            scored.append((score, src))

        scored.sort(key=lambda item: item[0], reverse=True)

        evidence = []
        for score, src in scored[:6]:
            abstract = src.get("abstract") or ""
            evidence.append(
                {
                    "title": src.get("title", "Unknown"),
                    "year": src.get("year"),
                    "source": src.get("source", "unknown"),
                    "doi": src.get("doi", ""),
                    "url": src.get("url", ""),
                    "relevance_score": round(score, 3),
                    "abstract_excerpt": abstract[:320],
                }
            )
        return evidence

    async def _create_task(self, db, project_id: str, agent_name: str, input_data: dict, dep_ids: list):
        from app.models import Task
        task = Task(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_name=agent_name,
            status="pending",
            input_data=json.dumps(input_data),
            dependencies=json.dumps(dep_ids),
            created_at=datetime.now(timezone.utc),
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
            db_task.started_at = datetime.now(timezone.utc)
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
                db_task2.completed_at = datetime.now(timezone.utc)
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
                db_task3.completed_at = datetime.now(timezone.utc)
            await db.flush()
            await sse_manager.publish(project_id, "task_update", {
                "task_id": task.id,
                "status": "failed",
                "agent": task.agent_name,
                "error": str(e),
            })
            raise
