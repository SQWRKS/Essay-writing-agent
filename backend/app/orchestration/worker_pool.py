import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.orchestration.task_graph import TaskGraph
from app.core.sse import sse_manager
from app.core.config import settings
from app.nlp.pipeline import NLPPipeline

logger = logging.getLogger(__name__)

# Shared NLP pipeline instance (one per process; all components are stateless
# between calls and the CacheManager connection is thread-safe)
_nlp_pipeline = NLPPipeline()

# Maximum number of queries forwarded to the WebSearchAgent per pipeline run
_WS_QUERY_LIMIT = 4


class PipelinePausedError(RuntimeError):
    """Raised when a user requests pause while a pipeline is running."""


class WorkerPool:
    def _build_revision_feedback(self, review_result: dict, grounding_result: dict, revision_attempt: int) -> dict:
        """Build structured revision guidance for the writer."""
        return {
            "revision_attempt": revision_attempt,
            "current_score": float(review_result.get("score", 0.0)),
            "review_feedback": review_result.get("feedback", ""),
            "blocking_issues": review_result.get("blocking_issues", [])[:5],
            "suggestions": review_result.get("suggestions", [])[:6],
            "weak_categories": [
                name
                for name, value in (review_result.get("category_scores", {}) or {}).items()
                if float(value) < 0.8
            ],
            "grounding_issues": grounding_result.get("issues", [])[:5],
            "unsupported_claim_count": int(grounding_result.get("unsupported_claim_count", 0)),
        }

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

    async def execute_project_pipeline(
        self,
        project_id: str,
        topic: str,
        db: AsyncSession,
        project_settings: dict | None = None,
    ):
        """Run the full multi-agent pipeline for *project_id*.

        Parameters
        ----------
        project_id:
            UUID of the project record.
        topic:
            Essay topic string.
        db:
            Active async SQLAlchemy session.
        project_settings:
            Optional fine-tune settings dict (from ``Project.settings_json``).
            All keys are optional; omitting them preserves the original
            pipeline defaults.  Recognised keys:

            * ``word_count_target`` (int) — target total word count; section
              targets are scaled proportionally.
            * ``writing_style`` (str) — style / tone hint for the writer.
            * ``context_text`` (str) — additional background context prepended
              to topic for the planner and research queries.
            * ``rubric`` (str) — marking rubric used by the reviewer.
        """
        from app.models import Project, Task
        from app.agents import AGENT_REGISTRY

        # ---- Unpack fine-tune settings (all optional, all have safe defaults) ----
        ps: dict = project_settings or {}
        ft_word_count: int | None = ps.get("word_count_target")  # None → use per-section defaults
        ft_writing_style: str = (ps.get("writing_style") or "").strip()
        ft_context_text: str = (ps.get("context_text") or "").strip()
        ft_rubric: str = (ps.get("rubric") or "").strip()

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
                "quality": {
                    "sections": {},
                    "summary": {
                        "approved_sections": 0,
                        "flagged_sections": 0,
                        "average_score": 0.0,
                    },
                },
            },
        }

        try:
            # --- PLANNER ---
            planner_input: dict = {"topic": topic}
            if ft_word_count:
                planner_input["word_count_target"] = ft_word_count
            if ft_writing_style:
                planner_input["writing_style"] = ft_writing_style
            if ft_context_text:
                planner_input["context_text"] = ft_context_text
            planner_task = await self._create_task(db, project_id, "planner", planner_input, [])
            graph.add_task(planner_task.id, "planner")
            await db.commit()

            plan_result = await self._run_task(planner_task, db, project_id, planner_input)
            graph.mark_completed(planner_task.id)
            plan = plan_result or {}

            # --- RESEARCH + WEB SEARCH (parallel) ---
            queries = plan.get("research_queries", [f"{topic} overview", f"{topic} methods"])
            research_input = {"queries": queries[:6], "sources": settings.RESEARCH_SOURCES, "topic": topic}
            research_task = await self._create_task(db, project_id, "research", research_input, [planner_task.id])
            graph.add_task(research_task.id, "research", [planner_task.id])

            # Create the web_search task record before the gather so it appears
            # in the task list immediately, even while it's still pending.
            ws_input = {"queries": queries[:_WS_QUERY_LIMIT], "topic": topic}
            ws_task = None
            if settings.WEB_SEARCH_ENABLED:
                ws_task = await self._create_task(db, project_id, "web_search", ws_input, [planner_task.id])
                graph.add_task(ws_task.id, "web_search", [planner_task.id])

            await db.commit()

            # Build coroutines: research always runs; web_search only when enabled
            if settings.WEB_SEARCH_ENABLED and ws_task is not None:
                # Research uses the main session; web_search uses its own session
                # so the two coroutines never share session state.
                research_outcome, ws_outcome = await asyncio.gather(
                    self._run_task(research_task, db, project_id, research_input),
                    self._run_agent_in_fresh_session(ws_task.id, "web_search", ws_input, project_id),
                    return_exceptions=True,
                )
                # Research failure is fatal; web_search failure is non-fatal
                if isinstance(research_outcome, BaseException):
                    raise research_outcome
                research_result = research_outcome
                graph.mark_completed(research_task.id)

                if isinstance(ws_outcome, BaseException):
                    logger.warning("WebSearchAgent failed (non-fatal): %s", ws_outcome)
                    web_search_result: dict = {}
                else:
                    web_search_result = ws_outcome
                    graph.mark_completed(ws_task.id)
            else:
                research_result = await self._run_task(research_task, db, project_id, research_input)
                graph.mark_completed(research_task.id)
                web_search_result = {}

            # Merge web search sources into the academic source pool
            all_sources = list(research_result.get("sources", []))
            if web_search_result:
                ws_sources = web_search_result.get("sources", [])
                existing_keys = {
                    (s.get("doi") or "").rstrip("/") or (s.get("url") or "").rstrip("/") or s.get("title", "")
                    for s in all_sources
                }
                added = 0
                for ws in ws_sources:
                    key = (ws.get("url") or "").rstrip("/") or ws.get("title", "")
                    if key and key not in existing_keys:
                        all_sources.append(ws)
                        existing_keys.add(key)
                        added += 1
                if added:
                    logger.info("WebSearchAgent added %d new sources to the research pool", added)

            # Build merged source breakdown for metadata
            merged_breakdown = dict(research_result.get("source_breakdown", {}))
            ws_count = len(web_search_result.get("sources", [])) if web_search_result else 0
            if ws_count:
                merged_breakdown["web_search"] = ws_count

            # --- NLP PREPROCESSING (non-LLM, parallel) ---
            # Cleans, summarises and keyword-filters each source abstract
            # before passing to the LLM writer.  Runs in a thread-pool so
            # it does not block the async event loop.
            try:
                all_sources = await _nlp_pipeline.preprocess_sources(
                    all_sources, topic, queries=queries
                )
            except Exception as _nlp_err:
                logger.warning("NLP source preprocessing failed (non-fatal): %s", _nlp_err)

            content["metadata"]["research"] = {
                "queries": queries,
                "source_breakdown": merged_breakdown,
                "total_found": len(all_sources),
                "summary": research_result.get("summary", ""),
                "top_sources": [
                    {
                        "title": source.get("title", "Unknown"),
                        "source": source.get("source", "unknown"),
                        "year": source.get("year"),
                        "relevance_score": source.get("relevance_score", 0.0),
                        "match_reasons": source.get("match_reasons", []),
                    }
                    for source in all_sources[:5]
                ],
            }

            # --- VERIFICATION ---
            verify_input = {"sources": all_sources}
            verify_task = await self._create_task(db, project_id, "verification", verify_input, [research_task.id])
            graph.add_task(verify_task.id, "verification", [research_task.id])
            await db.commit()

            verify_result = await self._run_task(verify_task, db, project_id, verify_input)
            graph.mark_completed(verify_task.id)
            verified_sources = verify_result.get("verified_sources", research_result.get("sources", []))
            verified_sources = sorted(
                verified_sources,
                key=lambda source: float(source.get("combined_quality_score") or source.get("relevance_score") or 0.0),
                reverse=True,
            )
            content["metadata"]["sources"] = verified_sources
            content["metadata"]["research"]["verification_summary"] = verify_result.get("verification_summary", {})

            # --- THESIS ---
            thesis_input = {
                "topic": topic,
                "research_summaries": research_result.get("summaries", []),
            }
            thesis_task = await self._create_task(db, project_id, "thesis", thesis_input, [verify_task.id])
            graph.add_task(thesis_task.id, "thesis", [verify_task.id])
            await db.commit()

            thesis_result = await self._run_task(thesis_task, db, project_id, thesis_input)
            graph.mark_completed(thesis_task.id)
            thesis_statement = thesis_result.get("thesis", "")
            content["metadata"]["thesis"] = thesis_statement

            # --- WRITER + REVIEWER per section ---
            sections = plan.get("sections", [])
            sections = [section for section in sections if section.get("include", True)]
            if not sections:
                sections = [{"key": "introduction", "title": "Introduction", "word_count_target": 500}]
            sections_by_key = {item.get("key", "introduction"): item for item in sections}

            writer_task_ids = []
            section_quality_entries = []
            for section_info in sections:
                sec_key = section_info.get("key", "introduction")
                section_evidence = self._build_section_evidence(section_info, verified_sources)
                evidence_count = len(section_evidence)
                section_word_target = int(section_info.get("word_count_target", 500) or 500)
                section_max_attempts = int(settings.MAX_REVISION_ATTEMPTS)

                # Results sections can burn tokens when evidence is weak; cap retries and target length.
                if sec_key == "results" and evidence_count < 3:
                    section_max_attempts = min(section_max_attempts, 1)
                    section_word_target = min(section_word_target, 450)

                writer_input = {
                    "section": sec_key,
                    "topic": topic,
                    "word_count": section_word_target,
                    "section_plan": section_info,
                    "research_data": {
                        "sources": verified_sources,
                        "summaries": research_result.get("summaries", []),
                        "thesis": thesis_statement,
                        "section_queries": section_info.get("research_queries", []),
                        "evidence_pack": section_evidence,
                        "research_summary": research_result.get("summary", ""),
                    },
                }
                # Inject optional fine-tune settings (no-op when empty)
                if ft_writing_style:
                    writer_input["writing_style"] = ft_writing_style
                writer_task = await self._create_task(db, project_id, "writer", writer_input, [verify_task.id])
                graph.add_task(writer_task.id, "writer", [verify_task.id])
                await db.commit()
                writer_task_ids.append(writer_task.id)

                write_result = await self._run_task(writer_task, db, project_id, writer_input)
                graph.mark_completed(writer_task.id)
                written_content = write_result.get("content", "")
                revision_attempts_used = 0
                section_target_score = max(settings.REVIEW_MIN_SCORE, settings.SECTION_SCORE_TARGET)
                section_started_at = time.monotonic()
                stop_reason = "initial_review"

                grounding_input = {
                    "section": sec_key,
                    "content": written_content,
                    "evidence_pack": section_evidence,
                    "revision_attempt": revision_attempts_used,
                }
                grounding_task = await self._create_task(db, project_id, "grounding", grounding_input, [writer_task.id])
                graph.add_task(grounding_task.id, "grounding", [writer_task.id])
                await db.commit()
                grounding_result = await self._run_task(grounding_task, db, project_id, grounding_input)
                graph.mark_completed(grounding_task.id)

                # Reviewer
                review_input = {
                    "section": sec_key,
                    "content": written_content,
                    "expected_word_count": section_info.get("word_count_target", 500),
                    "evidence_pack": section_evidence,
                    "grounding_summary": grounding_result,
                    "revision_attempt": revision_attempts_used,
                }
                if ft_rubric:
                    review_input["rubric"] = ft_rubric
                reviewer_task = await self._create_task(db, project_id, "reviewer", review_input, [writer_task.id])
                graph.add_task(reviewer_task.id, "reviewer", [writer_task.id])
                await db.commit()

                review_result = await self._run_task(reviewer_task, db, project_id, review_input)
                graph.mark_completed(reviewer_task.id)

                current_score = float(review_result.get("score", 0.0))
                approved = bool(review_result.get("approved", False)) and current_score >= section_target_score
                if approved:
                    stop_reason = "target_score_reached"
                best_snapshot = {
                    "content": written_content,
                    "review": review_result,
                    "grounding": grounding_result,
                    "score": current_score,
                    "subheadings": write_result.get("subheadings", []),
                }
                plateau_rounds = 0
                low_grounding_rounds = 1 if float(grounding_result.get("score", 0.0)) < 0.55 else 0

                while not approved:
                    if revision_attempts_used >= section_max_attempts:
                        stop_reason = "max_revision_attempts"
                        break

                    elapsed_minutes = (time.monotonic() - section_started_at) / 60.0
                    if elapsed_minutes >= settings.MAX_SECTION_REVISION_MINUTES:
                        stop_reason = "max_revision_minutes"
                        break

                    revision_attempts_used += 1
                    structured_feedback = self._build_revision_feedback(review_result, grounding_result, revision_attempts_used)
                    revised_input = {
                        **writer_input,
                        "revision_attempt": revision_attempts_used,
                        "feedback": json.dumps(structured_feedback),
                    }
                    writer_task2 = await self._create_task(db, project_id, "writer", revised_input, [reviewer_task.id])
                    graph.add_task(writer_task2.id, "writer", [reviewer_task.id])
                    await db.commit()
                    writer_task_ids.append(writer_task2.id)
                    write_result = await self._run_task(writer_task2, db, project_id, revised_input)
                    graph.mark_completed(writer_task2.id)
                    revised_content = write_result.get("content", written_content)

                    reground_input = {
                        "section": sec_key,
                        "content": revised_content,
                        "evidence_pack": section_evidence,
                        "revision_attempt": revision_attempts_used,
                    }
                    reground_task = await self._create_task(db, project_id, "grounding", reground_input, [writer_task2.id])
                    graph.add_task(reground_task.id, "grounding", [writer_task2.id])
                    await db.commit()
                    reground_result = await self._run_task(reground_task, db, project_id, reground_input)
                    graph.mark_completed(reground_task.id)

                    re_review_input = {
                        "section": sec_key,
                        "content": revised_content,
                        "expected_word_count": section_info.get("word_count_target", 500),
                        "evidence_pack": section_evidence,
                        "grounding_summary": reground_result,
                        "revision_attempt": revision_attempts_used,
                    }
                    if ft_rubric:
                        re_review_input["rubric"] = ft_rubric
                    re_review_task = await self._create_task(db, project_id, "reviewer", re_review_input, [writer_task2.id])
                    graph.add_task(re_review_task.id, "reviewer", [writer_task2.id])
                    await db.commit()
                    re_review_result = await self._run_task(re_review_task, db, project_id, re_review_input)
                    graph.mark_completed(re_review_task.id)

                    new_score = float(re_review_result.get("score", 0.0))
                    score_delta = new_score - current_score

                    written_content = revised_content
                    review_result = re_review_result
                    grounding_result = reground_result
                    current_score = new_score

                    if new_score > best_snapshot["score"]:
                        best_snapshot = {
                            "content": revised_content,
                            "review": re_review_result,
                            "grounding": reground_result,
                            "score": new_score,
                            "subheadings": write_result.get("subheadings", []),
                        }

                    if score_delta < settings.MIN_REVISION_DELTA:
                        plateau_rounds += 1
                    else:
                        plateau_rounds = 0

                    if float(reground_result.get("score", 0.0)) < 0.55:
                        low_grounding_rounds += 1
                    else:
                        low_grounding_rounds = 0

                    approved = bool(re_review_result.get("approved", False)) and new_score >= section_target_score
                    if approved:
                        stop_reason = "target_score_reached"
                        break
                    if plateau_rounds >= 2:
                        stop_reason = "score_plateau"
                        break
                    if low_grounding_rounds >= 2:
                        stop_reason = "grounding_persistent_low"
                        break

                if not approved and stop_reason == "initial_review":
                    stop_reason = "review_not_approved"

                written_content = best_snapshot["content"]
                review_result = best_snapshot["review"]
                grounding_result = best_snapshot["grounding"]

                content["sections"][sec_key] = written_content
                content["metadata"].setdefault("subheadings", {})[sec_key] = best_snapshot.get("subheadings", [])
                section_quality = {
                    "title": section_info.get("title", sec_key.title()),
                    "score": review_result.get("score", 0.0),
                    "approved": bool(review_result.get("approved", False)) and float(review_result.get("score", 0.0)) >= section_target_score,
                    "target_score": section_target_score,
                    "stop_reason": stop_reason,
                    "revision_attempts": revision_attempts_used,
                    "expected_word_count": section_word_target,
                    "evidence_count": evidence_count,
                    "actual_word_count": len(written_content.split()),
                    "citation_count": review_result.get("citation_count", 0),
                    "grounding_score": grounding_result.get("score", 0.0),
                    "grounding_issues": grounding_result.get("issues", []),
                    "unsupported_claim_count": grounding_result.get("unsupported_claim_count", 0),
                    "category_scores": review_result.get("category_scores", {}),
                    "blocking_issues": review_result.get("blocking_issues", []),
                    "feedback": review_result.get("feedback", ""),
                    "suggestions": review_result.get("suggestions", []),
                    "strengths": review_result.get("strengths", []),
                }
                content["metadata"]["quality"]["sections"][sec_key] = section_quality
                section_quality_entries.append(section_quality)

            if section_quality_entries:
                approved_count = sum(1 for item in section_quality_entries if item.get("approved"))
                average_score = sum(float(item.get("score", 0.0)) for item in section_quality_entries) / len(section_quality_entries)
                content["metadata"]["quality"]["summary"] = {
                    "approved_sections": approved_count,
                    "flagged_sections": len(section_quality_entries) - approved_count,
                    "average_score": round(average_score, 3),
                }

            # --- COHERENCE ---
            coherence_input = {
                "topic": topic,
                "sections": content["sections"],
                "quality_sections": content["metadata"]["quality"]["sections"],
            }
            coherence_task = await self._create_task(db, project_id, "coherence", coherence_input, writer_task_ids)
            graph.add_task(coherence_task.id, "coherence", writer_task_ids)
            await db.commit()

            coherence_result = await self._run_task(coherence_task, db, project_id, coherence_input)
            graph.mark_completed(coherence_task.id)

            coherence_target = max(settings.COHERENCE_MIN_SCORE, settings.COHERENCE_SCORE_TARGET)
            coherence_round = 0
            coherence_plateau_rounds = 0
            coherence_stop_reason = "initial_review"
            coherence_score = float(coherence_result.get("score", 0.0))
            coherence_approved = bool(coherence_result.get("approved", False)) and coherence_score >= coherence_target
            if coherence_approved:
                coherence_stop_reason = "target_score_reached"

            while not coherence_approved and coherence_round < settings.MAX_COHERENCE_REVISION_ROUNDS:
                coherence_round += 1

                flagged_keys = []
                flagged_keys.extend([str(key) for key in coherence_result.get("flagged_sections", [])])
                flagged_keys.extend([str(key) for key in coherence_result.get("repeated_opening_sections", [])])
                if not flagged_keys:
                    weakest = sorted(
                        content["metadata"]["quality"]["sections"].items(),
                        key=lambda item: float(item[1].get("score", 0.0)),
                    )
                    flagged_keys = [item[0] for item in weakest[:2]]

                deduped_keys = []
                for key in flagged_keys:
                    if key in content["sections"] and key not in deduped_keys:
                        deduped_keys.append(key)

                if not deduped_keys:
                    coherence_stop_reason = "no_rewritable_sections"
                    break

                for sec_key in deduped_keys:
                    section_text = content["sections"].get(sec_key, "")
                    section_info = sections_by_key.get(sec_key, {"key": sec_key, "title": sec_key.title(), "word_count_target": 500})
                    section_evidence = self._build_section_evidence(section_info, verified_sources)
                    section_quality = content["metadata"]["quality"]["sections"].get(sec_key, {})

                    coherence_feedback = {
                        "coherence_round": coherence_round,
                        "coherence_score": coherence_score,
                        "coherence_feedback": coherence_result.get("feedback", ""),
                        "coherence_issues": coherence_result.get("issues", []),
                        "coherence_suggestions": coherence_result.get("suggestions", []),
                        "section_opening": (coherence_result.get("section_summaries", {}).get(sec_key, {}) or {}).get("opening", ""),
                        "section_blocking_issues": section_quality.get("blocking_issues", []),
                    }
                    rewrite_input = {
                        "section": sec_key,
                        "topic": topic,
                        "word_count": section_info.get("word_count_target", 500),
                        "section_plan": section_info,
                        "feedback": json.dumps(coherence_feedback),
                        "research_data": {
                            "sources": verified_sources,
                            "section_queries": section_info.get("research_queries", []),
                            "evidence_pack": section_evidence,
                            "research_summary": research_result.get("summary", ""),
                        },
                    }
                    # Propagate writing_style through coherence rewrites (no-op when empty)
                    if ft_writing_style:
                        rewrite_input["writing_style"] = ft_writing_style

                    coherence_writer_task = await self._create_task(db, project_id, "writer", rewrite_input, [coherence_task.id])
                    graph.add_task(coherence_writer_task.id, "writer", [coherence_task.id])
                    await db.commit()
                    writer_task_ids.append(coherence_writer_task.id)
                    rewrite_result = await self._run_task(coherence_writer_task, db, project_id, rewrite_input)
                    graph.mark_completed(coherence_writer_task.id)
                    rewritten_content = rewrite_result.get("content", section_text)

                    reground_input = {
                        "section": sec_key,
                        "content": rewritten_content,
                        "evidence_pack": section_evidence,
                        "revision_attempt": int(section_quality.get("revision_attempts", 0)) + 1,
                    }
                    coherence_grounding_task = await self._create_task(db, project_id, "grounding", reground_input, [coherence_writer_task.id])
                    graph.add_task(coherence_grounding_task.id, "grounding", [coherence_writer_task.id])
                    await db.commit()
                    reground_result = await self._run_task(coherence_grounding_task, db, project_id, reground_input)
                    graph.mark_completed(coherence_grounding_task.id)

                    re_review_input = {
                        "section": sec_key,
                        "content": rewritten_content,
                        "expected_word_count": section_info.get("word_count_target", 500),
                        "evidence_pack": section_evidence,
                        "grounding_summary": reground_result,
                        "revision_attempt": int(section_quality.get("revision_attempts", 0)) + 1,
                    }
                    if ft_rubric:
                        re_review_input["rubric"] = ft_rubric
                    coherence_reviewer_task = await self._create_task(db, project_id, "reviewer", re_review_input, [coherence_writer_task.id])
                    graph.add_task(coherence_reviewer_task.id, "reviewer", [coherence_writer_task.id])
                    await db.commit()
                    re_review_result = await self._run_task(coherence_reviewer_task, db, project_id, re_review_input)
                    graph.mark_completed(coherence_reviewer_task.id)

                    content["sections"][sec_key] = rewritten_content
                    content["metadata"]["quality"]["sections"][sec_key] = {
                        **section_quality,
                        "score": re_review_result.get("score", section_quality.get("score", 0.0)),
                        "approved": bool(re_review_result.get("approved", False)) and float(re_review_result.get("score", 0.0)) >= max(settings.REVIEW_MIN_SCORE, settings.SECTION_SCORE_TARGET),
                        "revision_attempts": int(section_quality.get("revision_attempts", 0)) + 1,
                        "target_score": max(settings.REVIEW_MIN_SCORE, settings.SECTION_SCORE_TARGET),
                        "stop_reason": "coherence_rewrite_round",
                        "actual_word_count": len(rewritten_content.split()),
                        "citation_count": re_review_result.get("citation_count", 0),
                        "grounding_score": reground_result.get("score", 0.0),
                        "grounding_issues": reground_result.get("issues", []),
                        "unsupported_claim_count": reground_result.get("unsupported_claim_count", 0),
                        "category_scores": re_review_result.get("category_scores", {}),
                        "blocking_issues": re_review_result.get("blocking_issues", []),
                        "feedback": re_review_result.get("feedback", ""),
                        "suggestions": re_review_result.get("suggestions", []),
                        "strengths": re_review_result.get("strengths", []),
                    }

                updated_sections = content["metadata"]["quality"]["sections"]
                if updated_sections:
                    approved_count = sum(1 for item in updated_sections.values() if item.get("approved"))
                    average_score = sum(float(item.get("score", 0.0)) for item in updated_sections.values()) / len(updated_sections)
                    content["metadata"]["quality"]["summary"] = {
                        "approved_sections": approved_count,
                        "flagged_sections": len(updated_sections) - approved_count,
                        "average_score": round(average_score, 3),
                    }

                next_coherence_input = {
                    "topic": topic,
                    "sections": content["sections"],
                    "quality_sections": content["metadata"]["quality"]["sections"],
                }
                next_coherence_task = await self._create_task(db, project_id, "coherence", next_coherence_input, writer_task_ids)
                graph.add_task(next_coherence_task.id, "coherence", writer_task_ids)
                await db.commit()

                next_coherence_result = await self._run_task(next_coherence_task, db, project_id, next_coherence_input)
                graph.mark_completed(next_coherence_task.id)
                new_coherence_score = float(next_coherence_result.get("score", 0.0))
                score_delta = new_coherence_score - coherence_score
                coherence_score = new_coherence_score
                coherence_result = next_coherence_result
                coherence_task = next_coherence_task
                coherence_approved = bool(coherence_result.get("approved", False)) and coherence_score >= coherence_target

                if coherence_approved:
                    coherence_stop_reason = "target_score_reached"
                    break

                if score_delta < settings.MIN_REVISION_DELTA:
                    coherence_plateau_rounds += 1
                else:
                    coherence_plateau_rounds = 0

                if coherence_plateau_rounds >= 2:
                    coherence_stop_reason = "score_plateau"
                    break

            if not coherence_approved and coherence_stop_reason == "initial_review":
                if coherence_round >= settings.MAX_COHERENCE_REVISION_ROUNDS:
                    coherence_stop_reason = "max_coherence_rounds"
                else:
                    coherence_stop_reason = "coherence_not_approved"

            content["metadata"]["quality"]["coherence"] = coherence_result
            content["metadata"]["quality"]["summary"]["coherence_score"] = coherence_result.get("score", 0.0)
            content["metadata"]["quality"]["summary"]["coherence_approved"] = coherence_result.get("approved", False)
            content["metadata"]["quality"]["summary"]["coherence_target_score"] = coherence_target
            content["metadata"]["quality"]["summary"]["coherence_revision_rounds"] = coherence_round
            content["metadata"]["quality"]["summary"]["coherence_stop_reason"] = coherence_stop_reason

            # --- NLP ESSAY ANALYSIS (non-LLM) ---
            # Run after all sections are finalised.  Adds structure, readability,
            # and critic reports to metadata without making any LLM calls.
            try:
                nlp_analysis = await _nlp_pipeline.analyze_essay(content["sections"], topic)
                content["metadata"]["nlp_analysis"] = nlp_analysis
            except Exception as _nlp_err:
                logger.warning("NLP essay analysis failed (non-fatal): %s", _nlp_err)

            # --- CITATION ---
            citation_input = {"sources": verified_sources, "style": "harvard"}
            citation_task = await self._create_task(db, project_id, "citation", citation_input, [*writer_task_ids, coherence_task.id])
            graph.add_task(citation_task.id, "citation", [*writer_task_ids, coherence_task.id])
            await db.commit()

            citation_result = await self._run_task(citation_task, db, project_id, citation_input)
            graph.mark_completed(citation_task.id)
            content["metadata"]["citations"] = citation_result.get("formatted_citations", [])
            content["metadata"]["bibliography"] = citation_result.get("bibliography", "")

            # NLP citation validation (no LLM — validates fields + reformats)
            try:
                nlp_citations = _nlp_pipeline.validate_citations(verified_sources, style="harvard")
                content["metadata"].setdefault("nlp_analysis", {})["citations"] = nlp_citations
            except Exception as _nlp_err:
                logger.warning("NLP citation validation failed (non-fatal): %s", _nlp_err)

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

        except PipelinePausedError as e:
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.status = "paused"
                project.updated_at = datetime.now(timezone.utc)
                await db.commit()
            await sse_manager.publish(project_id, "pipeline_paused", {"project_id": project_id, "reason": str(e)})
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
        """Select top evidence items for a section using term overlap + source quality scores.

        Scoring combines:
        - Token overlap between section metadata/queries and source title+abstract
        - Source combined_quality_score or relevance_score
        - Recency bonus (sources ≤ 5 years old) for grounding freshness
        """
        if not sources:
            return []

        import time as _time
        current_year = _time.gmtime().tm_year

        terms = set()
        for field in [section_info.get("key", ""), section_info.get("title", ""), section_info.get("description", "")]:
            terms.update({tok.lower() for tok in str(field).split() if len(tok) > 3})
        for query in section_info.get("research_queries", [])[:4]:
            terms.update({tok.lower() for tok in str(query).split() if len(tok) > 3})
        for item in section_info.get("must_cover", [])[:3]:
            terms.update({tok.lower() for tok in str(item).split() if len(tok) > 3})

        scored = []
        for src in sources:
            title = (src.get("title") or "").lower()
            abstract = (src.get("abstract") or "").lower()
            blob = f"{title} {abstract}"
            overlap = sum(1 for term in terms if term in blob)
            base = float(src.get("combined_quality_score") or src.get("relevance_score") or 0.0)
            verification_score = float(src.get("verification_score") or 0.0)
            year = src.get("year") or 0
            recency = max(0.0, 1.0 - (max(0, current_year - int(year)) / 10.0)) if isinstance(year, int) and year > 1900 else 0.0
            score = base + (verification_score * 0.2) + min(1.0, overlap / 4.0) + (recency * 0.1)
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
                    "verification_score": round(float(src.get("verification_score") or 0.0), 3),
                    "abstract_excerpt": abstract[:320],
                    "match_reasons": src.get("match_reasons", []),
                }
            )
        return evidence

    async def _run_agent_in_fresh_session(
        self, task_id: str, agent_name: str, input_data: dict, project_id: str
    ) -> dict:
        """Run an agent in its own DB session.

        This allows two agents to execute concurrently with ``asyncio.gather``
        without sharing a SQLAlchemy session (which is not concurrency-safe).
        The method mirrors ``_run_task`` but opens a fresh
        ``AsyncSessionLocal`` session so no session state is shared with the
        caller's session.
        """
        from app.database import AsyncSessionLocal
        from app.agents import AGENT_REGISTRY
        from app.models import Task
        from sqlalchemy import select as sel

        async with AsyncSessionLocal() as session:
            # Mark task as running
            task_row = (await session.execute(sel(Task).where(Task.id == task_id))).scalar_one_or_none()
            if task_row:
                task_row.status = "running"
                task_row.started_at = datetime.now(timezone.utc)
            await session.commit()

            await sse_manager.publish(project_id, "task_update", {
                "task_id": task_id,
                "status": "running",
                "agent": agent_name,
            })

            agent_cls = AGENT_REGISTRY.get(agent_name)
            if not agent_cls:
                raise ValueError(f"Unknown agent: {agent_name}")

            agent = agent_cls()
            try:
                output = await agent.execute(input_data, project_id, session)

                task_row2 = (await session.execute(sel(Task).where(Task.id == task_id))).scalar_one_or_none()
                if task_row2:
                    task_row2.status = "completed"
                    task_row2.output_data = json.dumps(output)
                    task_row2.completed_at = datetime.now(timezone.utc)
                await session.commit()

                await sse_manager.publish(project_id, "task_update", {
                    "task_id": task_id,
                    "status": "completed",
                    "agent": agent_name,
                })
                return output
            except Exception as exc:
                task_row3 = (await session.execute(sel(Task).where(Task.id == task_id))).scalar_one_or_none()
                if task_row3:
                    task_row3.status = "failed"
                    task_row3.error = str(exc)
                    task_row3.completed_at = datetime.now(timezone.utc)
                await session.commit()

                await sse_manager.publish(project_id, "task_update", {
                    "task_id": task_id,
                    "status": "failed",
                    "agent": agent_name,
                    "error": str(exc),
                })
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
            created_at=datetime.now(timezone.utc),
        )
        db.add(task)
        await db.flush()
        return task

    async def _run_task(self, task, db, project_id: str, input_data: dict) -> dict:
        from app.models import Project, Task
        from app.agents import AGENT_REGISTRY
        from sqlalchemy import select as sel

        project_result = await db.execute(sel(Project).where(Project.id == project_id))
        project = project_result.scalar_one_or_none()
        if project is None:
            raise PipelinePausedError("Project no longer exists")
        if (project.status or "").lower() == "paused":
            result_cancel = await db.execute(sel(Task).where(Task.id == task.id))
            db_task_cancel = result_cancel.scalar_one_or_none()
            if db_task_cancel and db_task_cancel.status == "pending":
                db_task_cancel.status = "cancelled"
                db_task_cancel.error = "Pipeline paused by user"
                db_task_cancel.completed_at = datetime.now(timezone.utc)
                await db.commit()

                await sse_manager.publish(project_id, "task_update", {
                    "task_id": task.id,
                    "status": "cancelled",
                    "agent": task.agent_name,
                    "error": "Pipeline paused by user",
                })
            raise PipelinePausedError("Pause requested by user")

        result = await db.execute(sel(Task).where(Task.id == task.id))
        db_task = result.scalar_one_or_none()
        if db_task:
            db_task.status = "running"
            db_task.started_at = datetime.now(timezone.utc)
        await db.commit()

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
            await db.commit()

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
            await db.commit()
            await sse_manager.publish(project_id, "task_update", {
                "task_id": task.id,
                "status": "failed",
                "agent": task.agent_name,
                "error": str(e),
            })
            raise
