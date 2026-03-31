import json
import logging
import re
import time
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.routing.model_config import AGENT_MODELS

logger = logging.getLogger(__name__)


class VerificationAgent(AgentBase):
    name = "verification"

    def _score_source(self, source: dict) -> tuple[float, list[str], list[str]]:
        issues = []
        strengths = []

        title = (source.get("title") or "").strip()
        authors = source.get("authors") or []
        year = source.get("year")
        doi = (source.get("doi") or "").strip()
        abstract = (source.get("abstract") or "").strip()
        venue = (source.get("venue") or "").strip()
        source_name = source.get("source") or "unknown"

        score = 0.0

        if title:
            score += 0.18
            if len(title) > 20:
                strengths.append("specific title")
        else:
            issues.append("missing title")

        if authors:
            score += 0.14
            if len(authors) >= 2:
                strengths.append("multiple named authors")
        else:
            issues.append("missing authors")

        if isinstance(year, int) and 1900 <= year <= (time.gmtime().tm_year + 1):
            score += 0.12
            if year >= time.gmtime().tm_year - 7:
                strengths.append("recent publication")
        else:
            issues.append("missing or invalid year")

        if doi:
            if re.match(r"^10\.\d{4,}/", doi):
                score += 0.2
                strengths.append("valid DOI")
            else:
                issues.append("invalid DOI format")
        elif source_name in {"semantic_scholar", "web"}:
            issues.append("missing DOI")

        if abstract:
            abstract_score = min(0.2, len(abstract) / 1200)
            score += abstract_score
            if len(abstract) >= 120:
                strengths.append("descriptive abstract")
        else:
            issues.append("missing abstract")

        if venue:
            score += 0.05
            strengths.append("venue metadata available")

        source_bonus = {
            "semantic_scholar": 0.08,
            "web": 0.05,
            "arxiv": 0.03,
        }.get(source_name, 0.0)
        score += source_bonus

        return round(min(1.0, score), 3), issues, strengths[:5]

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        sources = input_data.get("sources", [])

        verified = []
        rejected = []

        for source in sources:
            score, issues, strengths = self._score_source(source)
            start = time.monotonic()
            duration = (time.monotonic() - start) * 1000
            await self._log_api_call(
                db,
                "/verification/check",
                "POST",
                self.name,
                duration,
                200,
                {"issues": issues, "score": score},
            )

            enriched = {
                **source,
                "verification_score": score,
                "issues": issues,
                "strengths": strengths,
                "credibility_label": "high" if score >= 0.8 else "medium" if score >= 0.65 else "low",
            }

            if score >= 0.65:
                verified.append(enriched)
            else:
                rejected.append(enriched)

        # Use LLM to assess credibility only when there are enough sources to
        # justify the API call (≥ 5 sources).  For smaller sets the heuristic
        # verification_score is already a reliable signal.
        if is_llm_available() and len(verified) >= 5:
            verified = await self._llm_assess_credibility(verified, project_id, db)

        verified.sort(
            key=lambda source: (
                float(source.get("verification_score", 0.0)),
                float(source.get("relevance_score", 0.0)),
            ),
            reverse=True,
        )

        for source in verified:
            source["combined_quality_score"] = round(
                (float(source.get("relevance_score", 0.0)) * 0.55)
                + (float(source.get("verification_score", 0.0)) * 0.45),
                3,
            )

        avg_score = sum(s["verification_score"] for s in verified) / len(verified) if verified else 0.0
        result = {
            "verified_sources": verified,
            "rejected_sources": rejected,
            "verification_score": round(avg_score, 3),
            "verification_summary": {
                "verified_count": len(verified),
                "rejected_count": len(rejected),
                "high_confidence_count": sum(1 for s in verified if s.get("verification_score", 0.0) >= 0.8),
            },
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    async def _llm_assess_credibility(self, sources: list, project_id: str, db) -> list:
        """Use LLM to assess and annotate source credibility.

        Two-pass strategy:
        1. **Flagging pass** (gpt-5-mini / cheap model) — quickly identifies
           sources that may have credibility concerns.
        2. **Deep assessment pass** (gpt-5 / expensive model) — only applied to
           sources flagged as potentially problematic, keeping cost low.

        Limited to the top 8 sources and truncated abstracts (150 chars) to keep
        the prompts compact.
        """
        if not sources:
            return sources

        top_sources = sources[:8]
        sources_text = json.dumps(
            [
                {
                    "title": s.get("title"),
                    "authors": s.get("authors"),
                    "year": s.get("year"),
                    "abstract": (s.get("abstract", "") or "")[:150],
                    "doi": s.get("doi"),
                }
                for s in top_sources
            ]
        )

        # ------------------------------------------------------------------
        # Pass 1: cheap flagging pass (gpt-5-mini)
        # ------------------------------------------------------------------
        flagged_indices: list[int] = []
        try:
            flag_prompt = (
                "You are a quick-check academic librarian. "
                "For each source in the JSON list below, output a JSON array where each element has: "
                "index (int, 0-based), flag (bool — true if the source has any credibility concern "
                "such as missing DOI, anonymous authors, very old year, or suspicious title).\n\n"
                f"Sources:\n{sources_text}"
            )
            flag_content = await timed_chat_completion(
                flag_prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["verification"]["cheap"],
                max_tokens=256,
            )
            flags = json.loads(flag_content)
            if isinstance(flags, list):
                flagged_indices = [
                    int(item["index"])
                    for item in flags
                    if isinstance(item, dict) and item.get("flag") and "index" in item
                ]
        except Exception as exc:
            logger.warning("Verification flagging pass failed (%s); proceeding without flags.", exc)

        # ------------------------------------------------------------------
        # Pass 2: deep assessment (gpt-5) for flagged sources only
        # ------------------------------------------------------------------
        try:
            if flagged_indices:
                flagged_sources = [top_sources[i] for i in flagged_indices if i < len(top_sources)]
                deep_text = json.dumps(
                    [
                        {
                            "title": s.get("title"),
                            "authors": s.get("authors"),
                            "year": s.get("year"),
                            "abstract": (s.get("abstract", "") or "")[:150],
                            "doi": s.get("doi"),
                        }
                        for s in flagged_sources
                    ]
                )
                deep_prompt = (
                    "You are an academic librarian evaluating research sources that have been flagged for review. "
                    "For each source in the JSON list below, assess its credibility and return a JSON array "
                    "where each element has: credibility_notes (string, max 50 words), credibility_boost (float -0.1 to 0.1).\n\n"
                    f"Sources:\n{deep_text}"
                )
                deep_content = await timed_chat_completion(
                    deep_prompt,
                    db=db,
                    agent_name=self.name,
                    log_api_call_fn=self._log_api_call,
                    model=AGENT_MODELS["verification"]["expensive"],
                    max_tokens=1024,
                )
                assessments = json.loads(deep_content)
                if isinstance(assessments, list):
                    for flagged_source, assessment in zip(flagged_sources, assessments):
                        boost = float(assessment.get("credibility_boost", 0.0))
                        flagged_source["verification_score"] = round(
                            min(1.0, max(0.0, flagged_source["verification_score"] + boost)), 3
                        )
                        flagged_source["credibility_notes"] = assessment.get("credibility_notes", "")
                        flagged_source["credibility_label"] = (
                            "high"
                            if flagged_source["verification_score"] >= 0.8
                            else "medium"
                            if flagged_source["verification_score"] >= 0.65
                            else "low"
                        )
            else:
                # No flagged sources — run a lightweight bulk assessment with cheap model
                prompt = (
                    "You are an academic librarian evaluating research sources. "
                    "For each source in the JSON list below, assess its credibility and return a JSON array "
                    "where each element has: credibility_notes (string, max 50 words), credibility_boost (float -0.1 to 0.1).\n\n"
                    f"Sources:\n{sources_text}"
                )
                content = await timed_chat_completion(
                    prompt,
                    db=db,
                    agent_name=self.name,
                    log_api_call_fn=self._log_api_call,
                    model=AGENT_MODELS["verification"]["cheap"],
                    max_tokens=1024,
                )
                assessments = json.loads(content)
                if not isinstance(assessments, list):
                    logger.warning("LLM credibility assessment returned non-list; skipping.")
                    return sources
                if len(assessments) != len(top_sources):
                    logger.warning(
                        "LLM credibility assessment count mismatch: expected %d, got %d; applying available assessments.",
                        len(top_sources),
                        len(assessments),
                    )
                for source, assessment in zip(top_sources, assessments):
                    boost = float(assessment.get("credibility_boost", 0.0))
                    source["verification_score"] = round(
                        min(1.0, max(0.0, source["verification_score"] + boost)), 3
                    )
                    source["credibility_notes"] = assessment.get("credibility_notes", "")
                    source["credibility_label"] = (
                        "high" if source["verification_score"] >= 0.8 else "medium" if source["verification_score"] >= 0.65 else "low"
                    )
        except Exception:
            pass
        return sources
