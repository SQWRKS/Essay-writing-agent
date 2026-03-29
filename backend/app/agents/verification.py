import json
import logging
import re
import time
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion

logger = logging.getLogger(__name__)


class VerificationAgent(AgentBase):
    name = "verification"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        sources = input_data.get("sources", [])

        verified = []
        rejected = []

        for source in sources:
            issues = []
            if not source.get("title"):
                issues.append("missing title")
            if not source.get("authors"):
                issues.append("missing authors")
            if not source.get("year"):
                issues.append("missing year")
            doi = source.get("doi", "")
            if doi and not re.match(r"^10\.\d{4,}/", doi):
                issues.append("invalid DOI format")
            if not source.get("abstract"):
                issues.append("missing abstract")

            start = time.monotonic()
            score = 1.0 - (len(issues) * 0.2)
            duration = (time.monotonic() - start) * 1000
            await self._log_api_call(db, "/verification/check", "POST", self.name, duration, 200, {"issues": issues})

            if score >= 0.6:
                verified.append({**source, "verification_score": score, "issues": issues})
            else:
                rejected.append({**source, "verification_score": score, "issues": issues})

        # Use LLM to assess credibility of verified sources when available
        if is_llm_available() and verified:
            verified = await self._llm_assess_credibility(verified, project_id, db)

        avg_score = sum(s["verification_score"] for s in verified) / len(verified) if verified else 0.0
        result = {
            "verified_sources": verified,
            "rejected_sources": rejected,
            "verification_score": round(avg_score, 3),
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    async def _llm_assess_credibility(self, sources: list, project_id: str, db) -> list:
        """Use LLM to assess and annotate source credibility."""
        try:
            sources_text = json.dumps(
                [
                    {
                        "title": s.get("title"),
                        "authors": s.get("authors"),
                        "year": s.get("year"),
                        "abstract": s.get("abstract", "")[:200],
                        "doi": s.get("doi"),
                    }
                    for s in sources[:10]
                ]
            )
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
                max_tokens=1024,
            )
            assessments = json.loads(content)
            if not isinstance(assessments, list):
                logger.warning("LLM credibility assessment returned non-list; skipping.")
                return sources
            if len(assessments) != len(sources):
                logger.warning(
                    "LLM credibility assessment count mismatch: expected %d, got %d; applying available assessments.",
                    len(sources),
                    len(assessments),
                )
            for source, assessment in zip(sources, assessments):
                boost = float(assessment.get("credibility_boost", 0.0))
                source["verification_score"] = round(
                    min(1.0, max(0.0, source["verification_score"] + boost)), 3
                )
                source["credibility_notes"] = assessment.get("credibility_notes", "")
        except Exception:
            pass
        return sources
