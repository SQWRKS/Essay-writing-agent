import re
import time
from app.agents.base import AgentBase


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

        avg_score = sum(s["verification_score"] for s in verified) / len(verified) if verified else 0.0
        result = {
            "verified_sources": verified,
            "rejected_sources": rejected,
            "verification_score": round(avg_score, 3),
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result
