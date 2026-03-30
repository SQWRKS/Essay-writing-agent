import json
import re

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion


class ThesisAgent(AgentBase):
    name = "thesis"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")
        summaries = input_data.get("research_summaries", [])

        if is_llm_available() and summaries:
            result = await self._llm_generate(topic, summaries, project_id, db)
        else:
            result = self._heuristic_generate(topic, summaries)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _heuristic_generate(self, topic: str, summaries: list[dict]) -> dict:
        strongest = summaries[:3]
        claims = []
        keywords = []
        for summary in strongest:
            finding = summary.get("key_findings", "")
            if finding:
                claims.append(finding)
            source = summary.get("source", {})
            keywords.extend(
                token
                for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", source.get("title", "").lower())
                if len(token) > 4
            )

        distinct_keywords = []
        for keyword in keywords:
            if keyword not in distinct_keywords:
                distinct_keywords.append(keyword)

        thesis = f"For {topic}, the strongest evidence suggests that {self._compress_claims(claims)}"
        if distinct_keywords:
            thesis += f", particularly when evaluated through {', '.join(distinct_keywords[:3])}."
        else:
            thesis += "."

        return {
            "thesis": thesis,
            "supporting_claims": claims[:3],
        }

    def _compress_claims(self, claims: list[str]) -> str:
        if not claims:
            return "robust, evidence-led design trade-offs determine the most credible conclusion"

        cleaned = [re.sub(r"\s+", " ", claim).strip().rstrip(".") for claim in claims if claim]
        if len(cleaned) == 1:
            return cleaned[0][0].lower() + cleaned[0][1:]
        return "; ".join(claim[0].lower() + claim[1:] for claim in cleaned[:2])

    async def _llm_generate(self, topic: str, summaries: list[dict], project_id: str, db) -> dict:
        try:
            prompt = (
                f"You are formulating the core thesis for a final-year academic paper on '{topic}'. "
                "Using only the structured research summaries, return JSON with keys 'thesis' and 'supporting_claims'. "
                "The thesis must be arguable, precise, technically grounded, and defensible with the evidence. "
                "Avoid generic framing. Supporting claims must be short bullet-style statements extracted from the evidence.\n\n"
                f"Structured research summaries:\n{json.dumps(summaries[:8])}"
            )
            response = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=600,
            )
            payload = json.loads(response)
            if isinstance(payload, dict) and payload.get("thesis"):
                payload.setdefault("supporting_claims", [])
                return payload
        except Exception:
            pass
        return self._heuristic_generate(topic, summaries)