import json
import logging
import re

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.core.config import settings
from app.routing.model_config import AGENT_MODELS

logger = logging.getLogger(__name__)

CLAIM_HINTS = {
    "shows", "suggests", "demonstrates", "indicates", "improves", "increases",
    "reduces", "supports", "found", "reported", "evidence", "because", "therefore",
}


class GroundingAgent(AgentBase):
    name = "grounding"

    def _split_sentences(self, content: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", content or "") if part.strip()]

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", (text or "").lower())
            if len(token) > 3
        }

    def _sentence_citations(self, sentence: str) -> list[str]:
        return re.findall(r"\[(\d+)\]", sentence or "")

    def _evidence_terms(self, evidence_pack: list[dict]) -> set[str]:
        terms = set()
        for evidence in evidence_pack[:8]:
            terms.update(self._tokenize(evidence.get("title", "")))
            terms.update(self._tokenize(evidence.get("abstract_excerpt", "")))
        return terms

    def _is_claim_like(self, sentence: str) -> bool:
        tokens = self._tokenize(sentence)
        if len(tokens) < 6:
            return False
        normalized = sentence.lower()
        return any(hint in normalized for hint in CLAIM_HINTS) or bool(re.search(r"\d", sentence))

    def _heuristic_grounding(self, content: str, evidence_pack: list[dict], revision_attempt: int = 0) -> dict:
        sentences = self._split_sentences(content)
        evidence_terms = self._evidence_terms(evidence_pack)
        claim_sentences = []
        supported_claims = []
        unsupported_claims = []
        weak_claims = []

        for sentence in sentences:
            if not self._is_claim_like(sentence):
                continue

            citations = self._sentence_citations(sentence)
            sentence_terms = self._tokenize(sentence)
            overlap = len(sentence_terms & evidence_terms)
            claim_record = {
                "text": sentence[:280],
                "citations": citations,
                "evidence_overlap": overlap,
            }
            claim_sentences.append(claim_record)

            if citations and overlap >= 1:
                supported_claims.append(claim_record)
            elif citations or overlap >= 2:
                weak_claims.append(claim_record)
            else:
                unsupported_claims.append(claim_record)

        citation_coverage = len(supported_claims) / max(1, len(claim_sentences)) if claim_sentences else 0.6
        overlap_score = (
            sum(item["evidence_overlap"] for item in claim_sentences) / max(1, len(claim_sentences) * 3)
            if claim_sentences else 0.6
        )
        unsupported_penalty = min(0.5, len(unsupported_claims) * 0.18)
        weak_penalty = min(0.2, len(weak_claims) * 0.08)
        score = max(0.0, min(1.0, 0.35 + (citation_coverage * 0.4) + (overlap_score * 0.25) - unsupported_penalty - weak_penalty))

        strengths = []
        if supported_claims:
            strengths.append("Claims are linked to cited evidence.")
        if citation_coverage >= 0.75:
            strengths.append("Most claim-like sentences are grounded in the evidence pack.")

        issues = []
        suggestions = []
        if unsupported_claims:
            issues.append("Some claim-like sentences lack both citations and evidence alignment.")
            suggestions.append("Revise unsupported claims so they cite or clearly reflect the retrieved evidence.")
        if weak_claims:
            issues.append("Several claims are only weakly supported by the current evidence pack.")
            suggestions.append("Strengthen weakly supported claims with clearer citations or more evidence-specific wording.")
        if not claim_sentences:
            issues.append("The section makes few concrete, checkable claims.")
            suggestions.append("Add more concrete, evidence-backed claims instead of general discussion.")

        approved = score >= settings.GROUNDING_MIN_SCORE and not unsupported_claims
        return {
            "score": round(score, 2),
            "approved": approved,
            "feedback": "Grounding check complete.",
            "strengths": strengths[:4],
            "issues": issues[:4],
            "suggestions": suggestions[:4],
            "claim_count": len(claim_sentences),
            "supported_claim_count": len(supported_claims),
            "unsupported_claim_count": len(unsupported_claims),
            "weak_claim_count": len(weak_claims),
            "unsupported_claims": unsupported_claims[:4],
            "weak_claims": weak_claims[:4],
            "revision_attempt": revision_attempt,
        }

    async def _llm_ground(
        self,
        content: str,
        evidence_pack: list[dict],
        heuristic: dict,
        project_id: str,
        db,
    ) -> dict:
        """LLM grounding pass: validate claim→evidence alignment and merge scores.

        Falls back to *heuristic* if the LLM is unavailable or returns malformed
        output.
        """
        try:
            evidence_summary = [
                {
                    "title": e.get("title", ""),
                    "excerpt": (e.get("abstract_excerpt") or "")[:200],
                }
                for e in evidence_pack[:6]
            ]
            prompt = (
                "You are a fact-grounding evaluator for academic writing.\n\n"
                "Evaluate the section content below against the evidence pack. "
                "Return JSON with keys:\n"
                "  score (0.0–1.0), approved (bool), issues (list of strings), "
                "suggestions (list of strings), unsupported_claim_count (int).\n\n"
                f"Evidence pack:\n{json.dumps(evidence_summary)}\n\n"
                f"Section content (first 1200 chars):\n{content[:1200]}"
            )
            response = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["grounding"]["default"],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=400,
            )
            payload = json.loads(response)
            if not isinstance(payload, dict) or "score" not in payload:
                return heuristic

            llm_score = float(payload.get("score", heuristic["score"]))
            # Blend: 60 % heuristic (rule-based precision) + 40 % LLM
            blended_score = round(0.6 * heuristic["score"] + 0.4 * llm_score, 3)
            merged = {
                **heuristic,
                "score": blended_score,
                "approved": blended_score >= settings.GROUNDING_MIN_SCORE
                    and heuristic.get("unsupported_claim_count", 0) == 0,
                "issues": list(dict.fromkeys(
                    heuristic.get("issues", []) + (payload.get("issues") or [])
                ))[:6],
                "suggestions": list(dict.fromkeys(
                    heuristic.get("suggestions", []) + (payload.get("suggestions") or [])
                ))[:6],
            }
            if "unsupported_claim_count" in payload:
                merged["unsupported_claim_count"] = max(
                    heuristic.get("unsupported_claim_count", 0),
                    int(payload["unsupported_claim_count"]),
                )
            return merged
        except Exception as exc:
            logger.debug("GroundingAgent LLM pass failed (non-fatal): %s", exc)
            return heuristic

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        content = input_data.get("content", "")
        evidence_pack = input_data.get("evidence_pack", [])
        revision_attempt = int(input_data.get("revision_attempt", 0))

        result = self._heuristic_grounding(content, evidence_pack, revision_attempt)

        if is_llm_available() and evidence_pack:
            result = await self._llm_ground(content, evidence_pack, result, project_id, db)

        await self._update_agent_state(db, project_id, "completed", result)
        return result