import re

from app.agents.base import AgentBase
from app.core.config import settings


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

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        content = input_data.get("content", "")
        evidence_pack = input_data.get("evidence_pack", [])
        revision_attempt = int(input_data.get("revision_attempt", 0))

        result = self._heuristic_grounding(content, evidence_pack, revision_attempt)
        await self._update_agent_state(db, project_id, "completed", result)
        return result