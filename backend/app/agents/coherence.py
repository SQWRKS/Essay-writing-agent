import re

from app.agents.base import AgentBase
from app.core.config import settings


class CoherenceAgent(AgentBase):
    name = "coherence"

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", (text or "").lower())
            if len(token) > 4
        }

    def _first_sentence(self, text: str) -> str:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
        return parts[0][:220]

    def _heuristic_coherence(self, topic: str, sections: dict[str, str], quality_sections: dict | None = None) -> dict:
        quality_sections = quality_sections or {}
        ordered_items = [(key, value) for key, value in sections.items() if value]
        if not ordered_items:
            return {
                "score": 0.0,
                "approved": False,
                "feedback": "No section content available for coherence validation.",
                "issues": ["Essay content is empty."],
                "suggestions": ["Generate section content before running coherence validation."],
                "section_summaries": {},
            }

        issues = []
        suggestions = []
        strengths = []
        section_summaries = {}
        repeated_openings = []
        weak_sections = []
        topic_terms = self._tokenize(topic)
        aggregate_terms = set()
        opening_sentences = []

        for key, content in ordered_items:
            opening = self._first_sentence(content)
            opening_sentences.append((key, opening.lower()))
            terms = self._tokenize(content)
            aggregate_terms |= terms
            quality = quality_sections.get(key, {})
            section_summaries[key] = {
                "opening": opening,
                "word_count": len(content.split()),
                "score": quality.get("score"),
                "grounding_score": quality.get("grounding_score"),
            }
            if quality.get("approved") is False:
                weak_sections.append(key)

        seen_openings = {}
        for key, opening in opening_sentences:
            if not opening:
                continue
            marker = opening[:100]
            if marker in seen_openings:
                repeated_openings.extend([seen_openings[marker], key])
            else:
                seen_openings[marker] = key

        repeated_openings = list(dict.fromkeys(repeated_openings))
        if repeated_openings:
            issues.append("Multiple sections begin with near-identical opening language, which weakens the essay's flow.")
            suggestions.append("Rewrite repeated openings so each section advances the argument from a distinct angle.")

        topic_coverage = len(topic_terms & aggregate_terms) / max(1, len(topic_terms)) if topic_terms else 1.0
        if topic_coverage < 0.35:
            issues.append("The essay does not consistently stay anchored to the central topic across sections.")
            suggestions.append("Reinforce the main topic and thesis across sections instead of drifting into generic discussion.")
        else:
            strengths.append("Sections remain tied to the central topic.")

        if weak_sections:
            issues.append("One or more sections remain below the quality bar, which weakens whole-essay coherence.")
            suggestions.append("Revise flagged sections before treating the essay as a coherent final draft.")

        discussion = sections.get("discussion", "")
        conclusion = sections.get("conclusion", "")
        if discussion and conclusion:
            discussion_terms = self._tokenize(discussion)
            conclusion_terms = self._tokenize(conclusion)
            overlap = len(discussion_terms & conclusion_terms) / max(1, min(len(discussion_terms), len(conclusion_terms)))
            if overlap < 0.08:
                issues.append("The conclusion appears weakly connected to the discussion and may not synthesize the essay effectively.")
                suggestions.append("Revise the conclusion so it clearly reflects the major claims developed in the discussion.")
            else:
                strengths.append("The conclusion reflects themes established earlier in the essay.")

        introduction = sections.get("introduction", "")
        if introduction and not any(term in aggregate_terms for term in self._tokenize(introduction) if term in topic_terms):
            issues.append("The introduction does not clearly establish terms that are sustained across the essay.")
            suggestions.append("Make the introduction establish the vocabulary and framing used throughout later sections.")

        score = 0.55
        score += min(0.2, topic_coverage * 0.2)
        score -= min(0.18, len(repeated_openings) * 0.05)
        score -= min(0.15, len(weak_sections) * 0.06)
        score -= 0.08 if any("conclusion" in issue.lower() for issue in issues) else 0.0
        score = max(0.0, min(1.0, score))

        approved = score >= settings.COHERENCE_MIN_SCORE and not weak_sections and not repeated_openings
        feedback = "Essay-level coherence check complete."
        return {
            "score": round(score, 2),
            "approved": approved,
            "feedback": feedback,
            "issues": issues[:5],
            "suggestions": suggestions[:5],
            "strengths": strengths[:4],
            "repeated_opening_sections": repeated_openings,
            "flagged_sections": weak_sections,
            "topic_coverage": round(topic_coverage, 2),
            "section_summaries": section_summaries,
        }

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        result = self._heuristic_coherence(
            input_data.get("topic", ""),
            input_data.get("sections", {}),
            input_data.get("quality_sections", {}),
        )
        await self._update_agent_state(db, project_id, "completed", result)
        return result