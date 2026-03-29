import json
import re

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion


GENERIC_PHRASES = [
    "this paper examines",
    "this study explores",
    "it is important to note",
    "in conclusion",
    "has gained significant attention",
    "plays a vital role",
    "in the modern world",
    "this section discusses",
]


class ReviewerAgent(AgentBase):
    name = "reviewer"
    approval_threshold = 0.72

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        section = input_data.get("section", "")
        content = input_data.get("content", "")
        thesis = input_data.get("thesis", "")
        research_notes = input_data.get("research_notes", [])

        heuristic = self._heuristic_review(section, content, thesis, research_notes)
        if is_llm_available():
            result = await self._llm_review(section, content, thesis, research_notes, heuristic, project_id, db)
        else:
            result = heuristic

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _heuristic_review(self, section: str, content: str, thesis: str, research_notes: list, threshold: float | None = None) -> dict:
        score = 1.0
        suggestions = []
        feedback = []
        lowered = content.lower()
        words = re.findall(r"\b\w+\b", lowered)

        repeated_ratio = self._repeated_phrase_ratio(content)
        generic_hits = [phrase for phrase in GENERIC_PHRASES if phrase in lowered]
        citation_count = len(re.findall(r"\[\d+\]", content))
        digit_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", content))
        long_words = [word for word in words if len(word) >= 9]
        domain_keywords = self._domain_keywords(thesis, research_notes)
        domain_hits = [keyword for keyword in domain_keywords if keyword in lowered]

        if len(words) < 120:
            score -= 0.18
            suggestions.append("Develop the argument into fuller paragraphs with stronger evidence.")
            feedback.append("The section is too short for final-year academic depth.")

        if repeated_ratio > 0.08:
            score -= min(0.28, repeated_ratio * 1.8)
            suggestions.append("Rewrite repeated sentences and vary clause structure.")
            feedback.append("The draft repeats phrasing too often.")

        if generic_hits:
            score -= min(0.2, len(generic_hits) * 0.05)
            suggestions.append("Replace generic academic filler with concrete technical claims.")
            feedback.append(f"Generic phrasing weakens specificity: {', '.join(generic_hits[:3])}.")

        if citation_count == 0:
            score -= 0.22
            suggestions.append("Ground factual claims with inline citations.")
            feedback.append("The section makes uncited claims.")
        elif citation_count >= 2:
            score += 0.04

        if digit_count == 0:
            score -= 0.08
            suggestions.append("Include quantitative evidence where the literature provides it.")
            feedback.append("No quantitative evidence is surfaced.")

        if len(domain_hits) < max(2, min(5, len(domain_keywords) // 4 or 2)):
            score -= 0.16
            suggestions.append("Use more domain-specific terminology tied to the thesis and evidence.")
            feedback.append("Technical depth is limited relative to the topic.")

        if len(long_words) < 8:
            score -= 0.08
            suggestions.append("Add discipline-specific detail, methodology, and comparative analysis.")

        if thesis:
            thesis_terms = [term for term in self._domain_keywords(thesis, []) if term in lowered]
            if not thesis_terms:
                score -= 0.12
                suggestions.append("Align the section more explicitly with the central thesis.")
                feedback.append("The thesis is not clearly advanced in the current draft.")

        score = max(0.0, min(1.0, score))
        required_threshold = threshold if threshold is not None else self.approval_threshold
        approved = score >= required_threshold
        if approved and not feedback:
            feedback.append("The section is specific, grounded, and technically credible.")

        return {
            "score": round(score, 2),
            "feedback": " ".join(feedback),
            "suggestions": suggestions,
            "approved": approved,
            "metrics": {
                "repeated_phrase_ratio": round(repeated_ratio, 3),
                "citation_count": citation_count,
                "generic_phrase_count": len(generic_hits),
                "quantitative_signal_count": digit_count,
                "domain_keyword_hits": len(domain_hits),
            },
        }

    def _repeated_phrase_ratio(self, content: str, ngram_size: int = 4) -> float:
        words = re.findall(r"\b\w+\b", content.lower())
        if len(words) < ngram_size * 2:
            return 0.0

        counts = {}
        total = 0
        for idx in range(len(words) - ngram_size + 1):
            gram = " ".join(words[idx:idx + ngram_size])
            counts[gram] = counts.get(gram, 0) + 1
            total += 1

        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / total if total else 0.0

    def _domain_keywords(self, thesis: str, research_notes: list) -> list[str]:
        corpus = [thesis]
        corpus.extend(note if isinstance(note, str) else json.dumps(note) for note in research_notes[:8])
        terms = []
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", " ".join(corpus).lower()):
            if len(token) >= 5 and token not in terms:
                terms.append(token)
        return terms[:20]

    async def _llm_review(self, section: str, content: str, thesis: str, research_notes: list, heuristic: dict, project_id: str, db) -> dict:
        try:
            prompt = (
                f"You are a strict dissertation supervisor reviewing the '{section}' section of a final-year academic paper. "
                "Return JSON with keys: score (0-1), feedback (string), suggestions (array of strings), approved (boolean). "
                "Be severe about repetition, generic academic filler, uncited claims, and lack of technical depth. "
                "Require named examples, evidence, and explicit alignment with the thesis.\n\n"
                f"Thesis:\n{thesis}\n\n"
                f"Research notes:\n{json.dumps(research_notes[:6])}\n\n"
                f"Heuristic review baseline:\n{json.dumps(heuristic)}\n\n"
                f"Draft:\n{content[:3500]}"
            )
            response_text = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                response_format={"type": "json_object"},
                temperature=0.15,
                max_tokens=700,
            )
            payload = json.loads(response_text)
            if not isinstance(payload, dict):
                return heuristic

            llm_score = float(payload.get("score", heuristic["score"]))
            merged_score = round(max(0.0, min(1.0, (heuristic["score"] * 0.6) + (llm_score * 0.4))), 2)
            suggestions = []
            for suggestion in heuristic.get("suggestions", []) + payload.get("suggestions", []):
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)

            feedback = " ".join(
                part for part in [heuristic.get("feedback", ""), payload.get("feedback", "")] if part
            ).strip()
            return {
                "score": merged_score,
                "feedback": feedback,
                "suggestions": suggestions,
                "approved": merged_score >= self.approval_threshold,
                "metrics": heuristic.get("metrics", {}),
            }
        except Exception:
            return heuristic
