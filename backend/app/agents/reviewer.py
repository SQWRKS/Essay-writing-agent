import json
import re
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text
from app.core.config import settings
from app.routing.model_config import AGENT_MODELS


TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "consequently",
    "nevertheless", "although", "while", "thus", "additionally",
}
LIMITATION_MARKERS = {
    "limitation", "limitations", "constraint", "constraints", "uncertain",
    "uncertainty", "caution", "cautious", "gap", "gaps",
}
GENERIC_PHRASES = {
    "a substantial body of literature exists",
    "this study employs a rigorous methodological framework",
    "the analysis of",
    "this study has provided a comprehensive examination",
    "researchers have explored multiple dimensions",
}
META_WRITING_PATTERNS = {
    "section objective:",
    "this section must address",
    "writing directive:",
    "revision guidance to address:",
    "must cover:",
    "evidence requirements:",
    "return only the final section text",
}


class ReviewerAgent(AgentBase):
    name = "reviewer"
    approval_threshold = 0.72

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        section = input_data.get("section", "")
        content = input_data.get("content", "")
        expected_word_count = input_data.get("expected_word_count")
        evidence_pack = input_data.get("evidence_pack", [])
        grounding_summary = input_data.get("grounding_summary", {})
        revision_attempt = int(input_data.get("revision_attempt", 0))
        rubric: str = (input_data.get("rubric") or "").strip()

        heuristic = self._heuristic_review(
            section, content, expected_word_count, evidence_pack,
            grounding_summary, revision_attempt, rubric=rubric,
        )
        if is_llm_available():
            result = await self._llm_review(
                section,
                content,
                expected_word_count,
                evidence_pack,
                grounding_summary,
                revision_attempt,
                project_id,
                db,
                rubric=rubric,
            )
        else:
            result = heuristic

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _extract_citation_markers(self, content: str) -> list[str]:
        return re.findall(r"\[(\d+)\]", content)

    def _paragraphs(self, content: str) -> list[str]:
        return [part.strip() for part in content.split("\n\n") if part.strip()]

    def _sentence_count(self, content: str) -> int:
        return len([chunk for chunk in re.split(r"[.!?]+", content) if chunk.strip()])

    def _repetition_ratio(self, content: str) -> float:
        words = [word.lower() for word in re.findall(r"\b[a-zA-Z]{4,}\b", content)]
        if not words:
            return 1.0
        return len(set(words)) / len(words)

    def _evidence_keyword_overlap(self, content: str, evidence_pack: list[dict]) -> float:
        if not evidence_pack:
            return 1.0

        content_terms = {word.lower() for word in re.findall(r"\b[a-zA-Z]{4,}\b", content)}
        evidence_terms = set()
        for evidence in evidence_pack[:6]:
            title = evidence.get("title", "")
            evidence_terms.update(word.lower() for word in re.findall(r"\b[a-zA-Z]{4,}\b", title))

        if not evidence_terms:
            return 0.7

        overlap = len(content_terms & evidence_terms)
        return min(1.0, overlap / max(3, min(8, len(evidence_terms))))

    def _heuristic_review(
        self,
        section: str,
        content: str,
        expected_word_count: int | None = None,
        evidence_pack: list[dict] | None = None,
        grounding_summary: dict | None = None,
        revision_attempt: int = 0,
        rubric: str = "",
    ) -> dict:
        evidence_pack = evidence_pack or []
        grounding_summary = grounding_summary or {}
        words = content.split()
        word_count = len(words)
        suggestions = []
        strengths = []
        blocking_issues = []
        normalized = content.lower()
        expected_word_count = expected_word_count or 450
        citation_count = len(self._extract_citation_markers(content))
        paragraphs = self._paragraphs(content)
        sentence_count = self._sentence_count(content)
        repetition_ratio = self._repetition_ratio(content)
        evidence_overlap = self._evidence_keyword_overlap(content, evidence_pack)
        meta_markers = [marker for marker in META_WRITING_PATTERNS if marker in normalized]

        coverage_score = min(1.0, word_count / max(120, expected_word_count * 0.9))
        if word_count < max(120, int(expected_word_count * 0.55)):
            blocking_issues.append("Section is materially under the requested length.")
            suggestions.append("Expand the section with more analysis, evidence, and explanation.")
        elif coverage_score >= 0.9:
            strengths.append("Section is close to the requested depth and length.")

        structure_score = 0.4
        if len(paragraphs) >= 2:
            structure_score += 0.25
        if sentence_count >= 5:
            structure_score += 0.15
        if any(word in normalized for word in TRANSITION_WORDS):
            structure_score += 0.2
            strengths.append("Section uses connective language to support flow.")
        else:
            suggestions.append("Add stronger transitions so the argument develops more clearly.")
        structure_score = min(1.0, structure_score)

        grounding_score = 0.35
        if evidence_pack:
            if citation_count >= max(2, min(5, len(evidence_pack))):
                grounding_score += 0.4
                strengths.append("Key claims are accompanied by citation markers.")
            elif citation_count == 0:
                blocking_issues.append("Evidence was provided but the section does not cite it.")
                suggestions.append("Add inline citation markers for factual claims and evidence-backed statements.")
            else:
                grounding_score += 0.2

            grounding_score += 0.25 * evidence_overlap
            if evidence_overlap < 0.25:
                suggestions.append("Use more evidence directly from the retrieved sources instead of generic discussion.")
        else:
            grounding_score = 0.65
        grounding_score = min(1.0, grounding_score)

        generic_hits = [phrase for phrase in GENERIC_PHRASES if phrase in normalized]
        digit_count = sum(1 for ch in content if ch.isdigit())
        evidence_terms = {
            token.lower()
            for item in evidence_pack[:8]
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", item.get("title", "") + " " + item.get("abstract", ""))
        }
        content_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", normalized))
        domain_hits = content_tokens & evidence_terms

        analysis_score = 0.35
        if any(word in normalized for word in LIMITATION_MARKERS):
            analysis_score += 0.25
            strengths.append("Section acknowledges uncertainty or limitations.")
        else:
            suggestions.append("Add at least one limitation, uncertainty, or counterpoint.")
        if any(token in normalized for token in ["because", "therefore", "suggests", "indicates", "implies"]):
            analysis_score += 0.25
        if digit_count > 0:
            analysis_score += 0.15
        analysis_score += 0.25 * min(1.0, sentence_count / 8)
        analysis_score = min(1.0, analysis_score)

        clarity_score = 0.45
        if repetition_ratio >= 0.45:
            clarity_score += 0.25
        else:
            suggestions.append("Reduce repetitive phrasing and vary sentence construction.")
        if not generic_hits:
            clarity_score += 0.2
        else:
            suggestions.append("Replace generic academic boilerplate with more specific argumentation.")
        if sentence_count >= 4:
            clarity_score += 0.1
        if meta_markers:
            clarity_score -= 0.35
            blocking_issues.append("Section contains instructional/meta-writing phrases instead of final argumentative prose.")
            suggestions.append("Rewrite the section as direct academic prose and remove planning or instruction headers.")
        clarity_score = min(1.0, clarity_score)
        clarity_score = max(0.0, clarity_score)

        category_scores = {
            "coverage": round(coverage_score, 2),
            "structure": round(structure_score, 2),
            "grounding": round(grounding_score, 2),
            "analysis": round(analysis_score, 2),
            "clarity": round(clarity_score, 2),
        }

        grounding_validator_score = float(grounding_summary.get("score", category_scores["grounding"]))
        category_scores["grounding"] = round((category_scores["grounding"] * 0.6) + (grounding_validator_score * 0.4), 2)
        score = round(sum(category_scores.values()) / len(category_scores), 2)

        if grounding_summary.get("unsupported_claim_count", 0) > 0:
            blocking_issues.append("Grounding validator found unsupported claim-like sentences.")
            suggestions.append("Revise unsupported claims so each one is tied to the evidence pack and citations.")
        for issue in grounding_summary.get("issues", [])[:2]:
            if issue not in blocking_issues:
                blocking_issues.append(issue)

        if category_scores["grounding"] < 0.45:
            blocking_issues.append("Section is not grounded strongly enough in the available evidence.")
        if category_scores["clarity"] < 0.45:
            blocking_issues.append("Section is too generic or repetitive to meet the quality bar.")
        if meta_markers and "Section contains instructional/meta-writing phrases instead of final argumentative prose." not in blocking_issues:
            blocking_issues.append("Section contains instructional/meta-writing phrases instead of final argumentative prose.")

        approved = score >= settings.REVIEW_MIN_SCORE and not blocking_issues
        if approved:
            feedback = "Section meets the current quality gate."
        else:
            feedback = "Section needs revision before it meets the current quality gate."

        return {
            "score": score,
            "feedback": feedback,
            "suggestions": suggestions[:6],
            "strengths": strengths[:5],
            "blocking_issues": list(dict.fromkeys(blocking_issues))[:5],
            "category_scores": category_scores,
            "citation_count": citation_count,
            "grounding_score": round(grounding_validator_score, 2),
            "revision_attempt": revision_attempt,
            "approved": approved,
            "metrics": {
                "repeated_phrase_ratio": round(repetition_ratio, 3),
                "citation_count": citation_count,
                "generic_phrase_count": len(generic_hits),
                "quantitative_signal_count": digit_count,
                "domain_keyword_hits": len(domain_hits),
            },
        }

    async def _llm_review(
        self,
        section: str,
        content: str,
        expected_word_count: int | None,
        evidence_pack: list[dict],
        grounding_summary: dict,
        revision_attempt: int,
        project_id: str,
        db,
        rubric: str = "",
    ) -> dict:
        try:
            evidence_digest = json.dumps([
                {
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "source": item.get("source"),
                    "relevance_score": item.get("relevance_score"),
                }
                for item in evidence_pack[:3]
            ])
            # Optional rubric block — only added when the user supplied one
            rubric_block = f"MARKING RUBRIC:\n{rubric[:800]}\nEvaluate the section against the above rubric criteria in addition to standard quality dimensions.\n\n" if rubric else ""
            # Truncate content to 2500 chars to reduce prompt tokens while
            # still giving the reviewer enough text to evaluate quality.
            # Chain-of-thought: the model reasons step-by-step before scoring,
            # which improves calibration and reduces surface-level judgements.
            prompt = (
                f"You are an academic editor reviewing the '{section}' section of an essay.\n"
                f"Expected word count: {expected_word_count or 450}. Revision attempt: {revision_attempt}.\n\n"
                f"{rubric_block}"
                f"Grounding summary: {json.dumps(grounding_summary)}\n"
                f"Evidence pack: {evidence_digest}\n\n"
                f"Section content:\n{truncate_text(content, 2500)}\n\n"
                "Think step-by-step before scoring:\n"
                "Step 1 – Coverage: Does the section reach the expected word count and depth?\n"
                "Step 2 – Structure: Is the argument logically ordered with clear transitions?\n"
                "Step 3 – Grounding: Is every factual claim backed by a citation or the evidence pack?\n"
                "Step 4 – Analysis: Does the section interpret evidence rather than just summarise it?\n"
                "Step 5 – Clarity: Is the prose specific, free of generic boilerplate, and non-repetitive?\n"
                "Step 6 – Blocking issues: List any issue that must be fixed before the section can be approved.\n\n"
                "After reasoning through each step, return a single JSON object with keys:\n"
                "score (float 0-1), approved (bool), feedback (string ≤2 sentences),\n"
                "suggestions (array), strengths (array), blocking_issues (array),\n"
                "category_scores (object with keys coverage, structure, grounding, analysis, clarity).\n"
                "Reject (approved=false) if the section is generic, weakly grounded, underdeveloped, "
                "or lacks evidence citations."
            )
            response_text = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["reviewer"]["default"],
                response_format={"type": "json_object"},
                temperature=0.15,
                max_tokens=700,
            )
            result = json.loads(response_text)
            result.setdefault("suggestions", [])
            result.setdefault("strengths", [])
            result.setdefault("blocking_issues", [])
            result.setdefault("category_scores", {})
            result.setdefault("feedback", "Section reviewed.")
            result["score"] = round(float(result.get("score", 0.0)), 2)
            result["approved"] = bool(result.get("approved", False)) and result["score"] >= settings.REVIEW_MIN_SCORE
            result["revision_attempt"] = revision_attempt
            result["citation_count"] = len(self._extract_citation_markers(content))
            result["grounding_score"] = round(float(grounding_summary.get("score", 0.0)), 2)
            return result
        except Exception:
            return self._heuristic_review(section, content, expected_word_count, evidence_pack, grounding_summary, revision_attempt, rubric=rubric)
