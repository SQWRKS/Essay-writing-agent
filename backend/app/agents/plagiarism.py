"""PlagiarismAgent — originality checker for generated essay sections.

Checks the assembled essay content for:
- Near-duplicate sentences across sections (intra-essay repetition).
- Verbatim or near-verbatim phrase overlap with the supplied source abstracts
  (potential unattributed copying from the evidence pack).
- Excessively repeated n-grams within a single section.

The heuristic path is always run.  When an LLM is available and the heuristic
flags concerns, an optional LLM pass produces a short natural-language
assessment and a credibility-adjusted similarity score.
"""

import re
from collections import Counter

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text
from app.routing.model_config import AGENT_MODELS

# Minimum n-gram length (words) used for fingerprint comparison.
_NGRAM_SIZE = 6
# Similarity threshold above which two passages are considered near-duplicates.
_SIMILARITY_THRESHOLD = 0.55
# Threshold for flagging a section as potentially borrowing source language.
_SOURCE_OVERLAP_THRESHOLD = 0.40


def _tokenize(text: str) -> list[str]:
    """Return a lower-cased word list, stripping punctuation."""
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", (text or "").lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _sentence_fingerprints(text: str) -> list[tuple[str, frozenset]]:
    """Return (sentence_text, ngram_fingerprint) pairs for every sentence."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 30]
    result = []
    for sentence in sentences:
        tokens = _tokenize(sentence)
        if len(tokens) < _NGRAM_SIZE:
            continue
        fingerprint = frozenset(_ngrams(tokens, _NGRAM_SIZE))
        result.append((sentence, fingerprint))
    return result


class PlagiarismAgent(AgentBase):
    name = "plagiarism"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        sections: dict[str, str] = input_data.get("sections", {})
        sources: list[dict] = input_data.get("sources", [])

        result = self._heuristic_check(sections, sources)

        # Run LLM pass only when heuristics raise concerns, to keep cost low.
        if is_llm_available() and (result["flagged_pairs"] or result["source_overlap_flags"]):
            result = await self._llm_assess(result, sections, sources, project_id, db)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    # ------------------------------------------------------------------
    # Heuristic checks
    # ------------------------------------------------------------------

    def _heuristic_check(
        self,
        sections: dict[str, str],
        sources: list[dict],
    ) -> dict:
        """Run all heuristic originality checks and return a result dict."""
        intra_flags = self._check_intra_essay_duplication(sections)
        source_flags = self._check_source_overlap(sections, sources)
        repetition_flags = self._check_intra_section_repetition(sections)

        total_flags = len(intra_flags) + len(source_flags) + len(repetition_flags)
        score = max(0.0, 1.0 - total_flags * 0.12)

        issues: list[str] = []
        suggestions: list[str] = []

        if intra_flags:
            issues.append(
                f"{len(intra_flags)} near-duplicate sentence pair(s) found across sections."
            )
            suggestions.append(
                "Rewrite repeated sentences so each section advances the argument from a distinct angle."
            )
        if source_flags:
            issues.append(
                f"{len(source_flags)} section(s) contain language that closely mirrors source abstracts."
            )
            suggestions.append(
                "Paraphrase borrowed language and ensure all source material is properly cited."
            )
        if repetition_flags:
            issues.append(
                f"{len(repetition_flags)} section(s) contain excessively repeated internal phrases."
            )
            suggestions.append(
                "Vary phrasing within sections to reduce internal repetition."
            )

        approved = total_flags == 0
        return {
            "score": round(score, 2),
            "approved": approved,
            "feedback": "Originality check complete." if approved else "Potential originality issues detected.",
            "issues": issues[:5],
            "suggestions": suggestions[:5],
            "flagged_pairs": intra_flags[:6],
            "source_overlap_flags": source_flags[:6],
            "repetition_flags": repetition_flags[:6],
            "total_flag_count": total_flags,
        }

    def _check_intra_essay_duplication(
        self, sections: dict[str, str]
    ) -> list[dict]:
        """Detect near-duplicate sentences shared between different sections."""
        flagged: list[dict] = []
        section_fingerprints: list[tuple[str, str, frozenset]] = []

        for section_key, text in sections.items():
            for sentence, fp in _sentence_fingerprints(text):
                section_fingerprints.append((section_key, sentence, fp))

        n = len(section_fingerprints)
        for i in range(n):
            for j in range(i + 1, n):
                key_a, sent_a, fp_a = section_fingerprints[i]
                key_b, sent_b, fp_b = section_fingerprints[j]
                if key_a == key_b:
                    continue
                similarity = _jaccard(fp_a, fp_b)
                if similarity >= _SIMILARITY_THRESHOLD:
                    flagged.append(
                        {
                            "section_a": key_a,
                            "section_b": key_b,
                            "similarity": round(similarity, 3),
                            "sentence_a": sent_a[:200],
                            "sentence_b": sent_b[:200],
                        }
                    )
                    if len(flagged) >= 20:
                        return flagged
        return flagged

    def _check_source_overlap(
        self, sections: dict[str, str], sources: list[dict]
    ) -> list[dict]:
        """Flag sections whose language closely mirrors source abstracts."""
        flagged: list[dict] = []
        source_fps: list[tuple[str, frozenset]] = []

        for src in sources[:15]:
            abstract = (src.get("abstract") or src.get("abstract_excerpt") or "").strip()
            if not abstract:
                continue
            tokens = _tokenize(abstract)
            if len(tokens) < _NGRAM_SIZE:
                continue
            source_fps.append((src.get("title", "Unknown"), frozenset(_ngrams(tokens, _NGRAM_SIZE))))

        if not source_fps:
            return []

        for section_key, text in sections.items():
            tokens = _tokenize(text)
            if len(tokens) < _NGRAM_SIZE:
                continue
            section_fp = frozenset(_ngrams(tokens, _NGRAM_SIZE))
            for source_title, source_fp in source_fps:
                overlap = _jaccard(section_fp, source_fp)
                if overlap >= _SOURCE_OVERLAP_THRESHOLD:
                    flagged.append(
                        {
                            "section": section_key,
                            "source_title": source_title,
                            "similarity": round(overlap, 3),
                        }
                    )
        return flagged

    def _check_intra_section_repetition(
        self, sections: dict[str, str]
    ) -> list[dict]:
        """Detect excessively repeated trigrams within a single section."""
        flagged: list[dict] = []
        for section_key, text in sections.items():
            tokens = _tokenize(text)
            if len(tokens) < 20:
                continue
            trigram_counts = Counter(_ngrams(tokens, 3))
            repeated = [
                {" ".join(ng): count}
                for ng, count in trigram_counts.most_common(5)
                if count >= 4
            ]
            if repeated:
                flagged.append({"section": section_key, "repeated_phrases": repeated})
        return flagged

    # ------------------------------------------------------------------
    # Optional LLM assessment
    # ------------------------------------------------------------------

    async def _llm_assess(
        self,
        heuristic: dict,
        sections: dict[str, str],
        sources: list[dict],
        project_id: str,
        db,
    ) -> dict:
        """Run a brief LLM assessment of flagged passages and refine the score.

        Only the flagged pairs and the first 200 chars of each flagged section
        are sent to keep the prompt compact.
        """
        try:
            flagged_pairs_summary = heuristic.get("flagged_pairs", [])[:4]
            source_overlap_summary = heuristic.get("source_overlap_flags", [])[:4]
            section_excerpts = {
                k: truncate_text(v, 200)
                for k, v in sections.items()
                if k in {p["section_a"] for p in flagged_pairs_summary}
                | {p["section_b"] for p in flagged_pairs_summary}
                | {p["section"] for p in source_overlap_summary}
            }
            excerpts_text = "\n".join(f"{k}: {v}" for k, v in section_excerpts.items())
            prompt = (
                "You are an academic integrity checker reviewing a multi-section essay.\n\n"
                "The heuristic scan found the following potential issues:\n"
                f"Near-duplicate sentence pairs: {flagged_pairs_summary}\n"
                f"Source language overlap: {source_overlap_summary}\n\n"
                f"Relevant section excerpts:\n{excerpts_text}\n\n"
                "Assess whether the flagged issues represent genuine originality concerns "
                "or acceptable paraphrasing with proper attribution.\n"
                "Return JSON with keys:\n"
                "  score_adjustment (float -0.2 to 0.0 — how much to reduce the originality score),\n"
                "  assessment (string ≤3 sentences),\n"
                "  confirmed_issues (array of strings)."
            )
            response = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["plagiarism"]["default"],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )
            import json
            payload = json.loads(response)
            adjustment = float(payload.get("score_adjustment", 0.0))
            refined_score = max(0.0, min(1.0, heuristic["score"] + adjustment))
            confirmed = payload.get("confirmed_issues", [])
            return {
                **heuristic,
                "score": round(refined_score, 2),
                "approved": refined_score >= 0.8 and not confirmed,
                "feedback": payload.get("assessment", heuristic["feedback"]),
                "issues": list(dict.fromkeys(confirmed + heuristic.get("issues", [])))[:5],
            }
        except Exception:
            return heuristic
