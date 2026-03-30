"""Essay validation layer — three independent non-LLM validators.

Classes
-------
EssayStructureValidator
    Checks for the presence of key structural components (intro, thesis, body
    arguments, counterargument, conclusion) using heuristics and returns a
    structured score with a list of missing components.

ReadabilityAnalyzer
    Computes Flesch Reading Ease, average sentence length, and passive-voice
    density for each section or the full essay.  Flags problematic passages.

RuleBasedCritic
    Detects surface-level quality issues (repeated phrases, weak hedges, lack
    of evidence indicators) **without rewriting** — produces flags only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Essay Structure Validator
# ---------------------------------------------------------------------------

# Keyword triggers for each structural component
_INTRO_TRIGGERS = re.compile(
    r"\b(introduction|this paper|this study|we examine|this research|we investigate|the aim|the purpose|the objective|the focus)\b",
    re.IGNORECASE,
)
_THESIS_TRIGGERS = re.compile(
    r"\b(this paper argues|we argue|the main argument|the central thesis|the thesis|we contend|we claim|it is argued|this study argues|the argument is|the position taken)\b",
    re.IGNORECASE,
)
_ARGUMENT_MARKERS = re.compile(
    r"\b(firstly|secondly|thirdly|furthermore|moreover|in addition|additionally"
    r"|another (?:key|important)"
    r"|a (?:key|further|major) (?:reason|argument|point)"
    r"|one (?:reason|argument)"
    r"|evidence (?:suggests|shows|indicates)"
    r"|studies (?:show|indicate|suggest)"
    r"|research (?:shows|indicates|demonstrates|suggests))",
    re.IGNORECASE,
)
_COUNTER_TRIGGERS = re.compile(
    r"\b(however|on the other hand|critics argue|some argue|it could be argued|a counter(argument)?|nevertheless|despite|although|while (some|critics)|opponents|the opposing view|contrary to)\b",
    re.IGNORECASE,
)
_CONCLUSION_TRIGGERS = re.compile(
    r"\b(in conclusion|to conclude|to summarise|to summarize|in summary|this (paper|study) has|the (findings|results|analysis) (suggest|show|demonstrate|indicate)|overall, |thus, (this|the)|this discussion (has|shows))\b",
    re.IGNORECASE,
)

_MIN_ARGUMENT_COUNT = 3


@dataclass
class StructureReport:
    """Result returned by :class:`EssayStructureValidator`."""

    score: float                              # 0.0 – 1.0
    present: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    argument_count: int = 0
    details: dict[str, str] = field(default_factory=dict)


class EssayStructureValidator:
    """Heuristic checker for academic essay structure.

    Checks the full essay text for the presence of:
    - Introduction
    - Thesis statement
    - At least 3 distinct argument markers
    - Counterargument acknowledgement
    - Conclusion

    A weighted score is returned (components have different weights).
    """

    _WEIGHTS: dict[str, float] = {
        "introduction": 0.20,
        "thesis": 0.25,
        "arguments": 0.30,
        "counterargument": 0.10,
        "conclusion": 0.15,
    }

    def validate(self, text: str) -> StructureReport:
        """Validate the structural completeness of *text*.

        Parameters
        ----------
        text:
            The full essay (or a long section).  Multi-section essays should
            be passed as a single concatenated string.
        """
        if not text or not text.strip():
            return StructureReport(score=0.0, missing=list(self._WEIGHTS.keys()))

        has_intro = bool(_INTRO_TRIGGERS.search(text))
        has_thesis = bool(_THESIS_TRIGGERS.search(text))
        arg_matches = _ARGUMENT_MARKERS.findall(text)
        argument_count = len(arg_matches)
        has_enough_args = argument_count >= _MIN_ARGUMENT_COUNT
        has_counter = bool(_COUNTER_TRIGGERS.search(text))
        has_conclusion = bool(_CONCLUSION_TRIGGERS.search(text))

        present = {
            "introduction": has_intro,
            "thesis": has_thesis,
            "arguments": has_enough_args,
            "counterargument": has_counter,
            "conclusion": has_conclusion,
        }
        missing = [k for k, v in present.items() if not v]

        score = sum(self._WEIGHTS[k] for k, v in present.items() if v)

        details: dict[str, str] = {}
        if not has_thesis:
            details["thesis"] = "No clear thesis statement detected (try phrases like 'this paper argues …')"
        if not has_enough_args:
            details["arguments"] = (
                f"Only {argument_count} argument marker(s) found; aim for ≥ {_MIN_ARGUMENT_COUNT}"
            )
        if not has_counter:
            details["counterargument"] = "No counterargument detected (add 'however / on the other hand …')"
        if not has_conclusion:
            details["conclusion"] = "No conclusion detected (add 'in conclusion …')"

        return StructureReport(
            score=round(score, 3),
            present=present,
            missing=missing,
            argument_count=argument_count,
            details=details,
        )


# ---------------------------------------------------------------------------
# Readability Analyzer
# ---------------------------------------------------------------------------

@dataclass
class ReadabilityReport:
    """Result returned by :class:`ReadabilityAnalyzer`."""

    flesch_score: float          # 0–100; higher = easier
    avg_sentence_length: float   # words per sentence
    passive_voice_ratio: float   # 0–1; fraction of sentences with passive voice
    flagged_sentences: list[str] = field(default_factory=list)
    grade: str = ""              # Plain-English interpretation


_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_SYLLABLE = re.compile(r"[aeiouAEIOU]")
_RE_PASSIVE = re.compile(
    r"\b(was|were|is|are|been|being)\s+[a-z]+ed\b",
    re.IGNORECASE,
)
_MAX_FLAGGED_SENTENCES = 5


def _count_syllables(word: str) -> int:
    """Rough syllable count: count vowel groups in the word."""
    word = word.lower().rstrip(".,;:!?\"'")
    vowel_groups = re.findall(r"[aeiou]+", word)
    count = len(vowel_groups)
    # Subtract silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


class ReadabilityAnalyzer:
    """Compute readability metrics for essay text.

    Metrics:
    - **Flesch Reading Ease** — standard formula; ≥ 60 is target for academic.
    - **Average sentence length** — flag if > 35 words.
    - **Passive voice ratio** — flag if > 40 % of sentences are passive.
    """

    AVG_SENTENCE_LENGTH_WARN = 35   # words
    PASSIVE_RATIO_WARN = 0.40       # fraction
    FLESCH_WARN = 40                # below this is "very difficult"

    def analyze(self, text: str) -> ReadabilityReport:
        """Compute readability metrics for *text*."""
        if not text or not text.strip():
            return ReadabilityReport(
                flesch_score=0.0,
                avg_sentence_length=0.0,
                passive_voice_ratio=0.0,
                grade="N/A",
            )

        sentences = [s.strip() for s in _RE_SENT_SPLIT.split(text) if s.strip()]
        if not sentences:
            return ReadabilityReport(
                flesch_score=0.0,
                avg_sentence_length=0.0,
                passive_voice_ratio=0.0,
                grade="N/A",
            )

        words = re.findall(r"\b\w+\b", text)
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(_count_syllables(w) for w in words)

        if total_words == 0 or total_sentences == 0:
            return ReadabilityReport(
                flesch_score=0.0,
                avg_sentence_length=0.0,
                passive_voice_ratio=0.0,
                grade="N/A",
            )

        asl = total_words / total_sentences       # average sentence length
        asw = total_syllables / total_words        # average syllables per word
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
        flesch = max(0.0, min(100.0, flesch))

        # Passive voice detection
        passive_count = sum(1 for s in sentences if _RE_PASSIVE.search(s))
        passive_ratio = passive_count / total_sentences

        # Flag problematic sentences (too long OR passive)
        flagged: list[str] = []
        for sent in sentences:
            sent_words = len(re.findall(r"\b\w+\b", sent))
            is_long = sent_words > self.AVG_SENTENCE_LENGTH_WARN
            is_passive = bool(_RE_PASSIVE.search(sent))
            if (is_long or is_passive) and len(flagged) < _MAX_FLAGGED_SENTENCES:
                tag = []
                if is_long:
                    tag.append(f"long ({sent_words} words)")
                if is_passive:
                    tag.append("passive voice")
                flagged.append(f"[{', '.join(tag)}] {sent[:120]}…")

        grade = self._grade(flesch)
        return ReadabilityReport(
            flesch_score=round(flesch, 1),
            avg_sentence_length=round(asl, 1),
            passive_voice_ratio=round(passive_ratio, 3),
            flagged_sentences=flagged,
            grade=grade,
        )

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "Very Easy"
        if score >= 80:
            return "Easy"
        if score >= 70:
            return "Fairly Easy"
        if score >= 60:
            return "Standard"
        if score >= 50:
            return "Fairly Difficult"
        if score >= 30:
            return "Difficult"
        return "Very Difficult"


# ---------------------------------------------------------------------------
# Rule-Based Critic
# ---------------------------------------------------------------------------

@dataclass
class CriticReport:
    """Result returned by :class:`RuleBasedCritic`."""

    issues: list[dict] = field(default_factory=list)  # list of {type, message, excerpt}
    issue_count: int = 0
    severity: str = "none"   # "none", "minor", "moderate", "major"


# Weak/hedge patterns (in academic writing these may indicate lack of rigour)
_WEAK_PATTERNS = re.compile(
    r"\b(I think|I believe|I feel|I suppose|maybe|perhaps|kind of|sort of|seems like|it seems|might be|could be|probably|likely that|it appears|one might say)\b",
    re.IGNORECASE,
)

# Evidence indicator patterns (their *absence* in argumentative sentences is flagged)
_EVIDENCE_INDICATORS = re.compile(
    r"\b(according to|study (shows|found|indicates|suggests)|research (shows|indicates|demonstrates|suggests)|evidence (suggests|shows|indicates|demonstrates)|data (show|suggest|indicate)|findings (show|suggest|indicate)|[A-Z][a-z]+ et al\.?|([A-Z][a-z]+, \d{4})|cited in|as (noted|observed|reported) by)\b",
    re.IGNORECASE,
)

_ARGUMENT_SENTENCE = re.compile(
    r"\b(because|therefore|thus|hence|consequently|this (suggests|shows|demonstrates|means|implies)|the (result|evidence|data) (shows?|indicates?|demonstrates?))\b",
    re.IGNORECASE,
)

_MIN_PHRASE_LENGTH = 4  # words


class RuleBasedCritic:
    """Detect surface-quality issues in essay text.

    Issues detected:
    1. **Repeated phrases** — n-grams (n ≥ 4) repeated more than twice.
    2. **Weak argument patterns** — hedging language in argumentative sentences.
    3. **Lack of evidence** — argumentative sentences without evidence markers.

    This component **flags issues only** — it never rewrites text.
    """

    def __init__(
        self,
        repeat_threshold: int = 2,
        ngram_size: int = 4,
    ) -> None:
        self.repeat_threshold = repeat_threshold
        self.ngram_size = ngram_size

    def critique(self, text: str) -> CriticReport:
        """Run all rule checks on *text* and return a :class:`CriticReport`."""
        if not text or not text.strip():
            return CriticReport()

        issues: list[dict] = []
        issues.extend(self._check_repeated_phrases(text))
        issues.extend(self._check_weak_arguments(text))
        issues.extend(self._check_missing_evidence(text))

        count = len(issues)
        severity = "none"
        if count >= 5:
            severity = "major"
        elif count >= 3:
            severity = "moderate"
        elif count >= 1:
            severity = "minor"

        return CriticReport(issues=issues, issue_count=count, severity=severity)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_repeated_phrases(self, text: str) -> list[dict]:
        """Find n-grams repeated more than *repeat_threshold* times."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        n = self.ngram_size
        if len(words) < n:
            return []

        freq: dict[tuple, int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i: i + n])
            freq[gram] = freq.get(gram, 0) + 1

        issues: list[dict] = []
        seen: set[tuple] = set()
        for gram, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count > self.repeat_threshold and gram not in seen:
                seen.add(gram)
                phrase = " ".join(gram)
                issues.append({
                    "type": "repeated_phrase",
                    "message": f"Phrase '{phrase}' appears {count} times",
                    "excerpt": phrase,
                })
                if len(issues) >= 5:
                    break
        return issues

    def _check_weak_arguments(self, text: str) -> list[dict]:
        """Find argumentative sentences that contain hedging language."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        issues: list[dict] = []
        for sent in sentences:
            if _ARGUMENT_SENTENCE.search(sent) and _WEAK_PATTERNS.search(sent):
                excerpt = sent[:100]
                issues.append({
                    "type": "weak_argument",
                    "message": "Argumentative sentence uses hedging language",
                    "excerpt": excerpt,
                })
                if len(issues) >= 3:
                    break
        return issues

    def _check_missing_evidence(self, text: str) -> list[dict]:
        """Find argumentative sentences that lack any evidence indicator."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        issues: list[dict] = []
        argument_count = 0
        for sent in sentences:
            if not _ARGUMENT_SENTENCE.search(sent):
                continue
            argument_count += 1
            if not _EVIDENCE_INDICATORS.search(sent):
                excerpt = sent[:100]
                issues.append({
                    "type": "missing_evidence",
                    "message": "Argumentative sentence lacks a supporting evidence indicator",
                    "excerpt": excerpt,
                })
                if len(issues) >= 3:
                    break
        return issues
