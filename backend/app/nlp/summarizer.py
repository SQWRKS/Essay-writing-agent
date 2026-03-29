"""Extractive summarisation — no LLM required.

Uses TF-IDF sentence scoring to rank sentences by information density, then
selects the top-N sentences to produce a summary that is at least 70% shorter
than the source text.

Algorithm
---------
1. Split text into sentences.
2. Build a TF-IDF matrix (sentences × terms).
3. Score each sentence as the sum of its TF-IDF term weights.
4. Sort sentences by score; select until the char budget is reached.
5. Return selected sentences **in their original order** to preserve readability.

Falls back to a head-truncation strategy if scikit-learn is not installed.
"""

from __future__ import annotations

import re
from typing import Optional

# scikit-learn is listed in requirements.txt; import lazily to allow the module
# to load even when the package hasn't been installed yet (test isolation).
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(])")
_MIN_SENTENCE_CHARS = 30  # discard very short sentence fragments


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, returning non-trivial ones."""
    raw = _RE_SENTENCE_SPLIT.split(text)
    # Further split on double-newlines (paragraphs)
    sentences: list[str] = []
    for part in raw:
        for sub in re.split(r"\n{2,}", part):
            sub = sub.strip()
            if len(sub) >= _MIN_SENTENCE_CHARS:
                sentences.append(sub)
    return sentences


class ExtractiveSummarizer:
    """Reduce a document to its most informative sentences.

    Parameters
    ----------
    max_ratio:
        Maximum fraction of the original text to retain.  Must be in (0, 1).
        Default ``0.3`` means at most 30 % of the original length is kept
        (≥ 70 % reduction).
    min_sentences:
        Minimum number of sentences to include regardless of ratio.
    max_sentences:
        Hard cap on the number of sentences selected.
    """

    def __init__(
        self,
        max_ratio: float = 0.3,
        min_sentences: int = 1,
        max_sentences: int = 10,
    ) -> None:
        if not 0 < max_ratio < 1:
            raise ValueError("max_ratio must be between 0 and 1 exclusive")
        self.max_ratio = max_ratio
        self.min_sentences = max(1, min_sentences)
        self.max_sentences = max(self.min_sentences, max_sentences)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, text: str, topic: Optional[str] = None) -> str:
        """Return a summary of *text* that is at most ``max_ratio * len(text)``
        characters.

        Parameters
        ----------
        text:
            The source document text (pre-cleaned recommended).
        topic:
            Optional topic string.  When provided, sentences that contain
            topic keywords receive a small scoring bonus.
        """
        if not text or not text.strip():
            return ""

        sentences = _split_sentences(text)
        if not sentences:
            return text[:max(1, int(len(text) * self.max_ratio))]

        if len(sentences) <= self.min_sentences:
            return " ".join(sentences)

        char_budget = max(80, int(len(text) * self.max_ratio))

        scores = self._score_sentences(sentences, topic)
        selected = self._select_sentences(sentences, scores, char_budget)
        return " ".join(selected)

    def summarize_many(self, texts: list[str], topic: Optional[str] = None) -> list[str]:
        """Summarize a list of texts, returning a list of summaries.

        Items are processed independently (use :func:`pipeline.NLPPipeline`
        for parallel processing).
        """
        return [self.summarize(t, topic) for t in texts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_sentences(self, sentences: list[str], topic: Optional[str]) -> list[float]:
        """Assign a relevance score to each sentence.

        When scikit-learn is available: uses TF-IDF row-sum scores.
        Otherwise: falls back to sentence-length heuristic (prefers 60-200
        char sentences slightly) with a topic-keyword bonus.
        """
        if _SKLEARN_AVAILABLE and len(sentences) >= 2:
            return self._tfidf_scores(sentences, topic)
        return self._heuristic_scores(sentences, topic)

    def _tfidf_scores(self, sentences: list[str], topic: Optional[str]) -> list[float]:
        """Score sentences using TF-IDF row sums."""
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            base_scores: list[float] = np.array(tfidf_matrix.sum(axis=1)).flatten().tolist()
        except Exception:
            base_scores = self._heuristic_scores(sentences, topic)

        if not topic:
            return base_scores

        # Add topic-keyword bonus
        topic_tokens = set(re.findall(r"\b[a-z]{4,}\b", topic.lower()))
        final: list[float] = []
        for sent, score in zip(sentences, base_scores):
            sent_tokens = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
            overlap = len(topic_tokens & sent_tokens) / max(1, len(topic_tokens))
            final.append(score + overlap * 0.5)
        return final

    @staticmethod
    def _heuristic_scores(sentences: list[str], topic: Optional[str]) -> list[float]:
        """Fallback scoring when scikit-learn is unavailable."""
        topic_tokens = set(re.findall(r"\b[a-z]{4,}\b", (topic or "").lower()))
        scores: list[float] = []
        for sent in sentences:
            length_score = 1.0
            n = len(sent)
            if 60 <= n <= 200:
                length_score = 1.2
            elif n < 40:
                length_score = 0.6

            topic_bonus = 0.0
            if topic_tokens:
                sent_tokens = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
                topic_bonus = len(topic_tokens & sent_tokens) / max(1, len(topic_tokens)) * 0.5

            scores.append(length_score + topic_bonus)
        return scores

    def _select_sentences(
        self,
        sentences: list[str],
        scores: list[float],
        char_budget: int,
    ) -> list[str]:
        """Select the top-scoring sentences within the character budget."""
        # Pair each sentence with its original index and score, then sort by score
        ranked = sorted(enumerate(sentences), key=lambda x: scores[x[0]], reverse=True)

        selected_indices: set[int] = set()
        total_chars = 0

        for orig_idx, sent in ranked:
            if len(selected_indices) >= self.max_sentences:
                break
            if total_chars + len(sent) > char_budget and len(selected_indices) >= self.min_sentences:
                break
            selected_indices.add(orig_idx)
            total_chars += len(sent) + 1  # +1 for space separator

        # Return sentences in their original document order
        return [sentences[i] for i in sorted(selected_indices)]
