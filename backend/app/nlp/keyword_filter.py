"""Keyword extraction and relevance-based sentence filter.

KeywordFilter extracts salient keywords from an essay topic and uses them to
score and filter sentences, **removing irrelevant content before it reaches
the LLM**.  This reduces token usage without losing important material.

Algorithm
---------
1. Extract candidate keywords from the topic string using TF-IDF scoring
   over a pseudo-corpus of topic words + simple frequency heuristics.
2. Score each sentence in the source text by the fraction of topic keywords
   it contains (normalised to [0, 1]).
3. Discard sentences whose score is below ``threshold``.

Falls back to a simple stopword-based approach when scikit-learn is absent.
"""

from __future__ import annotations

import re
from typing import Optional

# Optional (but listed in requirements.txt)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# Common English stopwords (subset — no external files needed)
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from as is was are were be been "
    "being have has had do does did will would could should may might must can "
    "this that these those it its it's we us our they them their he she him her "
    "also just more most only other such than very well all any if not no so up "
    "about above after again against before between both each further here into "
    "out over same shan some through under until while then there those through".split()
)


def _tokenize_lower(text: str) -> list[str]:
    """Return lowercase alphabetic tokens of length ≥ 3."""
    return [t for t in re.findall(r"\b[a-z]{3,}\b", text.lower()) if t not in _STOPWORDS]


class KeywordFilter:
    """Extract keywords from a topic and filter sentences by relevance.

    Parameters
    ----------
    threshold:
        Minimum fraction of topic keywords a sentence must contain to pass
        the filter.  Range [0, 1].  Default ``0.1`` (10 %).
    max_keywords:
        Maximum number of keywords to extract from the topic.
    expand_with_bigrams:
        If ``True``, also include bigrams from the topic as keyword phrases.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        max_keywords: int = 20,
        expand_with_bigrams: bool = True,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.max_keywords = max(1, max_keywords)
        self.expand_with_bigrams = expand_with_bigrams

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_keywords(self, topic: str, extra_context: Optional[list[str]] = None) -> list[str]:
        """Return a list of keywords extracted from *topic*.

        Parameters
        ----------
        topic:
            The essay topic string.
        extra_context:
            Optional list of additional context strings (e.g. research queries)
            used to expand or re-weight keywords via TF-IDF.
        """
        if not topic:
            return []

        tokens = _tokenize_lower(topic)
        bigrams: list[str] = []
        if self.expand_with_bigrams and len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

        base_keywords = list(dict.fromkeys(tokens + bigrams))

        if not extra_context or not _SKLEARN_AVAILABLE:
            return base_keywords[: self.max_keywords]

        # Re-weight using TF-IDF over topic + extra_context
        corpus = [topic] + extra_context
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
            matrix = vec.fit_transform(corpus)
            feature_names = vec.get_feature_names_out()
            # Use the topic row (index 0) scores
            topic_row = np.asarray(matrix[0].todense()).flatten()
            top_indices = topic_row.argsort()[::-1][: self.max_keywords]
            tfidf_keywords = [feature_names[i] for i in top_indices if topic_row[i] > 0]
            if tfidf_keywords:
                return tfidf_keywords
        except Exception:
            pass

        return base_keywords[: self.max_keywords]

    def score_sentence(self, sentence: str, keywords: list[str]) -> float:
        """Score *sentence* by keyword coverage.

        Returns a float in [0, 1] representing the fraction of *keywords*
        that appear in the sentence (case-insensitive, substring match for
        multi-word keywords).
        """
        if not keywords or not sentence:
            return 0.0
        sent_lower = sentence.lower()
        hits = sum(1 for kw in keywords if kw.lower() in sent_lower)
        return hits / len(keywords)

    def filter_sentences(
        self,
        text: str,
        keywords: list[str],
        threshold: Optional[float] = None,
    ) -> str:
        """Return *text* with low-relevance sentences removed.

        Parameters
        ----------
        text:
            Input text to filter.
        keywords:
            Keywords to score against (from :meth:`extract_keywords`).
        threshold:
            Override the instance threshold for this call.
        """
        if not text or not keywords:
            return text

        thr = threshold if threshold is not None else self.threshold
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        kept: list[str] = []
        for sent in sentences:
            score = self.score_sentence(sent, keywords)
            if score >= thr:
                kept.append(sent)

        if not kept:
            # Safety: always keep at least some content
            return text

        return " ".join(kept)

    def filter_sources(
        self,
        sources: list[dict],
        topic: str,
        abstract_field: str = "abstract",
        threshold: Optional[float] = None,
    ) -> list[dict]:
        """Filter a list of source dicts by abstract relevance to *topic*.

        Sources whose abstract keyword-score is below *threshold* are removed.
        Sources with no abstract are kept (can't score them).

        Parameters
        ----------
        sources:
            List of source dicts, each expected to have an ``abstract_field``.
        topic:
            The essay topic.
        abstract_field:
            Key in each source dict that holds the abstract text.
        threshold:
            Score threshold; defaults to the instance ``threshold``.
        """
        if not sources or not topic:
            return sources

        thr = threshold if threshold is not None else self.threshold
        keywords = self.extract_keywords(topic)
        if not keywords:
            return sources

        filtered: list[dict] = []
        for src in sources:
            abstract = (src.get(abstract_field) or "").strip()
            if not abstract:
                filtered.append(src)
                continue
            score = self.score_sentence(abstract, keywords)
            if score >= thr:
                filtered.append(src)
        return filtered
