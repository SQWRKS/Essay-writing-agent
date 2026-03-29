"""Hybrid document retriever — BM25 + vector similarity.

Combines two complementary ranking signals:
- **BM25** (lexical): exact-term matching, good for keyword queries.
- **TF-IDF cosine** (semantic proxy): captures related terms; optionally
  upgraded to real dense embeddings if *sentence-transformers* is installed.

Final score = ``alpha * bm25_norm + (1 - alpha) * embedding_norm``

Dependencies
------------
Required:
  - ``rank_bm25`` (lightweight pure-Python BM25)
  - ``scikit-learn`` (TF-IDF fallback for embeddings)
  - ``numpy``

Optional:
  - ``sentence_transformers`` — upgrades the embedding component from TF-IDF
    cosine to dense sentence embeddings.  Falls back silently if absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Required
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# Optional dense embeddings
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


@dataclass
class RetrievedChunk:
    """A document chunk returned by the retriever with its combined score."""

    text: str
    index: int        # original position in the corpus
    bm25_score: float
    embedding_score: float
    combined_score: float
    metadata: dict    # pass-through from the input corpus item


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())


class HybridRetriever:
    """Retrieve the most relevant text chunks from a corpus for a given query.

    Parameters
    ----------
    alpha:
        Weight of the BM25 score in the final combined score.
        ``1 - alpha`` is the weight of the embedding score.
        Range [0, 1].  Default ``0.5`` (equal blend).
    sbert_model:
        Name of the sentence-transformers model to load if the library is
        available.  Falls back to TF-IDF cosine if the model cannot be loaded.
    top_k:
        Number of results to return by default.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        sbert_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.top_k = top_k
        self._sbert_model_name = sbert_model
        self._sbert: Optional[object] = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build_index(self, documents: list[str], metadata: Optional[list[dict]] = None) -> "_IndexedCorpus":
        """Pre-process *documents* and return an :class:`_IndexedCorpus` that
        can be queried repeatedly without recomputing the index.

        Parameters
        ----------
        documents:
            List of text strings (pre-cleaned recommended).
        metadata:
            Optional per-document metadata dicts (same length as *documents*).
        """
        if not documents:
            return _IndexedCorpus([], [], None, None, metadata or [], self)

        meta = metadata or [{} for _ in documents]
        tokenized = [_tokenize(d) for d in documents]

        bm25 = BM25Okapi(tokenized) if _BM25_AVAILABLE else None

        # Build TF-IDF matrix for embedding fallback
        tfidf_vectorizer: Optional[TfidfVectorizer] = None
        tfidf_matrix = None
        if _SKLEARN_AVAILABLE:
            try:
                tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            except Exception:
                tfidf_vectorizer = None

        return _IndexedCorpus(
            documents=documents,
            tokenized=tokenized,
            bm25=bm25,
            tfidf=(tfidf_vectorizer, tfidf_matrix),
            metadata=meta,
            retriever=self,
        )

    # ------------------------------------------------------------------
    # Convenience one-shot retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        documents: list[str],
        metadata: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Build index and retrieve in a single call.

        Use :meth:`build_index` + :meth:`_IndexedCorpus.query` when you need
        to query the same corpus multiple times.
        """
        corpus = self.build_index(documents, metadata)
        return corpus.query(query, top_k=top_k or self.top_k)

    # ------------------------------------------------------------------
    # Dense embedding helper (lazy-loaded)
    # ------------------------------------------------------------------

    def _get_sbert(self) -> Optional[object]:
        """Return the SentenceTransformer model, loading it on first call."""
        if not _SBERT_AVAILABLE:
            return None
        if self._sbert is None:
            try:
                self._sbert = SentenceTransformer(self._sbert_model_name)
            except Exception:
                self._sbert = None
        return self._sbert


class _IndexedCorpus:
    """Internal class holding a pre-built index for repeated queries."""

    __slots__ = ("documents", "tokenized", "bm25", "tfidf", "metadata", "retriever")

    def __init__(
        self,
        documents: list[str],
        tokenized: list[list[str]],
        bm25: Optional[object],
        tfidf: tuple,
        metadata: list[dict],
        retriever: HybridRetriever,
    ) -> None:
        self.documents = documents
        self.tokenized = tokenized
        self.bm25 = bm25
        self.tfidf = tfidf          # (vectorizer, matrix) or (None, None)
        self.metadata = metadata
        self.retriever = retriever

    def query(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """Return the top-*k* most relevant chunks for *query*."""
        k = top_k or self.retriever.top_k
        n = len(self.documents)
        if n == 0:
            return []

        k = min(k, n)

        bm25_scores = self._bm25_scores(query, n)
        embedding_scores = self._embedding_scores(query, n)

        # Normalise both score vectors to [0, 1]
        bm25_norm = _min_max_norm(bm25_scores)
        emb_norm = _min_max_norm(embedding_scores)

        alpha = self.retriever.alpha
        combined = [alpha * b + (1 - alpha) * e for b, e in zip(bm25_norm, emb_norm)]

        ranked_indices = sorted(range(n), key=lambda i: combined[i], reverse=True)[:k]

        return [
            RetrievedChunk(
                text=self.documents[idx],
                index=idx,
                bm25_score=round(bm25_scores[idx], 4),
                embedding_score=round(embedding_scores[idx], 4),
                combined_score=round(combined[idx], 4),
                metadata=self.metadata[idx],
            )
            for idx in ranked_indices
        ]

    def _bm25_scores(self, query: str, n: int) -> list[float]:
        if self.bm25 is None:
            # Fallback: simple token overlap
            q_tokens = set(_tokenize(query))
            return [
                len(q_tokens & set(tokens)) / max(1, len(q_tokens))
                for tokens in self.tokenized
            ]
        q_tokens = _tokenize(query)
        raw = self.bm25.get_scores(q_tokens)
        return raw.tolist() if hasattr(raw, "tolist") else list(raw)

    def _embedding_scores(self, query: str, n: int) -> list[float]:
        # Try dense SBERT embeddings first
        sbert = self.retriever._get_sbert()
        if sbert is not None and _SKLEARN_AVAILABLE:
            try:
                q_emb = sbert.encode([query])
                d_emb = sbert.encode(self.documents)
                sims = cosine_similarity(q_emb, d_emb)[0]
                return sims.tolist()
            except Exception:
                pass

        # Fallback: TF-IDF cosine
        vectorizer, matrix = self.tfidf if self.tfidf else (None, None)
        if vectorizer is not None and matrix is not None and _SKLEARN_AVAILABLE:
            try:
                q_vec = vectorizer.transform([query])
                sims = cosine_similarity(q_vec, matrix)[0]
                return sims.tolist()
            except Exception:
                pass

        # Final fallback: token overlap
        q_tokens = set(_tokenize(query))
        return [
            len(q_tokens & set(tokens)) / max(1, len(q_tokens))
            for tokens in self.tokenized
        ]


def _min_max_norm(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1] using min-max scaling."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]
