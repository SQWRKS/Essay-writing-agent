"""NLPPipeline — orchestrator that wires together all non-LLM components.

This is the primary integration point between the essay-writing pipeline and
the NLP optimisation layer.  It exposes two async methods that can be called
from :class:`app.orchestration.worker_pool.WorkerPool`:

``preprocess_sources(sources, topic, queries)``
    Runs in parallel over all sources:
    1. Cleans each abstract with :class:`Preprocessor`
    2. Summarises it with :class:`ExtractiveSummarizer` (≥ 70 % reduction)
    3. Applies :class:`KeywordFilter` to keep only on-topic sentences
    Returns the enriched source list with a new ``processed_abstract`` field.

``analyze_essay(sections, topic)``
    Runs once after all sections are written:
    1. :class:`EssayStructureValidator` — checks structural completeness
    2. :class:`ReadabilityAnalyzer` — per-section + full-essay metrics
    3. :class:`RuleBasedCritic` — flags quality issues
    Returns a combined ``nlp_analysis`` dict for attaching to essay metadata.

``validate_citations(sources, style)``
    Extracts, formats, and validates citations via :class:`CitationManager`.

Both async methods use ``asyncio.get_event_loop().run_in_executor`` to run
CPU-bound NLP work off the event-loop thread so they don't block the async
pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.nlp.preprocessor import Preprocessor
from app.nlp.summarizer import ExtractiveSummarizer
from app.nlp.retriever import HybridRetriever
from app.nlp.validators import EssayStructureValidator, ReadabilityAnalyzer, RuleBasedCritic
from app.nlp.citation_manager import CitationManager
from app.nlp.keyword_filter import KeywordFilter
from app.nlp.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Shared thread-pool for CPU-bound NLP work (1-4 workers; small footprint)
_NLP_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nlp_worker")

# Default TTL for the pipeline's CacheManager (seconds).
# Cached items (preprocessed abstracts, etc.) are considered fresh for 1 hour.
_DEFAULT_CACHE_TTL: float = 3600.0


class NLPPipeline:
    """Orchestrates all non-LLM NLP components around the LLM pipeline.

    Parameters
    ----------
    cache_ttl:
        TTL in seconds for the CacheManager.  ``None`` → no expiry.
    summarize_ratio:
        Target compression ratio for ExtractiveSummarizer (default 0.3 = 70 %
        reduction).
    retriever_alpha:
        BM25 vs embedding weight in HybridRetriever.
    keyword_threshold:
        Minimum keyword-relevance score for KeywordFilter to pass a sentence.
    """

    def __init__(
        self,
        cache_ttl: Optional[float] = _DEFAULT_CACHE_TTL,
        summarize_ratio: float = 0.3,
        retriever_alpha: float = 0.5,
        keyword_threshold: float = 0.05,
    ) -> None:
        self._preprocessor = Preprocessor()
        self._summarizer = ExtractiveSummarizer(max_ratio=summarize_ratio)
        self._retriever = HybridRetriever(alpha=retriever_alpha)
        self._structure_validator = EssayStructureValidator()
        self._readability = ReadabilityAnalyzer()
        self._critic = RuleBasedCritic()
        self._citation_mgr = CitationManager()
        self._keyword_filter = KeywordFilter(threshold=keyword_threshold)
        self._cache = CacheManager(default_ttl=cache_ttl, namespace="nlp_pipeline")

    # ------------------------------------------------------------------
    # Source preprocessing (parallel, async-safe)
    # ------------------------------------------------------------------

    async def preprocess_sources(
        self,
        sources: list[dict],
        topic: str,
        queries: Optional[list[str]] = None,
    ) -> list[dict]:
        """Enrich source abstracts with cleaned + summarised text.

        Runs each source independently in the thread-pool so they execute
        in parallel.  The original source dicts are **not** mutated; new
        dicts with an added ``processed_abstract`` key are returned.

        Parameters
        ----------
        sources:
            List of source dicts (each with an ``abstract`` key).
        topic:
            Essay topic string used for keyword filtering.
        queries:
            Optional additional context queries for keyword extraction.
        """
        if not sources:
            return sources

        keywords = self._keyword_filter.extract_keywords(
            topic, extra_context=queries or []
        )
        loop = asyncio.get_event_loop()

        # Submit one task per source to the thread-pool
        futures = [
            loop.run_in_executor(
                _NLP_EXECUTOR,
                self._process_single_source,
                src,
                topic,
                keywords,
            )
            for src in sources
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

        enriched: list[dict] = []
        for src, res in zip(sources, results):
            if isinstance(res, Exception):
                logger.debug("NLPPipeline.preprocess_sources: error on source %r: %s", src.get("title"), res)
                enriched.append(src)
            else:
                enriched.append(res)
        return enriched

    def _process_single_source(
        self,
        src: dict,
        topic: str,
        keywords: list[str],
    ) -> dict:
        """Synchronous single-source preprocessing (runs in thread-pool)."""
        abstract = (src.get("abstract") or "").strip()
        if not abstract:
            return {**src, "processed_abstract": ""}

        # Check cache
        cache_key = self._cache.cache_key("source_abstract", topic, abstract[:80])
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**src, "processed_abstract": cached}

        try:
            # 1. Clean
            cleaned = self._preprocessor.clean_text(abstract)
            # 2. Summarise (≥ 70 % reduction)
            summarised = self._summarizer.summarize(cleaned, topic=topic)
            # 3. Keyword filter
            filtered = self._keyword_filter.filter_sentences(summarised, keywords)
            result = filtered or summarised or cleaned

            self._cache.set(cache_key, result)
            return {**src, "processed_abstract": result}
        except Exception as exc:
            logger.debug("NLPPipeline source preprocessing failed: %s", exc)
            return {**src, "processed_abstract": abstract[:300]}

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def retrieve_top_chunks(
        self,
        query: str,
        sources: list[dict],
        top_k: int = 5,
        text_field: str = "processed_abstract",
    ) -> list[dict]:
        """Return the top-*k* most relevant sources for *query*.

        Uses :class:`HybridRetriever` (BM25 + TF-IDF cosine / SBERT) to rank
        sources by the text in ``text_field`` (falls back to ``abstract``).

        Parameters
        ----------
        query:
            The section query or topic string.
        sources:
            Candidate source dicts.
        top_k:
            Number of sources to return.
        text_field:
            The source dict field to index.  Defaults to ``processed_abstract``
            (populated by :meth:`preprocess_sources`).
        """
        if not sources or not query:
            return sources[:top_k]

        texts = [
            (src.get(text_field) or src.get("abstract") or src.get("title") or "")
            for src in sources
        ]
        metadata = [src for src in sources]

        try:
            chunks = self._retriever.retrieve(query, texts, metadata=metadata, top_k=top_k)
            return [c.metadata for c in chunks]
        except Exception as exc:
            logger.debug("HybridRetriever failed: %s; returning head of list", exc)
            return sources[:top_k]

    # ------------------------------------------------------------------
    # Essay analysis (async, post-write)
    # ------------------------------------------------------------------

    async def analyze_essay(
        self,
        sections: dict[str, str],
        topic: str,
    ) -> dict:
        """Run structure, readability, and critic analysis on the full essay.

        Parameters
        ----------
        sections:
            Dict mapping section_key → section_text.
        topic:
            Essay topic string.

        Returns
        -------
        dict
            ``nlp_analysis`` dict with keys:
            ``structure``, ``readability``, ``critic``.
        """
        if not sections:
            return {}

        full_text = "\n\n".join(sections.values())
        loop = asyncio.get_event_loop()

        # Run all three validators in parallel
        structure_future = loop.run_in_executor(
            _NLP_EXECUTOR,
            self._structure_validator.validate,
            full_text,
        )
        readability_future = loop.run_in_executor(
            _NLP_EXECUTOR,
            self._readability.analyze,
            full_text,
        )
        critic_future = loop.run_in_executor(
            _NLP_EXECUTOR,
            self._critic.critique,
            full_text,
        )

        structure_result, readability_result, critic_result = await asyncio.gather(
            structure_future, readability_future, critic_future,
            return_exceptions=True,
        )

        # Per-section readability (also in parallel)
        section_readability: dict[str, dict] = {}
        if isinstance(readability_result, Exception):
            logger.debug("ReadabilityAnalyzer failed: %s", readability_result)
            readability_result = None

        per_section_futures = {
            key: loop.run_in_executor(_NLP_EXECUTOR, self._readability.analyze, text)
            for key, text in sections.items()
        }
        if per_section_futures:
            per_results = await asyncio.gather(
                *per_section_futures.values(), return_exceptions=True
            )
            for key, res in zip(per_section_futures.keys(), per_results):
                if isinstance(res, Exception):
                    continue
                section_readability[key] = {
                    "flesch_score": res.flesch_score,
                    "avg_sentence_length": res.avg_sentence_length,
                    "passive_voice_ratio": res.passive_voice_ratio,
                    "grade": res.grade,
                    "flagged_sentences": res.flagged_sentences,
                }

        return {
            "structure": (
                {
                    "score": structure_result.score,
                    "present": structure_result.present,
                    "missing": structure_result.missing,
                    "argument_count": structure_result.argument_count,
                    "details": structure_result.details,
                }
                if not isinstance(structure_result, Exception)
                else {}
            ),
            "readability": (
                {
                    "full_essay": {
                        "flesch_score": readability_result.flesch_score,
                        "avg_sentence_length": readability_result.avg_sentence_length,
                        "passive_voice_ratio": readability_result.passive_voice_ratio,
                        "grade": readability_result.grade,
                        "flagged_sentences": readability_result.flagged_sentences,
                    },
                    "sections": section_readability,
                }
                if readability_result is not None
                else {}
            ),
            "critic": (
                {
                    "issue_count": critic_result.issue_count,
                    "severity": critic_result.severity,
                    "issues": critic_result.issues,
                }
                if not isinstance(critic_result, Exception)
                else {}
            ),
        }

    # ------------------------------------------------------------------
    # Citation validation
    # ------------------------------------------------------------------

    def validate_citations(
        self,
        sources: list[dict],
        style: str = "harvard",
    ) -> dict:
        """Validate and format citations from source metadata.

        Returns
        -------
        dict
            ``{valid_count, invalid_count, citations, bibliography}``
        """
        if not sources:
            return {
                "valid_count": 0,
                "invalid_count": 0,
                "citations": [],
                "bibliography": "",
            }

        citations = self._citation_mgr.process_sources(sources)
        bib = self._citation_mgr.bibliography(citations, style=style)

        valid = [c for c in citations if c.is_valid]
        invalid = [c for c in citations if not c.is_valid]

        return {
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "citations": [
                {
                    "title": c.title,
                    "apa": c.apa,
                    "harvard": c.harvard,
                    "in_text_apa": c.in_text_apa,
                    "in_text_harvard": c.in_text_harvard,
                    "is_valid": c.is_valid,
                    "validation_issues": c.validation_issues,
                }
                for c in citations
            ],
            "bibliography": bib,
        }
