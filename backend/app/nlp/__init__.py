"""Non-LLM NLP optimisation layer.

All components in this module are entirely rule-based or use lightweight
statistical models (TF-IDF, BM25) — **no LLM calls are made here**.

Typical pipeline flow
---------------------
raw documents (source abstracts)
  → Preprocessor         — clean + chunk + section-detect
  → ExtractiveSummarizer — compact to top-N sentences (≥70% reduction)
  → HybridRetriever      — rank chunks by BM25 + embedding similarity
  → KeywordFilter        — filter sentences not relevant to topic
  → (LLM Writer input — smaller, more focused context)
  → (LLM output)
  → EssayStructureValidator — check intro / thesis / args / conclusion
  → ReadabilityAnalyzer     — Flesch ease, sentence length, passive voice
  → RuleBasedCritic         — repeated phrases, weak arguments, no evidence
  → CitationManager         — validate + format references
  → CacheManager            — persist results to avoid recomputation
"""

from app.nlp.preprocessor import Preprocessor
from app.nlp.summarizer import ExtractiveSummarizer
from app.nlp.retriever import HybridRetriever
from app.nlp.validators import EssayStructureValidator, ReadabilityAnalyzer, RuleBasedCritic
from app.nlp.citation_manager import CitationManager
from app.nlp.keyword_filter import KeywordFilter
from app.nlp.cache_manager import CacheManager
from app.nlp.pipeline import NLPPipeline

__all__ = [
    "Preprocessor",
    "ExtractiveSummarizer",
    "HybridRetriever",
    "EssayStructureValidator",
    "ReadabilityAnalyzer",
    "RuleBasedCritic",
    "CitationManager",
    "KeywordFilter",
    "CacheManager",
    "NLPPipeline",
]
