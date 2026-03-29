"""
WebSearchAgent — non-LLM web research using free public APIs.

Architecture
------------
1. **Parallel HTTP fetching** (asyncio.gather):
   - DuckDuckGo Instant Answers API (summaries / definitions)
   - Wikipedia Search + Extract API  (encyclopedic background)

2. **Preprocessing pipeline** (entirely rule-based, no LLM):
   a. Clean raw text (strip HTML entities, normalize whitespace)
   b. Tokenize into sentences
   c. Score sentences by keyword overlap with the query topic
   d. Select top-K sentences within a character budget (~400 chars)
   The result is a compact, information-dense "abstract" for each source.

3. **Output format** is fully compatible with the existing source pipeline
   (VerificationAgent, _build_section_evidence, _rank_sources) so sources
   can be merged transparently with academic results.

No API keys are required.  All external calls are wrapped with timeouts and
have graceful fallbacks so a network failure never aborts the pipeline.
"""

import asyncio
import html
import json
import logging
import re
import time
from typing import Optional

import httpx

from app.agents.base import AgentBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DDGO_URL = "https://api.duckduckgo.com/"
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_HTTP_TIMEOUT = 8.0          # seconds per request
_MAX_ABSTRACT_CHARS = 400    # characters to keep per preprocessed abstract
_MIN_SENTENCE_LEN = 35       # sentences shorter than this are discarded
_MAX_SENTENCE_LEN = 380      # sentences longer than this are truncated
_MAX_SOURCES_PER_QUERY = 3   # Wikipedia pages per query
_TOP_QUERIES = 3             # max queries sent to each API


class WebSearchAgent(AgentBase):
    """Non-LLM web research agent.

    Searches DuckDuckGo and Wikipedia in parallel, preprocesses the raw text
    to extract compact informative sentences, and returns source records
    compatible with the rest of the pipeline.
    """

    name = "web_search"

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        queries: list[str] = input_data.get("queries", [])
        topic: str = input_data.get("topic", "")

        if not queries and topic:
            queries = [topic, f"{topic} overview"]
        if not queries:
            result: dict = {"sources": [], "total_found": 0, "source_breakdown": {}, "summary": ""}
            await self._update_agent_state(db, project_id, "completed", result)
            return result

        # Run DuckDuckGo and Wikipedia searches in parallel
        ddg_coro = self._search_duckduckgo(queries[:_TOP_QUERIES], topic, db)
        wiki_coro = self._search_wikipedia(queries[:_TOP_QUERIES], topic, db)
        raw_results = await asyncio.gather(ddg_coro, wiki_coro, return_exceptions=True)

        all_sources: list[dict] = []
        for r in raw_results:
            if isinstance(r, list):
                all_sources.extend(r)
            elif isinstance(r, Exception):
                logger.warning("WebSearchAgent sub-search failed: %s", r)

        # Deduplicate by URL or title
        unique = self._deduplicate(all_sources)

        result = {
            "sources": unique,
            "total_found": len(unique),
            "source_breakdown": {"web_search": len(unique)},
            "summary": "",
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    # -----------------------------------------------------------------------
    # DuckDuckGo Instant Answers
    # -----------------------------------------------------------------------

    async def _search_duckduckgo(self, queries: list[str], topic: str, db) -> list[dict]:
        """Query DuckDuckGo Instant Answers for each query string.

        Returns at most one high-quality source per query (the abstract, if
        available) plus up to 3 related-topic snippets.
        """
        results: list[dict] = []
        current_year = time.gmtime().tm_year

        for query in queries:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                    resp = await client.get(
                        _DDGO_URL,
                        params={
                            "q": query,
                            "format": "json",
                            "no_html": "1",
                            "skip_disambig": "1",
                        },
                        headers={"User-Agent": "EssayWritingAgent/1.0 (research module)"},
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, _DDGO_URL, "GET", self.name, duration, resp.status_code)

                if resp.status_code != 200 or not resp.content:
                    continue

                data = resp.json()

                # Primary abstract (usually from Wikipedia)
                abstract_text = (data.get("AbstractText") or "").strip()
                abstract_url = (data.get("AbstractURL") or "").strip()
                abstract_title = (data.get("Heading") or query).strip()

                if abstract_text and abstract_url:
                    preprocessed = self._preprocess_text(abstract_text, topic)
                    if preprocessed:
                        results.append({
                            "title": abstract_title,
                            "authors": [],
                            "year": current_year,
                            "abstract": preprocessed,
                            "url": abstract_url,
                            "doi": "",
                            "source": "web_search",
                        })

                # Related topics as supplementary sources
                for related in (data.get("RelatedTopics") or [])[:3]:
                    if not isinstance(related, dict):
                        continue
                    text = (related.get("Text") or "").strip()
                    url = (related.get("FirstURL") or "").strip()
                    if not text or not url:
                        continue
                    preprocessed = self._preprocess_text(text, topic)
                    if preprocessed:
                        results.append({
                            "title": self._title_from_url(url) or query,
                            "authors": [],
                            "year": current_year,
                            "abstract": preprocessed,
                            "url": url,
                            "doi": "",
                            "source": "web_search",
                        })

            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, _DDGO_URL, "GET", self.name, duration, 500)
                logger.debug("DuckDuckGo search failed for '%s': %s", query, exc)

        return results

    # -----------------------------------------------------------------------
    # Wikipedia Search + Extract
    # -----------------------------------------------------------------------

    async def _search_wikipedia(self, queries: list[str], topic: str, db) -> list[dict]:
        """Search Wikipedia and extract intro text for each matching page.

        For each query we:
        1. Search Wikipedia for up to 3 matching page titles
        2. Batch-fetch the intro extract for all found titles
        3. Preprocess each extract to a compact abstract
        """
        results: list[dict] = []
        current_year = time.gmtime().tm_year
        seen_titles: set[str] = set()

        for query in queries:
            # Step 1: search
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                    search_resp = await client.get(
                        _WIKI_SEARCH_URL,
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "format": "json",
                            "srlimit": _MAX_SOURCES_PER_QUERY,
                            "srprop": "snippet|titlesnippet",
                        },
                        headers={"User-Agent": "EssayWritingAgent/1.0"},
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, _WIKI_SEARCH_URL, "GET", self.name, duration, search_resp.status_code)

                if search_resp.status_code != 200 or not search_resp.content:
                    continue

                search_data = search_resp.json()
                titles = [
                    hit["title"]
                    for hit in search_data.get("query", {}).get("search", [])
                    if hit.get("title") and hit["title"] not in seen_titles
                ]
                if not titles:
                    continue

                # Step 2: batch-fetch extracts for all titles from this query
                start2 = time.monotonic()
                async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                    extract_resp = await client.get(
                        _WIKI_SEARCH_URL,
                        params={
                            "action": "query",
                            "prop": "extracts",
                            "exintro": "true",
                            "explaintext": "true",
                            "titles": "|".join(titles),
                            "format": "json",
                            "redirects": "1",
                        },
                        headers={"User-Agent": "EssayWritingAgent/1.0"},
                    )
                duration2 = (time.monotonic() - start2) * 1000
                await self._log_api_call(db, _WIKI_SEARCH_URL, "GET", self.name, duration2, extract_resp.status_code)

                if extract_resp.status_code != 200 or not extract_resp.content:
                    continue

                extract_data = extract_resp.json()
                pages = extract_data.get("query", {}).get("pages", {})

                for page in pages.values():
                    title = (page.get("title") or "").strip()
                    extract = (page.get("extract") or "").strip()
                    page_id = page.get("pageid", -1)

                    # Skip missing pages (negative page IDs) or already processed
                    if page_id < 0 or not title or not extract or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    preprocessed = self._preprocess_text(extract, topic)
                    if not preprocessed:
                        continue

                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    results.append({
                        "title": title,
                        "authors": [],
                        "year": current_year,
                        "abstract": preprocessed,
                        "url": url,
                        "doi": "",
                        "source": "web_search",
                    })

            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, _WIKI_SEARCH_URL, "GET", self.name, duration, 500)
                logger.debug("Wikipedia search failed for '%s': %s", query, exc)

        return results

    # -----------------------------------------------------------------------
    # Preprocessing pipeline (no LLM)
    # -----------------------------------------------------------------------

    def _preprocess_text(self, text: str, topic: str, max_chars: int = _MAX_ABSTRACT_CHARS) -> str:
        """Extract compact, informative sentences from raw web text.

        Pipeline:
        1. Decode HTML entities and strip markup
        2. Split into candidate sentences
        3. Score each sentence by keyword overlap with the topic
        4. Select top-scoring sentences within the char budget
        5. Return the joined result

        Returns an empty string if no useful content is found.
        """
        if not text:
            return ""

        cleaned = self._clean_text(text)
        sentences = self._split_sentences(cleaned)
        if not sentences:
            return ""

        topic_terms = self._keyword_set(topic)
        scored: list[tuple[float, str]] = []

        for raw_sent in sentences:
            sent = raw_sent.strip()
            if len(sent) < _MIN_SENTENCE_LEN:
                continue
            if len(sent) > _MAX_SENTENCE_LEN:
                sent = sent[:_MAX_SENTENCE_LEN].rstrip() + "…"
            score = self._sentence_score(sent, topic_terms)
            scored.append((score, sent))

        if not scored:
            return ""

        # Sort descending by score, then pick sentences within budget
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: list[str] = []
        total_chars = 0
        for _, sent in scored:
            if total_chars + len(sent) + 1 > max_chars:
                break
            selected.append(sent)
            total_chars += len(sent) + 1
            if len(selected) >= 3:
                break

        return " ".join(selected).strip()

    def _clean_text(self, text: str) -> str:
        """Strip HTML markup, decode entities, and normalise whitespace."""
        # Decode HTML entities (e.g. &amp; → &, &#160; → nbsp)
        text = html.unescape(text or "")
        # Remove XML/HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove Wikipedia-style section headers (== Heading ==)
        text = re.sub(r"={2,}[^=]+=+", " ", text)
        # Collapse multiple whitespace / newlines into single spaces
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using punctuation boundaries."""
        # Split on sentence-ending punctuation followed by whitespace + capital
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)
        # Further split on " . " patterns that slip through
        sentences: list[str] = []
        for part in parts:
            sub = re.split(r"\.\s{2,}", part)
            sentences.extend(sub)
        return [s.strip() for s in sentences if s.strip()]

    def _keyword_set(self, text: str) -> set[str]:
        """Return meaningful lowercase tokens from text (≥4 chars, alphabetic)."""
        return {
            tok.lower()
            for tok in re.findall(r"\b[a-zA-Z]{4,}\b", text or "")
            if tok.lower() not in _STOP_WORDS
        }

    def _sentence_score(self, sentence: str, topic_terms: set[str]) -> float:
        """Score a sentence by keyword overlap with topic terms.

        Score = (matching_terms / max(1, min(topic_terms, 10))) + length_bonus
        The length bonus slightly rewards sentences of 60-200 chars (information-
        dense range) over very short or very long sentences.
        """
        if not topic_terms:
            # No topic context: prefer sentences of moderate length
            return 0.5 + _length_bonus(len(sentence))

        sent_terms = self._keyword_set(sentence)
        overlap = len(topic_terms & sent_terms)
        overlap_score = overlap / max(1, min(len(topic_terms), 10))
        return overlap_score + _length_bonus(len(sentence))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _deduplicate(self, sources: list[dict]) -> list[dict]:
        """Remove duplicate sources by URL or title."""
        seen: set[str] = set()
        unique: list[dict] = []
        for src in sources:
            key = (src.get("url") or "").rstrip("/") or src.get("title", "")
            if key and key not in seen:
                seen.add(key)
                unique.append(src)
        return unique

    @staticmethod
    def _title_from_url(url: str) -> Optional[str]:
        """Extract a human-readable title from a URL's last path segment."""
        if not url:
            return None
        segment = url.rstrip("/").split("/")[-1]
        # Replace underscores/hyphens with spaces and title-case
        title = re.sub(r"[_\-]+", " ", segment)
        title = re.sub(r"%20", " ", title)
        return title.strip().title() or None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _length_bonus(char_count: int) -> float:
    """Small bonus that peaks for sentences 60-200 chars long."""
    if 60 <= char_count <= 200:
        return 0.1
    if 35 <= char_count < 60:
        return 0.05
    return 0.0


# Common English stopwords to exclude from keyword scoring
_STOP_WORDS: frozenset[str] = frozenset({
    "about", "above", "after", "again", "against", "also", "been", "before",
    "being", "between", "both", "each", "from", "further", "have", "having",
    "here", "into", "more", "most", "other", "over", "same", "shan", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "until", "very", "were",
    "what", "when", "where", "which", "while", "with", "would", "your",
    "just", "like", "make", "many", "much", "only", "such", "than", "very",
    "well", "will", "with",
})
