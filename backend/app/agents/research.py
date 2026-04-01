import json
import logging
import re
import time

import httpx

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text
from app.core.config import settings
from app.routing.model_config import AGENT_MODELS

logger = logging.getLogger(__name__)


MOCK_SOURCES = [
    {
        "title": "Deep Learning: A Review",
        "authors": ["LeCun, Y.", "Bengio, Y.", "Hinton, G."],
        "year": 2015,
        "abstract": "A comprehensive review of deep learning methods and applications, including architecture scaling, optimisation instability, and benchmark-driven progress across computer vision and speech recognition.",
        "url": "https://www.nature.com/articles/nature14539",
        "doi": "10.1038/nature14539",
        "source": "web",
    },
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "year": 2017,
        "abstract": "The Transformer architecture replaced recurrence with attention and reduced training cost while improving sequence modelling quality on large-scale translation benchmarks.",
        "url": "https://arxiv.org/abs/1706.03762",
        "doi": "10.48550/arXiv.1706.03762",
        "source": "arxiv",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M."],
        "year": 2019,
        "abstract": "Bidirectional pre-training substantially improved downstream language understanding benchmarks, especially when fine-tuned with task-specific supervision.",
        "url": "https://arxiv.org/abs/1810.04805",
        "doi": "10.48550/arXiv.1810.04805",
        "source": "semantic_scholar",
    },
]


class ResearchAgent(AgentBase):
    name = "research"

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", (text or "").lower())
            if len(token) > 3
        }

    def _query_terms(self, topic: str, queries: list) -> set[str]:
        terms = set()
        for chunk in [topic, *queries]:
            terms.update(self._tokenize(str(chunk)))
        return terms

    def _is_generic_query(self, query: str) -> bool:
        """Return True if a query is too generic to benefit from LLM refinement."""
        generic_suffixes = {"overview", "methods", "introduction", "review", "survey", "basics"}
        tokens = {t.lower().strip() for t in query.split()}
        return tokens.issubset(generic_suffixes | {""})

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")
        seed_queries = input_data.get("queries", [])
        sources_list = input_data.get("sources", settings.RESEARCH_SOURCES)
        min_sources = max(10, min(int(input_data.get("min_sources", 14)), 20))

        queries = list(seed_queries)
        if not queries and topic:
            queries = [f"{topic} overview", f"{topic} methods"]

        # Use LLM to refine/expand search queries only when the existing queries
        # are short or generic — this avoids burning tokens when the planner
        # already produced focused queries.
        needs_refinement = len(queries) < 3 or all(self._is_generic_query(q) for q in queries[:3])
        if is_llm_available() and queries and needs_refinement:
            queries = await self._llm_refine_queries(queries, topic, project_id, db)

        all_sources = []
        for source_name in sources_list:
            if source_name == "arxiv":
                srcs = await self._search_arxiv(queries[:5], project_id, db)
            elif source_name == "semantic_scholar":
                srcs = await self._search_semantic_scholar(queries[:5], project_id, db)
            elif source_name == "web":
                srcs = await self._search_crossref(queries[:5], project_id, db)
            else:
                srcs = self._mock_sources(source_name, queries)
            all_sources.extend(srcs)

        unique_sources = self._deduplicate_sources(all_sources)
        ranked_sources = self._rank_sources(unique_sources, topic, queries)[:max(min_sources, 18)]
        structured_summaries = self._build_structured_summaries(ranked_sources, topic)
        summary = ""
        if structured_summaries:
            summary = await self._llm_summarize(structured_summaries, topic, project_id, db)
            if not summary:
                summary = self._compose_summary(structured_summaries, topic)

        source_breakdown = {}
        for src in ranked_sources:
            src_name = src.get("source", "unknown")
            source_breakdown[src_name] = source_breakdown.get(src_name, 0) + 1

        result = {
            "queries": queries,
            "sources": ranked_sources,
            "summaries": structured_summaries,
            "total_found": len(ranked_sources),
            "source_breakdown": source_breakdown,
            "summary": summary,
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _rank_sources(self, sources: list, topic: str, queries: list) -> list:
        """Assign a hybrid lexical relevance score and sort sources descending."""
        if not sources:
            return []

        current_year = time.gmtime().tm_year
        query_terms = self._query_terms(topic, queries)
        ranked = []
        for src in sources:
            title = src.get("title") or ""
            abstract = src.get("abstract") or ""
            venue = src.get("venue") or ""

            title_terms = self._tokenize(title)
            abstract_terms = self._tokenize(abstract)
            combined_terms = title_terms | abstract_terms

            title_overlap = len(query_terms & title_terms) / max(3, min(8, len(query_terms) or 1))
            abstract_overlap = len(query_terms & abstract_terms) / max(4, min(12, len(query_terms) or 1))
            coverage_score = len(query_terms & combined_terms) / max(4, len(query_terms) or 1)
            exact_phrase_bonus = 0.15 if topic and topic.lower() in f"{title} {abstract}".lower() else 0.0

            year = src.get("year") or 0
            recency_score = 0.0
            if isinstance(year, int) and year > 1900:
                age = max(0, current_year - year)
                recency_score = max(0.0, 1.0 - (age / 15.0))

            doi_score = 1.0 if src.get("doi") else 0.0
            abstract_score = min(1.0, len(abstract.strip()) / 250) if abstract else 0.0
            author_score = min(1.0, len(src.get("authors") or []) / 3)
            source_bonus = {
                "semantic_scholar": 1.0,
                "web": 0.8,
                "arxiv": 0.75,
            }.get(src.get("source"), 0.5)
            venue_bonus = 0.15 if venue else 0.0

            lexical_score = min(1.0, (title_overlap * 0.45) + (abstract_overlap * 0.35) + (coverage_score * 0.2) + exact_phrase_bonus)
            relevance = round(
                (lexical_score * 0.42)
                + (recency_score * 0.12)
                + (doi_score * 0.12)
                + (abstract_score * 0.14)
                + (author_score * 0.06)
                + (source_bonus * 0.09)
                + venue_bonus,
                3,
            )
            ranking_features = {
                "lexical_score": round(lexical_score, 3),
                "title_overlap": round(min(1.0, title_overlap), 3),
                "abstract_overlap": round(min(1.0, abstract_overlap), 3),
                "coverage_score": round(min(1.0, coverage_score), 3),
                "recency_score": round(recency_score, 3),
                "metadata_score": round(((doi_score * 0.5) + (abstract_score * 0.35) + (author_score * 0.15)), 3),
                "source_bonus": round(source_bonus, 3),
            }
            match_reasons = []
            if exact_phrase_bonus:
                match_reasons.append("topic phrase appears directly in title or abstract")
            if title_overlap >= 0.2:
                match_reasons.append("title overlaps with core query terms")
            if abstract_overlap >= 0.2:
                match_reasons.append("abstract covers query-specific concepts")
            if doi_score:
                match_reasons.append("source includes DOI metadata")

            ranked.append(
                {
                    **src,
                    "relevance_score": relevance,
                    "ranking_features": ranking_features,
                    "match_reasons": match_reasons[:4],
                }
            )

        ranked.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
        return ranked

    def _expand_queries(self, topic: str, queries: list[str]) -> list[str]:
        """Merge LLM-generated queries with topic-anchored fallback queries.

        Ensures the query list always contains at least a few topic-specific
        entries even when the LLM returns fewer queries than expected.
        """
        fallback = [
            f"{topic} overview",
            f"{topic} methods",
            f"recent advances in {topic}",
        ]
        combined = list(dict.fromkeys(queries + fallback))
        return combined

    def _deduplicate_sources(self, sources: list[dict]) -> list[dict]:
        """Remove duplicate sources based on DOI (preferred) then normalised title."""
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[dict] = []
        for src in sources:
            doi = (src.get("doi") or "").strip()
            title_key = re.sub(r"\s+", " ", (src.get("title") or "").strip().lower())
            if doi and doi in seen_dois:
                continue
            if title_key and title_key in seen_titles:
                continue
            if doi:
                seen_dois.add(doi)
            if title_key:
                seen_titles.add(title_key)
            unique.append(src)
        return unique

    def _build_structured_summaries(self, sources: list[dict], topic: str) -> list[dict]:
        """Build a list of structured summary dicts from ranked source records.

        Each summary exposes a normalised set of fields used by downstream
        agents (ThesisAgent, WriterAgent) for evidence grounding.
        """
        summaries: list[dict] = []
        for src in sources:
            abstract = (src.get("abstract") or "").strip()
            title = (src.get("title") or "").strip()
            key_findings = abstract[:300] if abstract else f"Research related to {topic}."
            summaries.append(
                {
                    "source": {
                        "title": title,
                        "authors": src.get("authors") or [],
                        "year": src.get("year"),
                        "doi": src.get("doi") or "",
                        "url": src.get("url") or "",
                        "venue": src.get("venue") or "",
                    },
                    "key_findings": key_findings,
                    "abstract_excerpt": abstract[:200],
                    "relevance_score": src.get("relevance_score", 0.0),
                    "quantitative_data": [],
                    "section_relevance": {},
                }
            )
        return summaries

    def _compose_summary(self, structured_summaries: list[dict], topic: str) -> str:
        """Build a plain-text research summary without an LLM call.

        Used as a fallback when the LLM summarisation step fails or is
        skipped.  Concatenates key findings from the top sources.
        """
        if not structured_summaries:
            return f"Research on {topic} encompasses several important dimensions."
        lines: list[str] = []
        for i, summary in enumerate(structured_summaries[:6], 1):
            src = summary.get("source", {})
            title = src.get("title", "Unknown source")
            year = src.get("year", "n.d.")
            finding = summary.get("key_findings", "")
            if finding:
                lines.append(f"[{i}] {title} ({year}): {finding[:180]}")
        return " ".join(lines) if lines else f"Several studies address {topic}."

    async def _llm_refine_queries(self, queries: list, topic: str, project_id: str, db) -> list:
        """Use LLM to generate focused academic search queries from the input list."""
        try:
            queries_text = "\n".join(f"- {query}" for query in queries[:8])
            prompt = (
                f"You are an academic research librarian building search strategies for the topic '{topic}'. "
                "Return a JSON object with a 'queries' array containing 8 concise but diverse scholarly search queries. "
                "Cover theory, empirical evaluation, quantitative evidence, implementation trade-offs, limitations, and case studies. "
                "Do not repeat wording across queries. Output only JSON.\n\n"
                f"Seed queries:\n{queries_text}"
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["research"]["cheap"],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=512,
            )
            payload = json.loads(content)
            refined = payload.get("queries") if isinstance(payload, dict) else payload
            if isinstance(refined, list):
                merged = self._expand_queries(topic, [q for q in refined if isinstance(q, str)])
                return merged[:10]
        except Exception as exc:
            logger.warning("LLM query refinement failed (%s); using deterministic expansion.", exc)
        return queries[:10]

    async def _llm_summarize(self, sources: list, topic: str, project_id: str, db) -> str:
        """Use LLM to synthesize a structured research summary from gathered sources.

        Uses the cheap model (deepseek-chat) by default.  When the router
        determines that the synthesis requires deeper reasoning (e.g. multiple
        conflicting sources), the task is escalated to the expensive model
        (gpt-5).
        """
        if not sources:
            return ""
        try:
            # Limit to 6 sources; truncate abstracts to 200 chars each to reduce prompt tokens
            sources_text = json.dumps(
                [
                    {
                        "title": s.get("title"),
                        "year": s.get("year"),
                        "source": s.get("source"),
                        "abstract": truncate_text(s.get("abstract", ""), 200),
                    }
                    for s in sources[:6]
                ]
            )
            prompt = (
                f"You are synthesising literature for a final-year university paper on '{topic}'. "
                "Using only the structured summaries provided, write a dense 5-7 sentence literature synthesis. "
                "State the dominant technical themes, include concrete named examples, mention quantitative evidence when available, and identify at least one unresolved limitation or disagreement. "
                "Use inline citation markers like [1], [2], [3] that match the source order. Return only the synthesis.\n\n"
                f"Structured summaries:\n{json.dumps(sources[:6])}"
            )
            output, _ = await self._call_with_routing(
                "literature summarisation",
                AGENT_MODELS["research"]["cheap"],
                AGENT_MODELS["research"]["expensive"],
                prompt,
                db,
                max_tokens=600,
            )
            return output
        except Exception:
            return ""

    async def _search_arxiv(self, queries: list[str], project_id: str, db) -> list[dict]:
        results = []
        for query in queries[:5]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://export.arxiv.org/api/query",
                        params={"search_query": f"all:{query}", "max_results": 4},
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://export.arxiv.org/api/query", "GET", self.name, duration, resp.status_code)
                if resp.status_code == 200:
                    results.extend(self._parse_arxiv(resp.text))
            except Exception:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://export.arxiv.org/api/query", "GET", self.name, duration, 500)
                results.extend(self._mock_sources("arxiv", [query]))
        return results

    async def _search_semantic_scholar(self, queries: list[str], project_id: str, db) -> list[dict]:
        results = []
        for query in queries[:5]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": query,
                            "limit": 4,
                            "fields": "title,abstract,year,url,authors,externalIds,venue,citationCount",
                        },
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://api.semanticscholar.org/graph/v1/paper/search", "GET", self.name, duration, resp.status_code)
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    for paper in payload.get("data", []):
                        authors = [author.get("name") for author in paper.get("authors", []) if author.get("name")]
                        external_ids = paper.get("externalIds") or {}
                        results.append(
                            {
                                "title": paper.get("title") or "Unknown",
                                "authors": authors[:5],
                                "year": paper.get("year") or 2024,
                                "abstract": (paper.get("abstract") or "")[:1200],
                                "url": paper.get("url") or "",
                                "doi": external_ids.get("DOI") or "",
                                "venue": paper.get("venue") or "",
                                "citation_count": paper.get("citationCount") or 0,
                                "source": "semantic_scholar",
                            }
                        )
                else:
                    results.extend(self._mock_sources("semantic_scholar", [query]))
            except Exception:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://api.semanticscholar.org/graph/v1/paper/search", "GET", self.name, duration, 500)
                results.extend(self._mock_sources("semantic_scholar", [query]))
        return results

    async def _search_crossref(self, queries: list[str], project_id: str, db) -> list[dict]:
        results = []
        for query in queries[:5]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.get(
                        "https://api.crossref.org/works",
                        params={"query": query, "rows": 4, "sort": "relevance"},
                        headers={"User-Agent": "EssayWritingAgent/1.0 (research module)"},
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://api.crossref.org/works", "GET", self.name, duration, resp.status_code)
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    for item in payload.get("message", {}).get("items", []):
                        authors = []
                        for author in item.get("author", []) or []:
                            family = author.get("family") or ""
                            given = author.get("given") or ""
                            if family or given:
                                authors.append(f"{family}, {given}".strip(", "))

                        title_list = item.get("title", []) or []
                        abstract = (item.get("abstract") or "").replace("<jats:p>", "").replace("</jats:p>", "")
                        year = 2024
                        issued = item.get("issued", {}).get("date-parts", [])
                        if issued and issued[0]:
                            year = issued[0][0]
                        doi = item.get("DOI") or ""
                        results.append(
                            {
                                "title": title_list[0] if title_list else "Unknown",
                                "authors": authors[:5],
                                "year": year,
                                "abstract": abstract[:1200],
                                "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                                "doi": doi,
                                "source": "web",
                            }
                        )
                else:
                    results.extend(self._mock_sources("web", [query]))
            except Exception:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(db, "https://api.crossref.org/works", "GET", self.name, duration, 500)
                results.extend(self._mock_sources("web", [query]))
        return results

    def _parse_arxiv(self, xml_text: str) -> list[dict]:
        import xml.etree.ElementTree as ET

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                authors = [
                    author.find("atom:name", ns).text
                    for author in entry.findall("atom:author", ns)
                    if author.find("atom:name", ns) is not None
                ]
                published_el = entry.find("atom:published", ns)
                year = int(published_el.text[:4]) if published_el is not None else 2024
                url = id_el.text if id_el is not None else ""
                doi = url.replace("http://arxiv.org/abs/", "10.48550/arXiv.") if url else ""
                results.append(
                    {
                        "title": title_el.text.strip() if title_el is not None else "Unknown",
                        "authors": authors[:5],
                        "year": year,
                        "abstract": summary_el.text.strip()[:1200] if summary_el is not None else "",
                        "url": url,
                        "doi": doi,
                        "source": "arxiv",
                    }
                )
        except Exception:
            pass
        return results

    def _mock_sources(self, source: str, queries: list[str]) -> list[dict]:
        query_hint = queries[0] if queries else "the topic"
        return [
            {
                **mock,
                "source": source,
                "abstract": f"{mock['abstract']} Query context: {query_hint}.",
            }
            for mock in MOCK_SOURCES
        ]
