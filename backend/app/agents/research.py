import json
import logging
import re
import time
import httpx
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text
from app.core.config import settings

logger = logging.getLogger(__name__)


MOCK_SOURCES = [
    {
        "title": "Deep Learning: A Review",
        "authors": ["LeCun, Y.", "Bengio, Y.", "Hinton, G."],
        "year": 2015,
        "abstract": "A comprehensive review of deep learning methods and applications.",
        "url": "https://www.nature.com/articles/nature14539",
        "doi": "10.1038/nature14539",
        "source": "web",
    },
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "year": 2017,
        "abstract": "We propose a new simple network architecture, the Transformer.",
        "url": "https://arxiv.org/abs/1706.03762",
        "doi": "10.48550/arXiv.1706.03762",
        "source": "arxiv",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M."],
        "year": 2019,
        "abstract": "We introduce BERT, a language representation model.",
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
        queries = input_data.get("queries", [])
        sources_list = input_data.get("sources", settings.RESEARCH_SOURCES)
        topic = input_data.get("topic", "")

        if not queries and topic:
            queries = [f"{topic} overview", f"{topic} methods"]

        # Use LLM to refine/expand search queries only when the existing queries
        # are short or generic — this avoids burning tokens when the planner
        # already produced focused queries.
        needs_refinement = len(queries) < 3 or all(self._is_generic_query(q) for q in queries[:3])
        if is_llm_available() and queries and needs_refinement:
            queries = await self._llm_refine_queries(queries, topic, project_id, db)

        all_sources = []
        for source in sources_list:
            if source == "arxiv":
                srcs = await self._search_arxiv(queries[:2], project_id, db)
            elif source == "semantic_scholar":
                srcs = await self._search_semantic_scholar(queries[:3], project_id, db)
            elif source == "web":
                srcs = await self._search_crossref(queries[:3], project_id, db)
            else:
                srcs = self._mock_sources(source, queries)
            all_sources.extend(srcs)

        # Deduplicate by DOI
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s.get("doi") or s.get("url") or s.get("title")
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        ranked_sources = self._rank_sources(unique_sources, topic, queries)

        source_breakdown = {}
        for src in ranked_sources:
            src_name = src.get("source", "unknown")
            source_breakdown[src_name] = source_breakdown.get(src_name, 0) + 1

        # Use LLM to synthesize a research summary from gathered sources
        summary = ""
        if is_llm_available() and ranked_sources:
            summary = await self._llm_summarize(ranked_sources, topic, project_id, db)

        result = {
            "sources": ranked_sources,
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

    async def _llm_refine_queries(self, queries: list, topic: str, project_id: str, db) -> list:
        """Use LLM to generate focused academic search queries from the input list."""
        try:
            queries_text = "\n".join(f"- {q}" for q in queries[:5])
            prompt = (
                f"You are a research librarian. Given the following search queries for the topic '{topic}', "
                "return an improved JSON list of up to 5 precise academic search queries that will find "
                "high-quality peer-reviewed papers. Output ONLY a JSON array of strings.\n\n"
                f"Original queries:\n{queries_text}"
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
            )
            refined = json.loads(content)
            if isinstance(refined, list) and all(isinstance(q, str) for q in refined):
                return refined
            logger.warning("LLM query refinement returned unexpected format; using original queries.")
        except Exception as exc:
            logger.warning("LLM query refinement failed (%s); using original queries.", exc)
        return queries

    async def _llm_summarize(self, sources: list, topic: str, project_id: str, db) -> str:
        """Use LLM to synthesize a structured research summary from gathered sources."""
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
                f"You are preparing research notes for an academic essay on '{topic}'. "
                "Using only the provided sources, write a concise but information-dense synthesis in 4-6 sentences. "
                "Your synthesis must include: (1) key themes, (2) 2-3 concrete findings/trends, "
                "(3) at least one research gap or unresolved debate. "
                "Use citation markers like [1], [2] that correspond to source order in the list.\n\n"
                f"Sources: {sources_text}"
            )
            return await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                max_tokens=600,
            )
        except Exception:
            return ""

    async def _search_arxiv(self, queries: list, project_id: str, db) -> list:
        results = []
        for query in queries[:2]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://export.arxiv.org/api/query",
                        params={"search_query": f"all:{query}", "max_results": 3},
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

    async def _search_semantic_scholar(self, queries: list, project_id: str, db) -> list:
        results = []
        for query in queries[:3]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": query,
                            "limit": 5,
                            "fields": "title,abstract,year,url,authors,externalIds,venue",
                        },
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(
                    db,
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    "GET",
                    self.name,
                    duration,
                    resp.status_code,
                )

                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    for paper in payload.get("data", []):
                        authors = [a.get("name") for a in paper.get("authors", []) if a.get("name")]
                        external_ids = paper.get("externalIds") or {}
                        doi = external_ids.get("DOI") or ""
                        results.append(
                            {
                                "title": paper.get("title") or "Unknown",
                                "authors": authors[:5],
                                "year": paper.get("year") or 2024,
                                "abstract": (paper.get("abstract") or "")[:800],
                                "url": paper.get("url") or "",
                                "doi": doi,
                                "venue": paper.get("venue") or "",
                                "source": "semantic_scholar",
                            }
                        )
                else:
                    results.extend(self._mock_sources("semantic_scholar", [query]))
            except Exception:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(
                    db,
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    "GET",
                    self.name,
                    duration,
                    500,
                )
                results.extend(self._mock_sources("semantic_scholar", [query]))
        return results

    async def _search_crossref(self, queries: list, project_id: str, db) -> list:
        results = []
        for query in queries[:3]:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.get(
                        "https://api.crossref.org/works",
                        params={
                            "query": query,
                            "rows": 5,
                            "sort": "relevance",
                        },
                        headers={"User-Agent": "EssayWritingAgent/1.0 (research module)"},
                    )
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(
                    db,
                    "https://api.crossref.org/works",
                    "GET",
                    self.name,
                    duration,
                    resp.status_code,
                )

                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    items = payload.get("message", {}).get("items", [])
                    for item in items:
                        author_items = item.get("author", []) or []
                        authors = []
                        for a in author_items:
                            family = a.get("family") or ""
                            given = a.get("given") or ""
                            if family or given:
                                label = f"{family}, {given}".strip(", ")
                                authors.append(label)

                        title_list = item.get("title", []) or []
                        abstract = item.get("abstract") or ""
                        if abstract:
                            abstract = abstract.replace("<jats:p>", "").replace("</jats:p>", "")

                        year = 2024
                        issued = item.get("issued", {}).get("date-parts", [])
                        if issued and issued[0]:
                            year = issued[0][0]

                        doi = item.get("DOI") or ""
                        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

                        results.append(
                            {
                                "title": title_list[0] if title_list else "Unknown",
                                "authors": authors[:5],
                                "year": year,
                                "abstract": abstract[:800],
                                "url": url,
                                "doi": doi,
                                "source": "web",
                            }
                        )
                else:
                    results.extend(self._mock_sources("web", [query]))
            except Exception:
                duration = (time.monotonic() - start) * 1000
                await self._log_api_call(
                    db,
                    "https://api.crossref.org/works",
                    "GET",
                    self.name,
                    duration,
                    500,
                )
                results.extend(self._mock_sources("web", [query]))
        return results

    def _parse_arxiv(self, xml_text: str) -> list:
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
                published_el = entry.find("atom:published", ns)
                year = int(published_el.text[:4]) if published_el is not None else 2024
                url = id_el.text if id_el is not None else ""
                doi = url.replace("http://arxiv.org/abs/", "10.48550/arXiv.") if url else ""
                results.append({
                    "title": title_el.text.strip() if title_el is not None else "Unknown",
                    "authors": authors[:3],
                    "year": year,
                    "abstract": summary_el.text.strip()[:300] if summary_el is not None else "",
                    "url": url,
                    "doi": doi,
                    "source": "arxiv",
                })
        except Exception:
            pass
        return results

    def _mock_sources(self, source: str, queries: list) -> list:
        return [dict(s, source=source) for s in MOCK_SOURCES[:2]]
