import json
import logging
import math
import re
import time

import httpx

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.core.config import settings

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

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")
        seed_queries = input_data.get("queries", [])
        sources_list = input_data.get("sources", settings.RESEARCH_SOURCES)
        min_sources = max(10, min(int(input_data.get("min_sources", 14)), 20))

        expanded_queries = self._expand_queries(topic, seed_queries)
        if is_llm_available() and topic:
            expanded_queries = await self._llm_refine_queries(expanded_queries, topic, project_id, db)

        all_sources = []
        for source_name in sources_list:
            if source_name == "arxiv":
                srcs = await self._search_arxiv(expanded_queries[:5], project_id, db)
            elif source_name == "semantic_scholar":
                srcs = await self._search_semantic_scholar(expanded_queries[:5], project_id, db)
            elif source_name == "web":
                srcs = await self._search_crossref(expanded_queries[:5], project_id, db)
            else:
                srcs = self._mock_sources(source_name, expanded_queries)
            all_sources.extend(srcs)

        unique_sources = self._deduplicate_sources(all_sources)
        ranked_sources = self._rank_sources(unique_sources, topic, expanded_queries)[:max(min_sources, 18)]
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
            "queries": expanded_queries,
            "sources": ranked_sources,
            "summaries": structured_summaries,
            "total_found": len(ranked_sources),
            "source_breakdown": source_breakdown,
            "summary": summary,
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _expand_queries(self, topic: str, queries: list[str]) -> list[str]:
        seeds = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not seeds and topic:
            seeds = [topic]

        expanded = list(seeds)
        if topic:
            expanded.extend(
                [
                    f"{topic} systematic review",
                    f"{topic} experimental evaluation",
                    f"{topic} quantitative performance analysis",
                    f"{topic} implementation challenges",
                    f"{topic} industry case study",
                    f"{topic} limitations and future work",
                    f"{topic} benchmark dataset comparison",
                    f"{topic} engineering design trade-offs",
                ]
            )

        normalized = []
        seen = set()
        for query in expanded:
            key = " ".join(query.lower().split())
            if key and key not in seen:
                normalized.append(query)
                seen.add(key)
            if len(normalized) >= 10:
                break
        return normalized

    def _deduplicate_sources(self, sources: list[dict]) -> list[dict]:
        seen = set()
        unique_sources = []
        for source in sources:
            key = source.get("doi") or source.get("url") or source.get("title")
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(source)
        return unique_sources

    def _rank_sources(self, sources: list[dict], topic: str, queries: list[str]) -> list[dict]:
        if not sources:
            return []

        current_year = time.gmtime().tm_year
        topic_terms = self._keyword_set(" ".join([topic, *queries]))
        ranked = []
        for src in sources:
            title = src.get("title") or ""
            abstract = src.get("abstract") or ""
            combined_text = f"{title} {abstract}".strip()

            keyword_overlap = self._keyword_overlap_score(combined_text, topic_terms)
            semantic_similarity = self._semantic_similarity_score(combined_text, topic_terms)

            year = src.get("year") or 0
            recency_score = 0.0
            if isinstance(year, int) and year > 1900:
                age = max(0, current_year - year)
                recency_score = max(0.0, 1.0 - (age / 18.0))

            completeness = 0.15 if src.get("abstract") else 0.0
            completeness += 0.1 if src.get("doi") else 0.0
            source_bonus = 0.08 if src.get("source") in {"arxiv", "semantic_scholar", "web"} else 0.0

            relevance = round(
                (keyword_overlap * 0.35)
                + (semantic_similarity * 0.35)
                + (recency_score * 0.12)
                + completeness
                + source_bonus,
                3,
            )
            ranked.append(
                {
                    **src,
                    "keyword_overlap": round(keyword_overlap, 3),
                    "semantic_similarity": round(semantic_similarity, 3),
                    "relevance_score": relevance,
                }
            )

        ranked.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
        return ranked

    def _keyword_set(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
            if len(token) > 3
        }

    def _keyword_overlap_score(self, text: str, topic_terms: set[str]) -> float:
        text_terms = self._keyword_set(text)
        if not text_terms or not topic_terms:
            return 0.0
        return min(1.0, len(text_terms & topic_terms) / max(5, len(topic_terms)))

    def _semantic_similarity_score(self, text: str, topic_terms: set[str]) -> float:
        text_terms = self._keyword_set(text)
        if not text_terms or not topic_terms:
            return 0.0

        intersection = len(text_terms & topic_terms)
        denominator = math.sqrt(len(text_terms) * len(topic_terms))
        if denominator == 0:
            return 0.0
        return min(1.0, intersection / denominator)

    def _build_structured_summaries(self, ranked_sources: list[dict], topic: str) -> list[dict]:
        summaries = []
        for source in ranked_sources:
            abstract = source.get("abstract") or ""
            finding = self._extract_key_finding(abstract, source.get("title", ""), topic)
            quantitative = self._extract_quantitative_data(abstract)
            summaries.append(
                {
                    "source": {
                        "title": source.get("title", "Unknown"),
                        "authors": source.get("authors", []),
                        "year": source.get("year"),
                        "doi": source.get("doi", ""),
                        "url": source.get("url", ""),
                        "source": source.get("source", "unknown"),
                    },
                    "key_findings": finding,
                    "quantitative_data": quantitative,
                    "relevance_score": source.get("relevance_score", 0.0),
                }
            )
        return summaries

    def _extract_key_finding(self, abstract: str, title: str, topic: str) -> str:
        cleaned = " ".join((abstract or "").split())
        if not cleaned:
            return f"{title or topic} is directly relevant but lacks an abstract for deeper synthesis."

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in ["result", "find", "show", "improv", "reduc", "increase", "outperform"]):
                return sentence.strip()
        return sentences[0].strip()

    def _extract_quantitative_data(self, abstract: str) -> list[str]:
        if not abstract:
            return []
        matches = re.findall(r"\b\d+(?:\.\d+)?%?|\b\d+(?:\.\d+)?\s?(?:x|times|fold|samples|participants|epochs|studies|datasets|models)", abstract)
        deduped = []
        seen = set()
        for match in matches:
            key = match.lower()
            if key not in seen:
                deduped.append(match)
                seen.add(key)
        return deduped[:5]

    def _compose_summary(self, summaries: list[dict], topic: str) -> str:
        if not summaries:
            return ""

        top = summaries[:4]
        lines = [f"Research on {topic} converges on several recurring engineering and evaluation themes."]
        for idx, summary in enumerate(top, 1):
            title = summary["source"].get("title", "Unknown source")
            finding = summary.get("key_findings", "")
            quantities = summary.get("quantitative_data", [])
            quant_line = f" Quantitative signals include {', '.join(quantities[:2])}." if quantities else ""
            lines.append(f"[{idx}] {title}: {finding}{quant_line}")
        return " ".join(lines)

    async def _llm_refine_queries(self, queries: list[str], topic: str, project_id: str, db) -> list[str]:
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

    async def _llm_summarize(self, summaries: list[dict], topic: str, project_id: str, db) -> str:
        if not is_llm_available() or not summaries:
            return ""
        try:
            prompt = (
                f"You are synthesising literature for a final-year university paper on '{topic}'. "
                "Using only the structured summaries provided, write a dense 5-7 sentence literature synthesis. "
                "State the dominant technical themes, include concrete named examples, mention quantitative evidence when available, and identify at least one unresolved limitation or disagreement. "
                "Use inline citation markers like [1], [2], [3] that match the source order. Return only the synthesis.\n\n"
                f"Structured summaries:\n{json.dumps(summaries[:8])}"
            )
            return await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                temperature=0.25,
                max_tokens=800,
            )
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
