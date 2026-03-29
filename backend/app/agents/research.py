import json
import logging
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

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        queries = input_data.get("queries", [])
        sources_list = input_data.get("sources", settings.RESEARCH_SOURCES)
        topic = input_data.get("topic", "")

        # Use LLM to refine/expand search queries when available
        if is_llm_available() and queries:
            queries = await self._llm_refine_queries(queries, topic, project_id, db)

        all_sources = []
        for source in sources_list:
            if source == "arxiv":
                srcs = await self._search_arxiv(queries[:2], project_id, db)
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

        # Use LLM to synthesize a research summary from gathered sources
        summary = ""
        if is_llm_available() and unique_sources:
            summary = await self._llm_summarize(unique_sources, topic, project_id, db)

        result = {
            "sources": unique_sources,
            "total_found": len(unique_sources),
            "summary": summary,
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

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
        """Use LLM to synthesize a brief research summary from gathered sources."""
        try:
            sources_text = json.dumps(
                [{"title": s.get("title"), "abstract": s.get("abstract", "")[:200]} for s in sources[:5]]
            )
            prompt = (
                f"Based on the following research sources about '{topic}', write a concise 2-3 sentence "
                "synthesis that highlights key themes, findings, and research gaps.\n\n"
                f"Sources: {sources_text}"
            )
            return await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                max_tokens=256,
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
