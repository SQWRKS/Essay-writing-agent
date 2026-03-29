import json
import time
import httpx
from app.agents.base import AgentBase
from app.core.config import settings


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

        result = {"sources": unique_sources, "total_found": len(unique_sources)}
        await self._update_agent_state(db, project_id, "completed", result)
        return result

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
