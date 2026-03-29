import json
import time
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.core.config import settings


SECTION_TEMPLATES = {
    "introduction": (
        "This paper examines {topic}. The significance of this research lies in its potential to advance "
        "our understanding of key concepts and methodologies. In recent years, {topic} has gained considerable "
        "attention from the scientific community due to its broad applicability and transformative potential. "
        "This work aims to provide a comprehensive analysis and contribute novel insights to the field."
    ),
    "literature_review": (
        "A substantial body of literature exists on {topic}. Early foundational works established the "
        "theoretical underpinnings, while more recent studies have expanded the scope significantly. "
        "Researchers have explored multiple dimensions including theoretical frameworks, empirical validations, "
        "and practical applications. Notable contributions have shaped the current understanding of the domain."
    ),
    "methodology": (
        "This study employs a rigorous methodological framework to investigate {topic}. The research design "
        "integrates both qualitative and quantitative approaches to ensure comprehensive coverage. Data collection "
        "followed established protocols with appropriate controls. Statistical analysis was conducted using "
        "validated tools and methods appropriate to the research questions."
    ),
    "results": (
        "The analysis of {topic} yielded several significant findings. The data demonstrate clear patterns "
        "consistent with the hypothesized relationships. Quantitative measures show statistically significant "
        "outcomes across multiple dimensions. These results provide empirical support for the theoretical "
        "framework proposed in the methodology section."
    ),
    "discussion": (
        "The findings regarding {topic} have important implications for both theory and practice. The results "
        "align with and extend prior work in meaningful ways. Several unexpected patterns emerged that warrant "
        "further investigation. The limitations of this study should be considered when interpreting results, "
        "and future research directions are identified."
    ),
    "conclusion": (
        "This study has provided a comprehensive examination of {topic}. The key contributions include a "
        "refined theoretical framework, empirical evidence supporting core hypotheses, and practical guidelines "
        "for application. Future research should build on these findings to further advance the field."
    ),
}


class WriterAgent(AgentBase):
    name = "writer"

    def _first_sentence(self, text: str, max_words: int = 24) -> str:
        if not text:
            return ""
        cleaned = " ".join(str(text).replace("\n", " ").split()).strip()
        if not cleaned:
            return ""

        for sep in [". ", "? ", "! "]:
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0].strip()
                break

        words = cleaned.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]).rstrip(".,;:") + "..."
        return cleaned.rstrip(".,;:") + "."

    def _build_sources_digest(self, sources: list[dict], limit: int = 8) -> str:
        lines = []
        for idx, src in enumerate(sources[:limit], 1):
            title = src.get("title", "Unknown source")
            year = src.get("year", "n.d.")
            source_name = src.get("source", "unknown")
            abstract = (src.get("abstract") or src.get("abstract_excerpt") or "").strip()
            if len(abstract) > 220:
                abstract = abstract[:220].rstrip() + "..."
            lines.append(f"[{idx}] {title} ({year}, {source_name}) :: {abstract}")
        return "\n".join(lines)

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        section = input_data.get("section", "introduction")
        topic = input_data.get("topic", "the topic")
        word_count_target = input_data.get("word_count", 500)
        research_data = input_data.get("research_data", {})

        if is_llm_available():
            result = await self._llm_write(section, topic, word_count_target, research_data, project_id, db)
        else:
            result = self._template_write(section, topic, word_count_target, research_data)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _template_write(self, section: str, topic: str, word_count_target: int, research_data: dict) -> dict:
        template = SECTION_TEMPLATES.get(section, SECTION_TEMPLATES["introduction"])
        content_parts = [template.format(topic=topic)]

        section_queries = research_data.get("section_queries", [])
        if section_queries:
            query_line = "; ".join(section_queries[:3])
            content_parts.append(
                "This section is guided by targeted questions: "
                f"{query_line}."
            )

        research_summary = self._first_sentence(research_data.get("research_summary", ""), max_words=30)
        if research_summary:
            content_parts.append(f"Current synthesis across the collected studies suggests {research_summary}")

        evidence_pack = research_data.get("evidence_pack", [])
        if evidence_pack:
            evidence_lines = []
            for idx, ev in enumerate(evidence_pack[:4], 1):
                title = ev.get("title", "Unknown source")
                year = ev.get("year", "n.d.")
                source_name = ev.get("source", "source")
                abstract_sentence = self._first_sentence(ev.get("abstract_excerpt", "") or ev.get("abstract", ""))
                if abstract_sentence:
                    evidence_lines.append(
                        f"[{idx}] {title} ({year}, {source_name}) reports that {abstract_sentence}"
                    )
                else:
                    evidence_lines.append(f"[{idx}] {title} ({year}, {source_name}) provides directly relevant evidence")
            content_parts.append("Evidence highlights: " + " ".join(evidence_lines))

        sources = research_data.get("sources", [])
        if sources and not evidence_pack:
            source_lines = []
            for idx, src in enumerate(sources[:4], 1):
                title = src.get("title", "Unknown source")
                year = src.get("year", "n.d.")
                source_lines.append(f"[{idx}] {title} ({year})")
            content_parts.append("Representative literature includes " + "; ".join(source_lines) + ".")

        content_parts.append(
            "A key limitation is that source coverage and reported outcomes vary across studies, so causal claims "
            "should be interpreted with caution and validated against additional datasets where possible."
        )

        content = "\n\n".join(part.strip() for part in content_parts if part and part.strip())

        # Expand with non-repetitive connective sentences to approach the requested length.
        expansion_pool = [
            f"Taken together, the available evidence indicates that research on {topic} is moving from broad conceptual framing toward more testable and comparative analyses.",
            "Across the cited studies, consistency appears strongest in core trends, while differences are most visible in study design, sampled populations, and evaluation criteria.",
            "This section therefore emphasizes convergent findings first, then addresses methodological variance and unresolved questions that limit direct generalization.",
            "Where numerical or methodological claims are reported, they should be read in context of publication year, data provenance, and domain-specific assumptions.",
        ]
        words = content.split()
        idx = 0
        while len(words) < int(word_count_target * 0.82):
            content += "\n\n" + expansion_pool[idx % len(expansion_pool)]
            idx += 1
            words = content.split()

        if len(words) > word_count_target:
            content = " ".join(words[:word_count_target]).strip()

        return {"section": section, "content": content, "word_count": len(content.split())}

    async def _llm_write(self, section: str, topic: str, word_count_target: int, research_data: dict, project_id: str, db) -> dict:
        try:
            evidence_pack = research_data.get("evidence_pack", [])[:10]
            section_queries = research_data.get("section_queries", [])[:6]
            research_summary = research_data.get("research_summary", "")
            sources_summary = self._build_sources_digest(research_data.get("sources", []), limit=10)
            evidence_summary = json.dumps(evidence_pack)
            prompt = (
                f"You are writing the '{section}' section of an academic essay on '{topic}'.\n"
                f"Target length: approximately {word_count_target} words.\n"
                f"Section-specific research queries: {json.dumps(section_queries)}\n\n"
                f"Research synthesis:\n{research_summary}\n\n"
                f"High-priority evidence pack (JSON):\n{evidence_summary}\n\n"
                f"Verified sources digest:\n{sources_summary}\n\n"
                "Requirements:\n"
                "1) Produce coherent academic prose with clear logic and transitions.\n"
                "2) Include concrete, source-grounded claims instead of generic statements.\n"
                "3) Use inline citation markers like [1], [2] for factual claims, methods, and numbers.\n"
                "4) Include at least one limitation or uncertainty where appropriate.\n"
                "5) Do not invent studies or facts outside the provided evidence.\n"
                "Return only the final section text."
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                temperature=0.45,
            )
            return {"section": section, "content": content, "word_count": len(content.split())}
        except Exception:
            return self._template_write(section, topic, word_count_target, research_data)
