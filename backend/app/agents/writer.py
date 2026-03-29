import json
import time
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text, quality_max_tokens
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

    def _format_list(self, items: list[str], bullet: str = "- ") -> str:
        return "\n".join(f"{bullet}{item}" for item in items if item)

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

    def _build_sources_digest(self, sources: list[dict], limit: int = 5) -> str:
        lines = []
        for idx, src in enumerate(sources[:limit], 1):
            title = src.get("title", "Unknown source")
            year = src.get("year", "n.d.")
            source_name = src.get("source", "unknown")
            abstract = (src.get("abstract") or src.get("abstract_excerpt") or "").strip()
            if len(abstract) > 100:
                abstract = abstract[:100].rstrip() + "…"
            lines.append(f"[{idx}] {title} ({year}, {source_name}) :: {abstract}")
        return "\n".join(lines)

    def _extract_subheadings(self, content: str, limit: int = 2) -> list[dict]:
        if not content:
            return []
        subheadings: list[dict] = []
        current_title = ""
        current_lines: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                if current_title:
                    subheadings.append({"title": current_title, "content": "\n".join(current_lines).strip()})
                    if len(subheadings) >= limit:
                        return subheadings[:limit]
                current_title = line[3:].strip()
                current_lines = []
            elif current_title:
                current_lines.append(raw_line)

        if current_title and len(subheadings) < limit:
            subheadings.append({"title": current_title, "content": "\n".join(current_lines).strip()})
        return subheadings[:limit]

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        section = input_data.get("section", "introduction")
        topic = input_data.get("topic", "the topic")
        word_count_target = input_data.get("word_count", 500)
        research_data = input_data.get("research_data", {})
        section_plan = input_data.get("section_plan", {})
        feedback = input_data.get("feedback", "")

        if not is_llm_available():
            raise RuntimeError(
                "No LLM provider is configured for WriterAgent. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running the pipeline."
            )

        result = await self._llm_write(section, topic, word_count_target, research_data, section_plan, feedback, project_id, db)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _template_write(
        self,
        section: str,
        topic: str,
        word_count_target: int,
        research_data: dict,
        section_plan: dict | None = None,
        feedback: str = "",
    ) -> dict:
        section_plan = section_plan or {}
        template = SECTION_TEMPLATES.get(section, SECTION_TEMPLATES["introduction"])
        content_parts = [template.format(topic=topic)]

        thesis_goal = section_plan.get("thesis_goal", "")
        if thesis_goal:
            content_parts.append(f"The central claim in this section is that {thesis_goal.strip().rstrip('.')}." )

        must_cover = section_plan.get("must_cover", [])
        if must_cover:
            content_parts.append(
                "The argument develops through "
                + ", ".join(item.strip() for item in must_cover[:4] if str(item).strip())
                + "."
            )

        evidence_requirements = section_plan.get("evidence_requirements", [])
        if evidence_requirements:
            content_parts.append(
                "Evidence in this section is used to substantiate claims about "
                + ", ".join(item.strip() for item in evidence_requirements[:3] if str(item).strip())
                + "."
            )

        writing_directive = section_plan.get("writing_directive", "")
        if writing_directive:
            content_parts.append(
                "The narrative emphasizes a clear argumentative progression from context to evidence and interpretation."
            )

        if feedback:
            content_parts.append(
                "Compared with earlier drafts, this section strengthens specificity, evidence linkage, and analytical depth."
            )

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

    async def _llm_write(
        self,
        section: str,
        topic: str,
        word_count_target: int,
        research_data: dict,
        section_plan: dict | None,
        feedback: str,
        project_id: str,
        db,
    ) -> dict:
        try:
            section_plan = section_plan or {}
            # Token-efficient evidence selection: top 3 items are enough for grounding
            evidence_pack = research_data.get("evidence_pack", [])[:3]
            section_queries = research_data.get("section_queries", [])[:3]
            # Truncate research summary to avoid bloating the prompt
            research_summary = truncate_text(research_data.get("research_summary", ""), 500)
            # Use a tighter sources digest (3 sources, 100-char abstracts)
            sources_summary = self._build_sources_digest(research_data.get("sources", []), limit=3)
            evidence_summary = json.dumps(evidence_pack)
            thesis_goal = section_plan.get("thesis_goal", "")
            must_cover = section_plan.get("must_cover", [])
            evidence_requirements = section_plan.get("evidence_requirements", [])
            writing_directive = section_plan.get("writing_directive", "")
            subheading_hints = section_plan.get("subheading_hints", [])[:2]
            prompt = (
                f"Write the '{section}' section (~{word_count_target} words) of an academic essay on '{topic}'.\n\n"
                f"SECTION OBJECTIVE: {thesis_goal}\n"
                f"MUST COVER: {'; '.join(str(i) for i in must_cover)}\n"
                f"EVIDENCE REQUIREMENTS: {'; '.join(str(i) for i in evidence_requirements)}\n"
                f"WRITING DIRECTIVE: {writing_directive}\n"
                f"SUBHEADINGS (optional, max 2): {', '.join(subheading_hints) if subheading_hints else 'None'}\n\n"
                f"RESEARCH SYNTHESIS:\n{research_summary}\n\n"
                f"EVIDENCE PACK (JSON):\n{evidence_summary}\n\n"
                f"SOURCE DIGEST:\n{sources_summary}\n\n"
                f"REVISION GUIDANCE:\n{truncate_text(feedback, 400) if feedback else 'None'}\n\n"
                "REQUIREMENTS:\n"
                "1) Write coherent academic prose with clear logic and paragraph-to-paragraph transitions.\n"
                "2) Ground every factual or analytical claim in the evidence pack; cite inline as [1], [2].\n"
                "3) Satisfy the thesis goal and all must-cover items for this section.\n"
                "4) Include at least one explicit limitation or uncertainty.\n"
                "5) Do not invent studies or facts not present in the evidence pack.\n"
                "6) If revision guidance is given, directly address each point rather than repeating prior weaknesses.\n"
                "7) Use '## Subheading' markdown only when it improves clarity (max 2).\n"
                "Return only the final section prose."
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                temperature=0.45,
                max_tokens=quality_max_tokens(),
            )
            return {
                "section": section,
                "content": content,
                "word_count": len(content.split()),
                "subheadings": self._extract_subheadings(content),
            }
        except Exception as exc:
            raise RuntimeError(f"WriterAgent failed to generate section '{section}': {exc}") from exc
