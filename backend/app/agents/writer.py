import json
import re
from collections import Counter

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

PROMPT_LEAK_PATTERNS = [
    "requirements:",
    "return only",
    "you are writing",
    "target length",
    "section-specific",
    "use inline citation markers",
]

GENERIC_PHRASES = [
    "this paper examines",
    "this study explores",
    "has gained considerable attention",
    "plays an important role",
    "in today's world",
    "it is clear that",
    "the significance of this research",
]


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
        writing_style: str = (input_data.get("writing_style") or "").strip()

        if not is_llm_available():
            raise RuntimeError(
                "No LLM provider is configured for WriterAgent. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running the pipeline."
            )

        result = await self._llm_write(
            section, topic, word_count_target, research_data, section_plan,
            feedback, project_id, db, writing_style=writing_style,
        )

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

        prompt = self.build_prompt(payload, stricter_feedback)
        return await timed_chat_completion(
            prompt,
            db=db,
            agent_name=self.name,
            log_api_call_fn=self._log_api_call,
            temperature=0.25 if attempt == 0 else 0.15,
            max_tokens=min(1600, max(900, payload["word_count"] * 3)),
        )

    def build_prompt(self, payload: dict, stricter_feedback: str = "") -> str:
        section_goal = SECTION_GUIDANCE.get(payload["section"], "advance the thesis with technical, evidence-based analysis")
        return (
            f"Write the {payload['section']} section of a final-year university paper on '{payload['topic']}'. "
            f"The section must {section_goal}.\n\n"
            f"Central thesis:\n{payload['thesis']}\n\n"
            f"Structured research notes:\n{json.dumps(payload['research_notes'][:8])}\n\n"
            f"Section queries:\n{json.dumps(payload['section_queries'][:5])}\n\n"
            f"Literature synthesis:\n{payload['research_summary']}\n\n"
            f"Revision feedback:\n{payload['feedback']}\n\n"
            f"Additional constraints:\n{stricter_feedback}\n\n"
            "Requirements:\n"
            f"- Write approximately {payload['word_count']} words in polished academic prose.\n"
            "- Use named studies, systems, datasets, methods, or case examples from the notes.\n"
            "- Include quantitative evidence whenever the notes provide it.\n"
            "- Use inline citations like [1], [2], [3] for factual claims.\n"
            "- Do not mention the prompt, instructions, target length, or that you are an AI.\n"
            "- Do not use generic filler such as 'this paper examines' or 'has gained considerable attention'.\n"
            "- Every paragraph must advance the thesis with concrete evidence or technical reasoning.\n"
            "Return only the final section text."
        )

    def _grounded_write(self, payload: dict) -> str:
        notes = payload["research_notes"][:6]
        section = payload["section"]
        topic = payload["topic"]
        thesis = payload["thesis"] or f"The literature on {topic} supports a specific, evidence-grounded argument."

        paragraphs = [
            f"{thesis} In the {section} context, the strongest evidence shows that {self._note_sentence(notes[0] if notes else topic, preserve_prefix=True)}",
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
        writing_style: str = "",
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
            # Optional writing-style directive line (empty string → omitted cleanly)
            style_line = f"WRITING STYLE: {writing_style}\n" if writing_style else ""
            prompt = (
                f"Write the '{section}' section (~{word_count_target} words) of an academic essay on '{topic}'.\n\n"
                f"SECTION OBJECTIVE: {thesis_goal}\n"
                f"MUST COVER: {'; '.join(str(i) for i in must_cover)}\n"
                f"EVIDENCE REQUIREMENTS: {'; '.join(str(i) for i in evidence_requirements)}\n"
                f"WRITING DIRECTIVE: {writing_directive}\n"
                f"{style_line}"
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
