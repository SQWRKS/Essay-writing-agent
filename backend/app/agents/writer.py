import json
import re
from collections import Counter

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion, truncate_text, quality_max_tokens
from app.core.config import settings
from app.routing.model_config import AGENT_MODELS, WRITER_REFINE_SECTIONS


# Maps section keys to the specific argumentative goal for build_prompt().
SECTION_GUIDANCE: dict[str, str] = {
    "introduction": "frame the topic, establish the central argument, and explain why it matters",
    "literature_review": "synthesize existing scholarship and identify unresolved tensions or gaps",
    "methodology": "justify the chosen approach and acknowledge its limitations",
    "results": "present findings with specificity and distinguish robust patterns from weaker observations",
    "discussion": "interpret findings, weigh implications against limitations, and relate back to the thesis",
    "conclusion": "reinforce the core argument and end with a precise contribution or future direction",
}

# A one-paragraph few-shot example used to demonstrate the desired prose style.
_FEW_SHOT_EXAMPLE = (
    "EXAMPLE OF DESIRED ACADEMIC PROSE:\n"
    '"Transformer-based retrieval systems demonstrated substantial gains in clinical question '
    "answering, with benchmarks reporting up to 18% improvement in answer accuracy when retrieval "
    "precision exceeded 0.75 [1]. However, these gains were sensitive to corpus coverage: "
    "performance degraded noticeably in specialist sub-domains where annotated training data "
    'remained scarce [2]. A key limitation acknowledged by reviewers is that most evaluations '
    'relied on short-form factoid questions rather than longitudinal diagnostic reasoning, '
    'leaving the generalizability of results to high-stakes clinical settings uncertain."'
)


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
        # Accept thesis and research_notes from the top level of input_data so
        # callers that pre-populate them outside research_data still work.
        thesis: str = (input_data.get("thesis") or research_data.get("thesis") or "").strip()
        research_notes: list[str] = (
            input_data.get("research_notes")
            or research_data.get("research_notes")
            or []
        )

        if not is_llm_available():
            raise RuntimeError(
                "No LLM provider is configured for WriterAgent. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running the pipeline."
            )

        result = await self._llm_write(
            section, topic, word_count_target, research_data, section_plan,
            feedback, project_id, db,
            writing_style=writing_style,
            thesis=thesis,
            research_notes=research_notes,
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
        """Produce a structured fallback section without an LLM.

        The output includes prose grounded in the supplied evidence pack,
        inline citation markers, and optional thesis/must-cover language.
        Meta-writing phrases (directives, objectives, revision notes) are
        intentionally excluded so the result is usable as draft prose.
        """
        section_plan = section_plan or {}
        research_data = research_data or {}
        template = SECTION_TEMPLATES.get(section, SECTION_TEMPLATES["introduction"])
        content_parts = [template.format(topic=topic)]

        thesis_goal = section_plan.get("thesis_goal", "")
        if thesis_goal:
            content_parts.append(
                f"The central argument is that {thesis_goal.strip().rstrip('.')}."
            )

        must_cover = section_plan.get("must_cover", [])
        if must_cover:
            items = ", ".join(item.strip() for item in must_cover[:4] if str(item).strip())
            content_parts.append(f"The analysis covers {items}.")

        evidence_requirements = section_plan.get("evidence_requirements", [])
        if evidence_requirements:
            reqs = ", ".join(item.strip() for item in evidence_requirements[:3] if str(item).strip())
            content_parts.append(f"The discussion is expected to address {reqs}.")

        research_summary = (research_data.get("research_summary") or "").strip()
        if research_summary:
            content_parts.append(research_summary[:300].rstrip() + ("…" if len(research_summary) > 300 else ""))

        # Inline evidence highlights with citation markers
        evidence_pack = research_data.get("evidence_pack", [])
        if evidence_pack:
            highlight_lines = []
            for idx, item in enumerate(evidence_pack[:4], 1):
                finding = (item.get("key_findings") or item.get("abstract") or "").strip()
                quant = item.get("quantitative_data", [])
                quant_str = "; ".join(str(q) for q in quant[:2]) if quant else ""
                source_title = (item.get("source") or {}).get("title", "")
                if finding:
                    line = f"[{idx}] {finding[:200]}"
                    if quant_str:
                        line += f" ({quant_str})"
                    if source_title:
                        line += f" [{source_title}]"
                    highlight_lines.append(line)
            if highlight_lines:
                synthesis = (
                    "These findings collectively support the central claim and highlight "
                    "key mechanisms, trade-offs, and boundaries of current knowledge."
                )
                content_parts.append(
                    "Evidence highlights:\n" + "\n".join(highlight_lines) + "\n\n" + synthesis
                )

        if feedback:
            content_parts.append(
                "Revisions address specificity, evidence linkage, and analytical depth."
            )

        content = "\n\n".join(content_parts)
        return {"section": section, "content": content, "word_count": len(content.split())}

    def build_prompt(self, payload: dict, stricter_feedback: str = "") -> str:
        """Build a full LLM prompt for a section from a structured payload dict.

        Payload keys: section, topic, thesis, research_notes (list[str]),
        section_queries (list[str]), research_summary, feedback, word_count.
        """
        section_goal = SECTION_GUIDANCE.get(
            payload.get("section", ""),
            "advance the thesis with technical, evidence-based analysis",
        )
        return (
            f"Write the {payload['section']} section of a final-year university paper on '{payload['topic']}'. "
            f"The section must {section_goal}.\n\n"
            f"{_FEW_SHOT_EXAMPLE}\n\n"
            f"Central thesis:\n{payload.get('thesis', '')}\n\n"
            f"Structured research notes:\n{json.dumps(payload.get('research_notes', [])[:8])}\n\n"
            f"Section queries:\n{json.dumps(payload.get('section_queries', [])[:5])}\n\n"
            f"Literature synthesis:\n{payload.get('research_summary', '')}\n\n"
            f"Revision feedback:\n{payload.get('feedback', '')}\n\n"
            f"Additional constraints:\n{stricter_feedback}\n\n"
            "Requirements:\n"
            f"- Write approximately {payload.get('word_count', 500)} words in polished academic prose.\n"
            "- Use named studies, systems, datasets, methods, or case examples from the notes.\n"
            "- Include quantitative evidence whenever the notes provide it.\n"
            "- Use inline citations like [1], [2], [3] for factual claims.\n"
            "- Do not mention the prompt, instructions, target length, or that you are an AI.\n"
            "- Do not use generic filler such as 'this paper examines' or 'has gained considerable attention'.\n"
            "- Every paragraph must advance the thesis with concrete evidence or technical reasoning.\n"
            "Return only the final section text."
        )

    def _grounded_write(self, payload: dict) -> dict:
        """Build a minimal grounded draft from research notes without an LLM.

        Used as a deterministic fallback when the primary LLM path is unavailable.
        Assembles content paragraph-by-paragraph from the research notes supplied
        in ``payload`` and trims or expands to approximate ``word_count_target``.
        """
        notes: list[str] = payload.get("research_notes", [])[:6]
        section: str = payload.get("section", "introduction")
        topic: str = payload.get("topic", "the topic")
        word_count_target: int = int(payload.get("word_count", 500))
        thesis: str = (
            payload.get("thesis")
            or f"The literature on {topic} supports a specific, evidence-grounded argument."
        )

        paragraphs: list[str] = []

        # Opening paragraph: thesis + first evidence note
        first_note = notes[0] if notes else f"{topic} is an active area of research."
        paragraphs.append(
            f"{thesis} "
            f"In the {section} context, the available evidence indicates that {first_note.rstrip('.')}."
        )

        # Body paragraphs: one per remaining note
        for i, note in enumerate(notes[1:], 2):
            paragraphs.append(f"Further evidence [{i}] shows that {note.rstrip('.')}.")

        # Closing synthesis
        paragraphs.append(
            f"Taken together, the evidence above supports the central argument "
            f"and highlights both the contributions and the remaining uncertainties in the {topic} literature."
        )

        content = "\n\n".join(paragraphs)

        # Trim to word_count_target if overlong
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
        thesis: str = "",
        research_notes: list[str] | None = None,
    ) -> dict:
        """Write a section using cost-optimised model routing.

        Draft strategy:
        - All sections are first drafted with the cheap model (deepseek-chat).
        - Sections in ``WRITER_REFINE_SECTIONS`` (introduction, conclusion) are
          then refined using the expensive model (claude-3.5-sonnet) to ensure
          the highest-quality prose for the most reader-visible parts of the
          essay.
        - All other sections keep the cheap draft as the final output.
        """
        try:
            section_plan = section_plan or {}
            research_notes = research_notes or []
            # Token-efficient evidence selection: top 3 items are enough for grounding
            evidence_pack = research_data.get("evidence_pack", [])[:3]
            section_queries = research_data.get("section_queries", [])[:3]
            # Truncate research summary to avoid bloating the prompt
            research_summary = truncate_text(research_data.get("research_summary", ""), 500)
            # Use a tighter sources digest (3 sources, 100-char abstracts)
            sources_summary = self._build_sources_digest(research_data.get("sources", []), limit=3)
            evidence_summary = json.dumps(evidence_pack)
            # Thesis: prefer explicitly passed arg, fall back to research_data field
            effective_thesis = thesis or truncate_text(research_data.get("thesis", ""), 300)
            # Research notes digest (e.g. "[1] Title: finding." format)
            notes_digest = "\n".join(research_notes[:8]) if research_notes else ""
            thesis_goal = section_plan.get("thesis_goal", "")
            must_cover = section_plan.get("must_cover", [])
            evidence_requirements = section_plan.get("evidence_requirements", [])
            writing_directive = section_plan.get("writing_directive", "")
            subheading_hints = section_plan.get("subheading_hints", [])[:2]
            # Optional writing-style directive line (empty string → omitted cleanly)
            style_line = f"WRITING STYLE: {writing_style}\n" if writing_style else ""
            # Thesis line (only added when a thesis is available)
            thesis_line = f"CENTRAL THESIS: {effective_thesis}\n" if effective_thesis else ""
            # Research notes line (only added when notes are available)
            notes_line = f"RESEARCH NOTES:\n{notes_digest}\n\n" if notes_digest else ""
            prompt = (
                f"Write the '{section}' section (~{word_count_target} words) of an academic essay on '{topic}'.\n\n"
                f"{_FEW_SHOT_EXAMPLE}\n\n"
                f"{thesis_line}"
                f"SECTION OBJECTIVE: {thesis_goal}\n"
                f"MUST COVER: {'; '.join(str(i) for i in must_cover)}\n"
                f"EVIDENCE REQUIREMENTS: {'; '.join(str(i) for i in evidence_requirements)}\n"
                f"WRITING DIRECTIVE: {writing_directive}\n"
                f"{style_line}"
                f"SUBHEADINGS (optional, max 2): {', '.join(subheading_hints) if subheading_hints else 'None'}\n\n"
                f"RESEARCH SYNTHESIS:\n{research_summary}\n\n"
                f"{notes_line}"
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

            # Step 1: Draft with cheap model
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["writer"]["draft"],
                temperature=0.45,
                max_tokens=quality_max_tokens(),
            )

            # Step 2: Refine with expensive model for high-impact sections only
            if section in WRITER_REFINE_SECTIONS:
                refine_prompt = (
                    f"You are refining the '{section}' section of an academic essay on '{topic}'.\n\n"
                    "Improve the prose quality, argumentative clarity, and elegance of the draft below. "
                    "Preserve all factual claims, citations, and structure. "
                    "Focus on the opening argument and closing synthesis.\n\n"
                    f"DRAFT:\n{content}\n\n"
                    "Return only the refined section prose."
                )
                content = await timed_chat_completion(
                    refine_prompt,
                    db=db,
                    agent_name=self.name,
                    log_api_call_fn=self._log_api_call,
                    model=AGENT_MODELS["writer"]["refine"],
                    temperature=0.3,
                    max_tokens=quality_max_tokens(),
                )

            import re as _re
            citation_count = len(_re.findall(r"\[\d+\]", content))
            return {
                "section": section,
                "content": content,
                "word_count": len(content.split()),
                "subheadings": self._extract_subheadings(content),
                "validation": {
                    "citation_count": citation_count,
                    "has_limitation": any(
                        kw in content.lower()
                        for kw in ("limitation", "uncertain", "constraint", "caveat", "however")
                    ),
                },
            }
        except Exception as exc:
            raise RuntimeError(f"WriterAgent failed to generate section '{section}': {exc}") from exc

