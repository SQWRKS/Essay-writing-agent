import json
from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.core.config import settings


SECTION_TEMPLATES = [
    {
        "key": "introduction",
        "title": "Introduction",
        "description": "Provides background, motivation, and overview of the research topic.",
        "word_count_target": 500,
        "thesis_goal": "Frame the topic, establish the essay's central argument, and explain why the problem matters.",
        "must_cover": [
            "historical or conceptual context",
            "core problem or debate",
            "thesis or analytical stance",
        ],
        "evidence_requirements": [
            "use at least two sources to establish context",
            "anchor major factual claims with citations",
        ],
        "writing_directive": "Move quickly from background into a defensible thesis rather than staying descriptive.",
        "subheading_hints": ["Context", "Problem Framing"],
    },
    {
        "key": "literature_review",
        "title": "Literature Review",
        "description": "Surveys existing research and identifies gaps.",
        "word_count_target": 800,
        "thesis_goal": "Map the major strands of scholarship and identify the most important tensions or gaps.",
        "must_cover": [
            "dominant schools of thought or approaches",
            "points of agreement and disagreement",
            "clear research gap or unresolved tension",
        ],
        "evidence_requirements": [
            "compare multiple sources rather than summarizing one at a time",
            "cite representative and recent scholarship",
        ],
        "writing_directive": "Synthesize sources into an argument about the field instead of listing papers sequentially.",
        "subheading_hints": ["Major Debates", "Research Gaps"],
    },
    {
        "key": "methodology",
        "title": "Methodology",
        "description": "Describes research methods and experimental design.",
        "word_count_target": 700,
        "thesis_goal": "Explain how evidence is gathered or evaluated and justify why the chosen approach is credible.",
        "must_cover": [
            "method or analytical approach",
            "why the approach fits the question",
            "limits or trade-offs of the method",
        ],
        "evidence_requirements": [
            "ground methodological claims in cited sources where relevant",
            "acknowledge methodological limitations explicitly",
        ],
        "writing_directive": "Be explicit about justification and limitations, not only procedure.",
        "subheading_hints": ["Approach", "Limitations"],
    },
    {
        "key": "results",
        "title": "Results",
        "description": "Presents findings and data analysis.",
        "word_count_target": 600,
        "thesis_goal": "Present the strongest findings clearly and distinguish robust patterns from weaker observations.",
        "must_cover": [
            "primary findings or patterns",
            "supporting evidence or data points",
            "uncertainties or weaker findings",
        ],
        "evidence_requirements": [
            "anchor numerical or empirical statements with citations",
            "avoid overstating what the evidence proves",
        ],
        "writing_directive": "Prioritize specificity and evidence over broad summary language.",
        "subheading_hints": ["Primary Findings", "Uncertainty"],
    },
    {
        "key": "discussion",
        "title": "Discussion",
        "description": "Interprets results and discusses implications.",
        "word_count_target": 600,
        "thesis_goal": "Interpret the findings, explain their implications, and relate them back to the essay's central argument.",
        "must_cover": [
            "interpretation of the strongest findings",
            "implications for the field or problem",
            "limitations, counterarguments, or open questions",
        ],
        "evidence_requirements": [
            "tie interpretations back to cited findings",
            "avoid claims that go beyond the available evidence",
        ],
        "writing_directive": "Show analytical judgment by weighing implications against limitations.",
        "subheading_hints": ["Interpretation", "Implications"],
    },
    {
        "key": "conclusion",
        "title": "Conclusion",
        "description": "Summarizes contributions and future directions.",
        "word_count_target": 300,
        "thesis_goal": "Reinforce the essay's core argument and end with a precise statement of contribution or future direction.",
        "must_cover": [
            "restated argument or answer",
            "key takeaway from the evidence",
            "future direction or unresolved issue",
        ],
        "evidence_requirements": [
            "avoid introducing new unsupported claims",
            "synthesize rather than repeat earlier wording",
        ],
        "writing_directive": "End decisively and analytically rather than repeating the introduction.",
        "subheading_hints": ["Key Takeaways", "Future Work"],
    },
]


class PlannerAgent(AgentBase):
    name = "planner"

    def _as_bool(self, value, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return default

    def _normalize_section_plan(self, raw_section: dict, fallback: dict | None = None) -> dict:
        fallback = fallback or {}
        section = dict(raw_section or {})
        normalized = {
            "key": section.get("key") or fallback.get("key") or "introduction",
            "title": section.get("title") or fallback.get("title") or "Introduction",
            "description": section.get("description") or fallback.get("description") or "",
            "research_queries": section.get("research_queries") or fallback.get("research_queries") or [],
            "word_count_target": int(section.get("word_count_target") or fallback.get("word_count_target") or 500),
            "thesis_goal": section.get("thesis_goal") or fallback.get("thesis_goal") or "",
            "must_cover": section.get("must_cover") or fallback.get("must_cover") or [],
            "evidence_requirements": section.get("evidence_requirements") or fallback.get("evidence_requirements") or [],
            "writing_directive": section.get("writing_directive") or fallback.get("writing_directive") or "",
            "include": self._as_bool(section.get("include"), self._as_bool(fallback.get("include"), True)),
            "subheading_hints": section.get("subheading_hints") or fallback.get("subheading_hints") or [],
        }
        if not isinstance(normalized["must_cover"], list):
            normalized["must_cover"] = [str(normalized["must_cover"])]
        if not isinstance(normalized["evidence_requirements"], list):
            normalized["evidence_requirements"] = [str(normalized["evidence_requirements"])]
        if not isinstance(normalized["research_queries"], list):
            normalized["research_queries"] = [str(normalized["research_queries"])]
        if not isinstance(normalized["subheading_hints"], list):
            normalized["subheading_hints"] = [str(normalized["subheading_hints"])]
        return normalized

    def _normalize_plan(self, plan: dict) -> dict:
        raw_sections = plan.get("sections") or []
        sections = []
        fallback_map = {item["key"]: item for item in SECTION_TEMPLATES}

        if raw_sections:
            for raw in raw_sections:
                key = str((raw or {}).get("key") or "").strip().lower()
                fallback = fallback_map.get(key)
                normalized = self._normalize_section_plan(raw, fallback)
                if normalized.get("include", True):
                    sections.append(normalized)
        else:
            for fallback in SECTION_TEMPLATES:
                sections.append(self._normalize_section_plan({}, fallback))

        if not sections:
            sections = [self._normalize_section_plan({}, SECTION_TEMPLATES[0])]

        research_queries = plan.get("research_queries") or []
        if not isinstance(research_queries, list):
            research_queries = [str(research_queries)]

        if not research_queries:
            for section in sections:
                research_queries.extend(section.get("research_queries", []))

        return {
            "sections": sections,
            "research_queries": list(dict.fromkeys(research_queries)),
            "estimated_total_words": int(plan.get("estimated_total_words") or sum(s["word_count_target"] for s in sections)),
        }

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")

        if is_llm_available():
            result = await self._llm_plan(topic, project_id, db)
        else:
            result = self._template_plan(topic)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _template_plan(self, topic: str) -> dict:
        sections = []
        all_queries = []
        for tmpl in SECTION_TEMPLATES:
            queries = [
                f"{topic} {tmpl['key']} overview",
                f"recent advances in {topic} {tmpl['key']}",
                f"{topic} {tmpl['key']} methodology best practices",
            ]
            sections.append({
                "key": tmpl["key"],
                "title": tmpl["title"],
                "description": tmpl["description"],
                "research_queries": queries,
                "word_count_target": tmpl["word_count_target"],
                "thesis_goal": tmpl["thesis_goal"],
                "must_cover": tmpl["must_cover"],
                "evidence_requirements": tmpl["evidence_requirements"],
                "writing_directive": tmpl["writing_directive"],
            })
            all_queries.extend(queries)
        return self._normalize_plan({
            "sections": sections,
            "research_queries": list(set(all_queries)),
            "estimated_total_words": sum(s["word_count_target"] for s in sections),
        })

    async def _llm_plan(self, topic: str, project_id: str, db) -> dict:
        try:
            prompt = (
                f"Create a detailed academic essay plan for the topic: '{topic}'. "
                "Return a JSON object with keys: sections (list), research_queries (list), estimated_total_words (int). "
                "Each section must have: key, title, description, research_queries (list of 3), word_count_target, thesis_goal, must_cover (list), evidence_requirements (list), writing_directive, include (bool), subheading_hints (list of 0-2 short strings). "
                "Include only sections relevant to this topic. For non-empirical topics, set include=false or omit a Results section."
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                response_format={"type": "json_object"},
            )
            return self._normalize_plan(json.loads(content))
        except Exception:
            return self._template_plan(topic)
