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
    },
    {
        "key": "literature_review",
        "title": "Literature Review",
        "description": "Surveys existing research and identifies gaps.",
        "word_count_target": 800,
    },
    {
        "key": "methodology",
        "title": "Methodology",
        "description": "Describes research methods and experimental design.",
        "word_count_target": 700,
    },
    {
        "key": "results",
        "title": "Results",
        "description": "Presents findings and data analysis.",
        "word_count_target": 600,
    },
    {
        "key": "discussion",
        "title": "Discussion",
        "description": "Interprets results and discusses implications.",
        "word_count_target": 600,
    },
    {
        "key": "conclusion",
        "title": "Conclusion",
        "description": "Summarizes contributions and future directions.",
        "word_count_target": 300,
    },
]


class PlannerAgent(AgentBase):
    name = "planner"

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
            })
            all_queries.extend(queries)
        return {
            "sections": sections,
            "research_queries": list(set(all_queries)),
            "estimated_total_words": sum(s["word_count_target"] for s in sections),
        }

    async def _llm_plan(self, topic: str, project_id: str, db) -> dict:
        try:
            prompt = (
                f"Create a detailed academic essay plan for the topic: '{topic}'. "
                "Return a JSON object with keys: sections (list), research_queries (list), estimated_total_words (int). "
                "Each section must have: key, title, description, research_queries (list of 3), word_count_target."
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                response_format={"type": "json_object"},
            )
            return json.loads(content)
        except Exception:
            return self._template_plan(topic)
