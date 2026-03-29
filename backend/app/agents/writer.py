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
        content = template.format(topic=topic)
        # Pad to approximate word count
        sources = research_data.get("sources", [])
        if sources:
            refs = " ".join(f"[{i+1}]" for i in range(min(len(sources), 3)))
            content += f" Supporting evidence from the literature {refs} corroborates these observations."
        words = content.split()
        while len(words) < word_count_target * 0.8:
            words += content.split()
        content = " ".join(words[:word_count_target])
        return {"section": section, "content": content, "word_count": len(content.split())}

    async def _llm_write(self, section: str, topic: str, word_count_target: int, research_data: dict, project_id: str, db) -> dict:
        try:
            sources_summary = json.dumps(research_data.get("sources", [])[:3])
            prompt = (
                f"Write the '{section}' section of an academic essay on '{topic}'. "
                f"Target approximately {word_count_target} words. "
                f"Available sources: {sources_summary}. "
                "Write in formal academic style with citations where appropriate."
            )
            content = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
            )
            return {"section": section, "content": content, "word_count": len(content.split())}
        except Exception:
            return self._template_write(section, topic, word_count_target, research_data)
