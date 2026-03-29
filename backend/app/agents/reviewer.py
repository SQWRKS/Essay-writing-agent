import json
import time
from app.agents.base import AgentBase
from app.core.config import settings


class ReviewerAgent(AgentBase):
    name = "reviewer"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        section = input_data.get("section", "")
        content = input_data.get("content", "")

        if settings.OPENAI_API_KEY:
            result = await self._llm_review(section, content, project_id, db)
        else:
            result = self._heuristic_review(section, content)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _heuristic_review(self, section: str, content: str) -> dict:
        words = content.split()
        word_count = len(words)
        score = 0.5
        suggestions = []
        feedback_parts = []

        if word_count < 100:
            score -= 0.2
            suggestions.append("Expand the content significantly.")
            feedback_parts.append("Content is too short.")
        elif word_count > 200:
            score += 0.2

        if any(word in content.lower() for word in ["however", "therefore", "furthermore", "moreover"]):
            score += 0.1
            feedback_parts.append("Good use of transition words.")
        else:
            suggestions.append("Add transition words for better flow.")

        if content.count(".") > 3:
            score += 0.1
        if any(c.isdigit() for c in content):
            score += 0.1
            feedback_parts.append("Includes numerical data, which is good.")

        score = min(1.0, max(0.0, score))
        approved = score >= 0.6
        return {
            "score": round(score, 2),
            "feedback": " ".join(feedback_parts) if feedback_parts else "Content reviewed.",
            "suggestions": suggestions,
            "approved": approved,
        }

    async def _llm_review(self, section: str, content: str, project_id: str, db) -> dict:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            prompt = (
                f"Review the following '{section}' section of an academic essay. "
                "Provide a JSON response with: score (0-1 float), feedback (string), suggestions (list of strings), approved (bool).\n\n"
                f"Content:\n{content[:2000]}"
            )
            start = time.monotonic()
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            duration = (time.monotonic() - start) * 1000
            await self._log_api_call(db, "/openai/chat", "POST", self.name, duration, 200)
            return json.loads(response.choices[0].message.content)
        except Exception:
            return self._heuristic_review(section, content)
