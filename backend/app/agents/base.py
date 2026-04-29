import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings


class AgentBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def execute(self, input_data: dict, project_id: str, db: AsyncSession) -> dict:
        pass

    async def _call_with_routing(
        self,
        task_type: str,
        cheap_model: str,
        expensive_model: str,
        prompt: str,
        db: AsyncSession,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, bool]:
        """Call a cheap model, route the output, and escalate if recommended.

        1. Calls ``cheap_model`` with ``prompt``.
        2. Passes the output to ``route_task`` for an escalation decision.
        3. If escalation is recommended, calls ``expensive_model`` and returns
           its output together with ``escalated=True``.
        4. Otherwise returns the cheap-model output with ``escalated=False``.

        If the cheap-model call or the routing step fails, the exception
        propagates so the caller can apply its own fallback strategy.

        Args:
            task_type: Short task description for the router (e.g.
                ``"literature summarisation"``).
            cheap_model: Model name constant from ``model_config`` for the
                initial low-cost call.
            expensive_model: Model name constant from ``model_config`` for the
                escalated high-cost call.
            prompt: The prompt text sent to both models.
            db: Active async database session (used for API call logging).
            response_format: Optional response format hint passed to the API.
            temperature: Optional temperature override.
            max_tokens: Optional max-tokens override.

        Returns:
            ``(output_text, was_escalated)`` where ``was_escalated`` is True if
            the expensive model was used.
        """
        from app.agents.llm_client import timed_chat_completion
        from app.routing.router import route_task

        cheap_output = await timed_chat_completion(
            prompt,
            db=db,
            agent_name=self.name,
            log_api_call_fn=self._log_api_call,
            model=cheap_model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        should_escalate = await route_task(task_type, prompt, cheap_output)

        if should_escalate:
            expensive_output = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=expensive_model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return expensive_output, True

        return cheap_output, False

    async def _log_api_call(
        self,
        db: AsyncSession,
        endpoint: str,
        method: str,
        agent_name: str,
        duration_ms: float,
        status_code: int,
        metadata: dict | None = None,
    ):
        from app.models import ApiLog
        log = ApiLog(
            endpoint=endpoint,
            method=method,
            agent_name=agent_name,
            duration_ms=duration_ms,
            status_code=status_code,
            timestamp=datetime.now(timezone.utc),
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        db.add(log)

    async def _update_agent_state(
        self, db: AsyncSession, project_id: str, status: str, output: dict | None = None
    ):
        from app.models import AgentState
        result = await db.execute(
            select(AgentState).where(
                AgentState.project_id == project_id,
                AgentState.agent_name == self.name,
            )
        )
        state = result.scalar_one_or_none()
        if state is None:
            state = AgentState(
                project_id=project_id,
                agent_name=self.name,
                status=status,
                last_output=json.dumps(output) if output else None,
                updated_at=datetime.now(timezone.utc),
            )
            db.add(state)
        else:
            state.status = status
            state.last_output = json.dumps(output) if output else None
            state.updated_at = datetime.now(timezone.utc)
        await db.commit()
