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
        await db.commit()

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
