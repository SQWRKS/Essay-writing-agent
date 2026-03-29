import asyncio
import json
from typing import AsyncGenerator, Dict
from collections import defaultdict


class SSEManager:
    def __init__(self):
        self._queues: Dict[str, list] = defaultdict(list)

    async def subscribe(self, project_id: str) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[project_id].append(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if event is None:
                        break
                    yield event
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"heartbeat\"}\n\n"
        finally:
            self._queues[project_id].remove(queue)
            if not self._queues[project_id]:
                del self._queues[project_id]

    async def publish(self, project_id: str, event_type: str, data: dict):
        payload = json.dumps({"type": event_type, **data})
        message = f"data: {payload}\n\n"
        queues = list(self._queues.get(project_id, []))
        for queue in queues:
            await queue.put(message)

    async def close(self, project_id: str):
        queues = list(self._queues.get(project_id, []))
        for queue in queues:
            await queue.put(None)


sse_manager = SSEManager()
