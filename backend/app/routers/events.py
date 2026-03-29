from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.core.sse import sse_manager

router = APIRouter(prefix="/projects", tags=["events"])


@router.get("/{project_id}/events")
async def project_events(project_id: str):
    async def event_generator():
        async for event in sse_manager.subscribe(project_id):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
