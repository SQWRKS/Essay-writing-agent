"""
Cost-optimised model routing module.

Provides ``route_task`` which evaluates the output of a cheap LLM call and
decides whether the task should be escalated to a high-cost model.

Usage example::

    from app.routing.router import route_task

    cheap_output = await timed_chat_completion(prompt, model=MODEL_DEEPSEEK, ...)
    if await route_task("literature summarisation", prompt, cheap_output):
        final_output = await timed_chat_completion(prompt, model=MODEL_GPT5, ...)
    else:
        final_output = cheap_output
"""

import json
import logging

from app.routing.model_config import ROUTER_MODEL

logger = logging.getLogger(__name__)

# Maximum characters of cheap-model output included in the routing prompt.
# Keeping this bounded prevents the routing call itself from becoming
# expensive on unusually long outputs.
_ROUTER_MAX_OUTPUT_CHARS: int = 2000

# ---------------------------------------------------------------------------
# Router prompt (exact text as specified)
# ---------------------------------------------------------------------------
_ROUTER_PROMPT_TEMPLATE = (
    "You are a routing agent in a cost-sensitive AI system.\n\n"
    "Your job is to decide whether a task requires a high-cost, high-intelligence model, "
    "or if a low-cost model is sufficient.\n\n"
    "TASK TYPE:\n{task_type}\n\n"
    "OUTPUT TO EVALUATE:\n{model_output}\n\n"
    "Decide:\n\n"
    "1. Does this output show shallow reasoning, lack of depth, or generic thinking?\n\n"
    "2. Does the task require:\n"
    "   \n"
    "   - deep reasoning?\n"
    "   - multi-step logic?\n"
    "   - synthesis of multiple ideas?\n"
    "   - critical evaluation or judgement?\n\n"
    "3. Would errors here significantly reduce the quality of a university-level essay?\n\n"
    "If ANY of the above are true → escalate.\n\n"
    "Respond ONLY in JSON:\n\n"
    "{{\n"
    '"escalate": true/false,\n'
    '"reason": "short explanation"\n'
    "}}"
)


async def route_task(task_type: str, input_text: str, cheap_model_output: str) -> bool:
    """Evaluate cheap-model output and decide whether to escalate to a high-cost model.

    Calls the lightweight ``ROUTER_MODEL`` with the standard router prompt,
    parses the JSON response, and returns ``True`` if escalation is recommended.

    If the routing call fails for any reason (network error, malformed JSON,
    missing API key, etc.) the function returns ``False`` so the caller can
    proceed with the cheap output rather than blocking the pipeline.

    Args:
        task_type: Short description of the task being evaluated, e.g.
            ``"essay planning"`` or ``"literature summarisation"``.
        input_text: The original task input passed to the cheap model.
            Currently unused in the prompt but available for future extensions.
        cheap_model_output: The text output produced by the cheap model that
            the router should evaluate.

    Returns:
        ``True`` if the router recommends escalating to an expensive model,
        ``False`` otherwise.
    """
    from app.agents.llm_client import chat_completion, is_llm_available  # late import to avoid cycles

    if not is_llm_available():
        return False

    prompt = _ROUTER_PROMPT_TEMPLATE.format(
        task_type=task_type,
        # Cap the evaluated output to keep the routing call cheap
        model_output=cheap_model_output[:_ROUTER_MAX_OUTPUT_CHARS],
    )

    try:
        response = await chat_completion(
            prompt,
            model=ROUTER_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=128,
        )
        payload = json.loads(response)
        escalate: bool = bool(payload.get("escalate", False))
        reason: str = payload.get("reason", "")
        logger.debug(
            "Router decision for task_type=%r: escalate=%s reason=%r",
            task_type,
            escalate,
            reason,
        )
        return escalate
    except Exception as exc:
        logger.warning(
            "route_task failed (task_type=%r): %s; defaulting to no escalation",
            task_type,
            exc,
        )
        return False
