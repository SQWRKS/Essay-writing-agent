"""
Centralized LLM client factory for all agents.

Supports OpenAI (default) and Anthropic as LLM providers.
The provider is selected via the LLM_PROVIDER setting.

Token-efficiency helpers:
  - ``truncate_text(text, max_chars)`` — hard-trims a string to ``max_chars``.
  - ``quality_max_tokens(default)`` — returns a token ceiling appropriate for
    the current QUALITY_MODE so callers don't have to import settings directly.

Anthropic prompt caching:
  When the provider is Anthropic the last large context block in the user
  message is marked with ``cache_control: {"type": "ephemeral"}`` so that
  repeated calls that share that prefix benefit from caching (up to ~90 %
  token cost reduction on subsequent invocations within the TTL window).
"""

import asyncio
import time
from typing import Any

from app.core.config import settings


# ---------------------------------------------------------------------------
# Token / text helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int) -> str:
    """Hard-trim ``text`` to ``max_chars`` characters, appending '…' if cut."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def quality_max_tokens(default: int | None = None) -> int:
    """Return a max-tokens ceiling appropriate for the current QUALITY_MODE.

    Balanced mode uses a lower ceiling to reduce costs while still producing
    complete responses.  Quality mode uses the configured maximum.
    """
    if default is not None:
        return default
    mode = (settings.QUALITY_MODE or "quality").lower()
    if mode == "balanced":
        return min(settings.LLM_MAX_TOKENS, 2048)
    return settings.LLM_MAX_TOKENS


# ---------------------------------------------------------------------------
# Public completion helpers
# ---------------------------------------------------------------------------

async def chat_completion(
    prompt: str,
    *,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request to the configured LLM provider.

    Returns the text content of the response.
    Raises RuntimeError if no provider API key is configured.
    Raises asyncio.TimeoutError (wrapped as RuntimeError) if the request
    exceeds LLM_REQUEST_TIMEOUT seconds.
    """
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else quality_max_tokens()
    timeout = float(settings.LLM_REQUEST_TIMEOUT)

    provider = settings.LLM_PROVIDER.lower()

    if provider == "anthropic":
        coro = _anthropic_completion(prompt, temperature=temp, max_tokens=tokens)
    else:
        coro = _openai_completion(
            prompt,
            temperature=temp,
            max_tokens=tokens,
            response_format=response_format,
        )

    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"LLM API request timed out after {int(timeout)}s "
            f"(provider={provider}, model={settings.LLM_MODEL if provider != 'anthropic' else settings.ANTHROPIC_MODEL})"
        ) from None


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _openai_completion(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    response_format: dict | None = None,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def _anthropic_completion(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Send a completion request to Anthropic, enabling prompt caching.

    The user message content is split into two blocks so that a large leading
    context block can be cached (marked ``ephemeral``), while the trailing
    task-specific instruction is never cached (avoiding stale cache hits when
    the task changes).  On cache hit, Anthropic charges ~10 % of the normal
    input-token cost for the cached prefix.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Split the prompt at the last double-newline that is at least 200 chars
    # before the end.  Everything up to that point is the "context" (cacheable);
    # the rest is the "task" (not cached so it can vary per call).
    split_pos = _find_cache_split(prompt)
    if split_pos > 0:
        context_part = prompt[:split_pos]
        task_part = prompt[split_pos:]
        content: list[Any] = [
            {
                "type": "text",
                "text": context_part,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": task_part},
        ]
    else:
        content = [{"type": "text", "text": prompt}]

    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    return response.content[0].text


def _find_cache_split(prompt: str, min_trailing: int = 200, min_leading: int = 200) -> int:
    """Return an index to split the prompt for Anthropic prompt caching.

    We look for a paragraph break (double newline) that leaves at least
    ``min_trailing`` characters as the uncached task portion and at least
    ``min_leading`` characters as the cached context portion.  Returns 0 if no
    suitable split point is found (no caching applied).
    """
    if len(prompt) < min_leading + min_trailing:
        return 0
    # Search backward from the split window to find a clean paragraph break.
    search_end = len(prompt) - min_trailing
    idx = prompt.rfind("\n\n", min_leading, search_end)
    if idx == -1:
        idx = prompt.rfind("\n", min_leading, search_end)
    return idx + 1 if idx != -1 else 0


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_llm_available() -> bool:
    """Return True if at least one LLM provider is configured."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    return bool(settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Instrumented wrapper used by all agents
# ---------------------------------------------------------------------------

async def timed_chat_completion(
    prompt: str,
    *,
    db,
    agent_name: str,
    log_api_call_fn,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request and log the duration to the database.

    ``log_api_call_fn`` should be the agent's ``_log_api_call`` bound method.
    """
    start = time.monotonic()
    try:
        content = await chat_completion(
            prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration = (time.monotonic() - start) * 1000
        endpoint = f"/{settings.LLM_PROVIDER}/chat"
        await log_api_call_fn(db, endpoint, "POST", agent_name, duration, 200)
        return content
    except Exception:
        duration = (time.monotonic() - start) * 1000
        endpoint = f"/{settings.LLM_PROVIDER}/chat"
        await log_api_call_fn(db, endpoint, "POST", agent_name, duration, 500)
        raise
