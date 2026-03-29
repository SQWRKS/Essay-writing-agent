"""
Centralized LLM client factory for all agents.

Supports OpenAI (default) and Anthropic as LLM providers.
The provider is selected via the LLM_PROVIDER setting.
"""

import asyncio
import time
from typing import Any

from app.core.config import settings


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
    tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
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
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def is_llm_available() -> bool:
    """Return True if at least one LLM provider is configured."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    return bool(settings.OPENAI_API_KEY)


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
