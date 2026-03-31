"""
Centralized LLM client factory for all agents.

Supports OpenAI (default), Anthropic, DeepSeek, and Mistral as LLM providers.
The default provider is selected via the LLM_PROVIDER setting.  Individual
calls may override the model (and therefore the provider) by passing the
``model`` keyword argument, which enables cost-optimised per-agent routing.

Token-efficiency helpers:
  - ``truncate_text(text, max_chars)`` — hard-trims a string to ``max_chars``.
  - ``quality_max_tokens(default)`` — returns a token ceiling appropriate for
    the current QUALITY_MODE so callers don't have to import settings directly.

Anthropic prompt caching:
  When the provider is Anthropic the last large context block in the user
  message is marked with ``cache_control: {"type": "ephemeral"}`` so that
  repeated calls that share that prefix benefit from caching (up to ~90 %
  token cost reduction on subsequent invocations within the TTL window).

Per-model routing:
  Pass ``model=<model_name>`` to ``chat_completion`` / ``timed_chat_completion``
  to override the default model.  The provider is determined automatically from
  ``app.routing.model_config.MODEL_PROVIDER_MAP``.  DeepSeek and Mistral use
  the OpenAI-compatible client with provider-specific base URLs and API keys
  configured in ``Settings``.
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

def _resolve_provider(model: str | None) -> tuple[str, str | None]:
    """Return ``(provider, resolved_model)`` for the given model name.

    When ``model`` is ``None`` the default ``settings.LLM_PROVIDER`` and its
    associated model name are used (backward-compatible behaviour).  When
    ``model`` is provided the provider is looked up from
    ``MODEL_PROVIDER_MAP``; if the model is not listed the default provider is
    used as a safe fallback.
    """
    if model is None:
        return settings.LLM_PROVIDER.lower(), None

    from app.routing.model_config import MODEL_PROVIDER_MAP  # late import to avoid cycles

    provider = MODEL_PROVIDER_MAP.get(model, settings.LLM_PROVIDER.lower())
    return provider, model


async def chat_completion(
    prompt: str,
    *,
    model: str | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request to the appropriate LLM provider.

    When ``model`` is ``None`` the default provider and model configured via
    ``settings`` are used (backward-compatible).  When ``model`` is supplied
    the provider is resolved automatically from ``MODEL_PROVIDER_MAP`` and the
    appropriate API key / base URL is selected from ``settings``.

    Returns the text content of the response.
    Raises RuntimeError if no API key is configured for the resolved provider.
    Raises RuntimeError (wrapping asyncio.TimeoutError) if the request exceeds
    ``LLM_REQUEST_TIMEOUT`` seconds.
    """
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else quality_max_tokens()
    timeout = float(settings.LLM_REQUEST_TIMEOUT)

    provider, resolved_model = _resolve_provider(model)

    if provider == "anthropic":
        coro = _anthropic_completion(
            prompt,
            model=resolved_model,
            temperature=temp,
            max_tokens=tokens,
        )
    elif provider in ("deepseek", "mistral"):
        coro = _openai_compat_completion(
            prompt,
            provider=provider,
            model=resolved_model,
            temperature=temp,
            max_tokens=tokens,
            response_format=response_format,
        )
    else:
        coro = _openai_completion(
            prompt,
            model=resolved_model,
            temperature=temp,
            max_tokens=tokens,
            response_format=response_format,
        )

    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        display_model = resolved_model or (
            settings.ANTHROPIC_MODEL if provider == "anthropic" else settings.LLM_MODEL
        )
        raise RuntimeError(
            f"LLM API request timed out after {int(timeout)}s "
            f"(provider={provider}, model={display_model})"
        ) from None


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _openai_completion(
    prompt: str,
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    response_format: dict | None = None,
) -> str:
    """Send a completion request to OpenAI."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    resolved = model or settings.LLM_MODEL
    kwargs: dict[str, Any] = {
        "model": resolved,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def _openai_compat_completion(
    prompt: str,
    *,
    provider: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    response_format: dict | None = None,
) -> str:
    """Send a completion request to an OpenAI-compatible provider (DeepSeek, Mistral).

    Both DeepSeek and Mistral expose an OpenAI-compatible REST API, so the
    standard ``openai`` client can be reused by supplying a custom ``base_url``
    and the provider-specific API key from ``settings``.
    """
    from openai import AsyncOpenAI

    if provider == "deepseek":
        api_key = settings.DEEPSEEK_API_KEY
        base_url = settings.DEEPSEEK_BASE_URL
    elif provider == "mistral":
        api_key = settings.MISTRAL_API_KEY
        base_url = settings.MISTRAL_BASE_URL
    else:
        raise ValueError(f"Unknown OpenAI-compatible provider: {provider!r}")

    if not api_key:
        raise RuntimeError(
            f"No API key configured for provider {provider!r}. "
            f"Set {'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'MISTRAL_API_KEY'} in your .env file."
        )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    resolved = model or settings.LLM_MODEL
    kwargs: dict[str, Any] = {
        "model": resolved,
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
    model: str | None,
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
        model=model or settings.ANTHROPIC_MODEL,
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
    """Return True if the default LLM provider is configured."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    return bool(settings.OPENAI_API_KEY)


def is_model_available(model: str) -> bool:
    """Return True if the API key required for ``model`` is configured.

    Uses ``MODEL_PROVIDER_MAP`` to determine the provider for the given model
    name, then checks the corresponding API key in ``settings``.  Falls back
    to ``is_llm_available()`` if the model is not listed in the provider map.
    """
    from app.routing.model_config import MODEL_PROVIDER_MAP  # late import to avoid cycles

    provider = MODEL_PROVIDER_MAP.get(model)
    if provider is None:
        return is_llm_available()
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if provider == "deepseek":
        return bool(settings.DEEPSEEK_API_KEY)
    if provider == "mistral":
        return bool(settings.MISTRAL_API_KEY)
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
    model: str | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request and log the duration to the database.

    ``log_api_call_fn`` should be the agent's ``_log_api_call`` bound method.
    When ``model`` is provided it overrides the default model for this call;
    the provider is resolved automatically from ``MODEL_PROVIDER_MAP``.
    """
    start = time.monotonic()
    provider, _ = _resolve_provider(model)
    endpoint = f"/{provider}/chat"
    try:
        content = await chat_completion(
            prompt,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration = (time.monotonic() - start) * 1000
        await log_api_call_fn(db, endpoint, "POST", agent_name, duration, 200)
        return content
    except Exception:
        duration = (time.monotonic() - start) * 1000
        await log_api_call_fn(db, endpoint, "POST", agent_name, duration, 500)
        raise
