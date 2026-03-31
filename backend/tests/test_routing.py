"""Tests for the cost-optimised routing module."""

import json
import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# model_config tests
# ---------------------------------------------------------------------------

def test_model_config_constants():
    from app.routing.model_config import (
        MODEL_GPT5,
        MODEL_CLAUDE_SONNET,
        MODEL_DEEPSEEK,
        MODEL_MISTRAL_SMALL,
        MODEL_GPT5_MINI,
        ROUTER_MODEL,
        AGENT_MODELS,
        MODEL_PROVIDER_MAP,
        WRITER_REFINE_SECTIONS,
    )
    assert MODEL_GPT5 == "gpt-5"
    assert MODEL_CLAUDE_SONNET == "claude-3.5-sonnet"
    assert MODEL_DEEPSEEK == "deepseek-chat"
    assert MODEL_MISTRAL_SMALL == "mistral-small"
    assert MODEL_GPT5_MINI == "gpt-5-mini"
    assert ROUTER_MODEL == MODEL_GPT5_MINI


def test_model_config_provider_map():
    from app.routing.model_config import MODEL_PROVIDER_MAP, MODEL_GPT5, MODEL_CLAUDE_SONNET, MODEL_DEEPSEEK, MODEL_MISTRAL_SMALL, MODEL_GPT5_MINI
    assert MODEL_PROVIDER_MAP[MODEL_GPT5] == "openai"
    assert MODEL_PROVIDER_MAP[MODEL_GPT5_MINI] == "openai"
    assert MODEL_PROVIDER_MAP[MODEL_CLAUDE_SONNET] == "anthropic"
    assert MODEL_PROVIDER_MAP[MODEL_DEEPSEEK] == "deepseek"
    assert MODEL_PROVIDER_MAP[MODEL_MISTRAL_SMALL] == "mistral"


def test_model_config_all_agents_covered():
    from app.routing.model_config import AGENT_MODELS
    expected_agents = {
        "planner", "research", "thesis", "verification", "writer",
        "grounding", "reviewer", "coherence", "citation", "figure", "web_search",
    }
    assert expected_agents <= set(AGENT_MODELS.keys())


def test_model_config_agent_assignments():
    from app.routing.model_config import AGENT_MODELS, MODEL_GPT5, MODEL_DEEPSEEK, MODEL_CLAUDE_SONNET, MODEL_MISTRAL_SMALL, MODEL_GPT5_MINI
    assert AGENT_MODELS["planner"]["default"] == MODEL_GPT5
    assert AGENT_MODELS["thesis"]["default"] == MODEL_GPT5
    assert AGENT_MODELS["reviewer"]["default"] == MODEL_GPT5
    assert AGENT_MODELS["research"]["cheap"] == MODEL_DEEPSEEK
    assert AGENT_MODELS["research"]["expensive"] == MODEL_GPT5
    assert AGENT_MODELS["writer"]["draft"] == MODEL_DEEPSEEK
    assert AGENT_MODELS["writer"]["refine"] == MODEL_CLAUDE_SONNET
    assert AGENT_MODELS["coherence"]["cheap"] == MODEL_MISTRAL_SMALL
    assert AGENT_MODELS["coherence"]["expensive"] == MODEL_GPT5
    assert AGENT_MODELS["verification"]["cheap"] == MODEL_GPT5_MINI
    assert AGENT_MODELS["verification"]["expensive"] == MODEL_GPT5
    assert AGENT_MODELS["citation"]["default"] == MODEL_GPT5_MINI


def test_writer_refine_sections():
    from app.routing.model_config import WRITER_REFINE_SECTIONS
    assert "introduction" in WRITER_REFINE_SECTIONS
    assert "conclusion" in WRITER_REFINE_SECTIONS
    # Body sections should NOT be refined with the expensive model
    assert "methodology" not in WRITER_REFINE_SECTIONS
    assert "results" not in WRITER_REFINE_SECTIONS


# ---------------------------------------------------------------------------
# llm_client._resolve_provider tests
# ---------------------------------------------------------------------------

def test_resolve_provider_default():
    from app.agents.llm_client import _resolve_provider
    from app.core.config import settings
    provider, model = _resolve_provider(None)
    assert provider == settings.LLM_PROVIDER.lower()
    assert model is None


def test_resolve_provider_openai_model():
    from app.agents.llm_client import _resolve_provider
    provider, model = _resolve_provider("gpt-5")
    assert provider == "openai"
    assert model == "gpt-5"


def test_resolve_provider_anthropic_model():
    from app.agents.llm_client import _resolve_provider
    provider, model = _resolve_provider("claude-3.5-sonnet")
    assert provider == "anthropic"
    assert model == "claude-3.5-sonnet"


def test_resolve_provider_deepseek_model():
    from app.agents.llm_client import _resolve_provider
    provider, model = _resolve_provider("deepseek-chat")
    assert provider == "deepseek"
    assert model == "deepseek-chat"


def test_resolve_provider_mistral_model():
    from app.agents.llm_client import _resolve_provider
    provider, model = _resolve_provider("mistral-small")
    assert provider == "mistral"
    assert model == "mistral-small"


def test_is_model_available_no_keys():
    """With no API keys configured, is_model_available should return False."""
    from app.agents.llm_client import is_model_available
    from app.core.config import settings
    # Temporarily clear keys
    orig_openai = settings.OPENAI_API_KEY
    orig_deepseek = settings.DEEPSEEK_API_KEY
    try:
        settings.OPENAI_API_KEY = ""
        settings.DEEPSEEK_API_KEY = ""
        assert is_model_available("gpt-5") is False
        assert is_model_available("deepseek-chat") is False
    finally:
        settings.OPENAI_API_KEY = orig_openai
        settings.DEEPSEEK_API_KEY = orig_deepseek


# ---------------------------------------------------------------------------
# router.route_task tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_task_returns_false_when_no_llm():
    """route_task should return False (no escalation) when LLM is unavailable."""
    from app.routing.router import route_task
    with patch("app.agents.llm_client.is_llm_available", return_value=False):
        result = await route_task("test task", "some input", "some output")
    assert result is False


@pytest.mark.asyncio
async def test_route_task_escalate_true():
    """route_task returns True when the routing model recommends escalation."""
    from app.routing.router import route_task
    escalate_response = json.dumps({"escalate": True, "reason": "shallow reasoning"})
    with patch("app.agents.llm_client.is_llm_available", return_value=True), \
         patch("app.agents.llm_client.chat_completion", new=AsyncMock(return_value=escalate_response)):
        result = await route_task("essay planning", "plan an essay", "A basic plan.")
    assert result is True


@pytest.mark.asyncio
async def test_route_task_escalate_false():
    """route_task returns False when the routing model does not recommend escalation."""
    from app.routing.router import route_task
    no_escalate_response = json.dumps({"escalate": False, "reason": "output is sufficient"})
    with patch("app.agents.llm_client.is_llm_available", return_value=True), \
         patch("app.agents.llm_client.chat_completion", new=AsyncMock(return_value=no_escalate_response)):
        result = await route_task("citation formatting", "format citations", "[1] Smith, 2020.")
    assert result is False


@pytest.mark.asyncio
async def test_route_task_handles_llm_error_gracefully():
    """route_task returns False (safe default) when the LLM call raises an exception."""
    from app.routing.router import route_task
    with patch("app.agents.llm_client.is_llm_available", return_value=True), \
         patch("app.agents.llm_client.chat_completion", new=AsyncMock(side_effect=RuntimeError("timeout"))):
        result = await route_task("grounding", "ground this", "some content")
    assert result is False


@pytest.mark.asyncio
async def test_route_task_handles_malformed_json():
    """route_task returns False when the LLM returns non-JSON output."""
    from app.routing.router import route_task
    with patch("app.agents.llm_client.is_llm_available", return_value=True), \
         patch("app.agents.llm_client.chat_completion", new=AsyncMock(return_value="not json at all")):
        result = await route_task("coherence review", "check coherence", "Some text.")
    assert result is False


# ---------------------------------------------------------------------------
# router prompt template test
# ---------------------------------------------------------------------------

def test_router_prompt_contains_required_fields():
    """The router prompt template must include all key decision criteria."""
    from app.routing.router import _ROUTER_PROMPT_TEMPLATE
    prompt = _ROUTER_PROMPT_TEMPLATE.format(task_type="test", model_output="test output")
    assert "escalate" in prompt
    assert "deep reasoning" in prompt
    assert "multi-step logic" in prompt
    assert "university-level essay" in prompt
    assert "JSON" in prompt
