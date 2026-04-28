"""
Model configuration for cost-optimised LLM routing.

All model name constants, provider mappings, and per-agent model assignments
are defined here.  To change which model an agent uses, update only this file.
"""

# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

# High-cost models
MODEL_GPT5: str = "gpt-5"                      # reasoning, planning, verification, reviewing
MODEL_CLAUDE_SONNET: str = "claude-3.5-sonnet"  # high-quality writing / refinement

# Low-cost models
MODEL_DEEPSEEK: str = "deepseek-chat"           # drafting, summarisation
MODEL_MISTRAL_SMALL: str = "mistral-small"      # rewriting, grounding, coherence

# Ultra-low-cost model
MODEL_GPT5_MINI: str = "gpt-5-mini"             # citation formatting, simple transforms

# Model used internally by the router to evaluate cheap-model output
ROUTER_MODEL: str = MODEL_GPT5_MINI

# ---------------------------------------------------------------------------
# Provider map: model name → provider identifier
# Supported providers: "openai", "anthropic", "deepseek", "mistral"
# ---------------------------------------------------------------------------
MODEL_PROVIDER_MAP: dict[str, str] = {
    MODEL_GPT5: "openai",
    MODEL_GPT5_MINI: "openai",
    MODEL_CLAUDE_SONNET: "anthropic",
    MODEL_DEEPSEEK: "deepseek",
    MODEL_MISTRAL_SMALL: "mistral",
}

# ---------------------------------------------------------------------------
# Per-agent model assignments
# Keys per agent:
#   "default"   – single model for agents without routing
#   "cheap"     – initial low-cost call before routing decision
#   "expensive" – escalation target when router recommends it
#   "draft"     / "refine" – WriterAgent-specific split
#   "reasoning" / "formatting" – FigureAgent-specific split
# ---------------------------------------------------------------------------
AGENT_MODELS: dict[str, dict[str, str]] = {
    "planner": {
        "default": MODEL_GPT5,
    },
    "research": {
        "cheap": MODEL_DEEPSEEK,       # drafting / per-source summarisation
        "expensive": MODEL_GPT5,       # escalate when synthesising multiple sources
    },
    "thesis": {
        "default": MODEL_GPT5,          # backward-compat alias for expensive model
        "cheap": MODEL_DEEPSEEK,        # initial cheap generation pass
        "expensive": MODEL_GPT5,        # escalate when thesis is weak or vague
    },
    "verification": {
        "cheap": MODEL_GPT5_MINI,      # optional flagging pre-check
        "expensive": MODEL_GPT5,       # used for flagged / complex credibility assessment
    },
    "writer": {
        "draft": MODEL_DEEPSEEK,       # full section draft
        "refine": MODEL_CLAUDE_SONNET, # intro, conclusion, and key argument sections only
    },
    "grounding": {
        # Currently a heuristic-only agent; model reserved for future LLM integration
        "default": MODEL_MISTRAL_SMALL,
    },
    "reviewer": {
        "default": MODEL_GPT5,
    },
    "coherence": {
        "cheap": MODEL_MISTRAL_SMALL,  # default coherence check
        "expensive": MODEL_GPT5,       # escalate when restructuring is required
    },
    "citation": {
        # Currently a rule-based agent; model reserved for future LLM integration
        "default": MODEL_GPT5_MINI,
    },
    "plagiarism": {
        "default": MODEL_GPT5_MINI,    # lightweight assessment of flagged passages
    },
    "figure": {
        # Currently a matplotlib-based agent; models reserved for future LLM integration
        "reasoning": MODEL_GPT5,
        "formatting": MODEL_MISTRAL_SMALL,
    },
    "web_search": {
        # Currently a no-LLM agent; model reserved for future summarisation use
        "default": MODEL_DEEPSEEK,
    },
}

# Sections that qualify for expensive-model refinement in WriterAgent
WRITER_REFINE_SECTIONS: frozenset[str] = frozenset({"introduction", "conclusion"})
