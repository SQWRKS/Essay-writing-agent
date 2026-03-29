from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.verification import VerificationAgent
from app.agents.writer import WriterAgent
from app.agents.grounding import GroundingAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.coherence import CoherenceAgent
from app.agents.citation import CitationAgent
from app.agents.figure import FigureAgent

AGENT_REGISTRY = {
    "planner": PlannerAgent,
    "research": ResearchAgent,
    "verification": VerificationAgent,
    "writer": WriterAgent,
    "grounding": GroundingAgent,
    "reviewer": ReviewerAgent,
    "coherence": CoherenceAgent,
    "citation": CitationAgent,
    "figure": FigureAgent,
}
