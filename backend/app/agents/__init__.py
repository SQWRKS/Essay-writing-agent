from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.verification import VerificationAgent
from app.agents.writer import WriterAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.citation import CitationAgent
from app.agents.figure import FigureAgent

AGENT_REGISTRY = {
    "planner": PlannerAgent,
    "research": ResearchAgent,
    "verification": VerificationAgent,
    "writer": WriterAgent,
    "reviewer": ReviewerAgent,
    "citation": CitationAgent,
    "figure": FigureAgent,
}
