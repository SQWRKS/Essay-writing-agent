import os
import json
import uuid
from app.agents.base import AgentBase

# Save figures relative to the backend directory (parent of app/)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(BACKEND_DIR, "static", "figures")


class FigureAgent(AgentBase):
    name = "figure"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")
        section = input_data.get("section", "results")
        data = input_data.get("data", {})

        figures = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(FIGURES_DIR, exist_ok=True)

            # Figure 1: Bar chart of research metrics
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = data.get("categories", ["Accuracy", "Precision", "Recall", "F1-Score"])
            values = data.get("values", [0.85, 0.82, 0.88, 0.85])
            ax.bar(categories, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
            ax.set_title(f"Performance Metrics - {topic[:40]}", fontsize=12)
            ax.set_ylabel("Score")
            max_value = max(values) if values else 1
            y_max = 1.0 if max_value <= 1.0 else max_value * 1.2
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Metric")
            plt.tight_layout()
            fname = f"fig_{project_id[:8]}_{uuid.uuid4().hex[:6]}.png"
            fpath = os.path.join(FIGURES_DIR, fname)
            plt.savefig(fpath, dpi=100, bbox_inches="tight")
            plt.close(fig)
            figures.append({
                "title": f"Performance Metrics for {topic}",
                "path": fpath,
                "url": f"/static/figures/{fname}",
                "section": section,
                "description": "Bar chart showing key performance metrics.",
                "data": {"categories": categories, "values": values},
            })

            # Figure 2: Line chart for trends
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            years = data.get("years", list(range(2018, 2025)))
            trend = data.get("trend", [i * 0.05 + 0.5 for i in range(len(years))])
            trend_label = data.get("trend_label", "Relative Impact")
            ax2.plot(years, trend, marker="o", color="#4C72B0", linewidth=2)
            ax2.set_title(f"Research Trend - {topic[:40]}", fontsize=12)
            ax2.set_xlabel("Year")
            ax2.set_ylabel(trend_label)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            fname2 = f"fig_{project_id[:8]}_{uuid.uuid4().hex[:6]}_trend.png"
            fpath2 = os.path.join(FIGURES_DIR, fname2)
            plt.savefig(fpath2, dpi=100, bbox_inches="tight")
            plt.close(fig2)
            figures.append({
                "title": f"Research Trend for {topic}",
                "path": fpath2,
                "url": f"/static/figures/{fname2}",
                "section": section,
                "description": "Line chart showing research trend over time.",
                "data": {"years": years, "trend": trend},
            })
        except Exception as e:
            figures.append({
                "title": f"Figure for {topic}",
                "path": "",
                "description": f"Figure generation failed: {e}",
                "data": {},
            })

        result = {"figures": figures}
        await self._update_agent_state(db, project_id, "completed", result)
        return result
