import json
import logging
import os
import uuid

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.routing.model_config import AGENT_MODELS

logger = logging.getLogger(__name__)

# Save figures relative to the backend directory (parent of app/)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(BACKEND_DIR, "static", "figures")

# Keys that indicate the data dict carries temporal / trend information.
_TREND_KEYS = frozenset({"years", "year", "dates", "time", "trend", "over_time"})
# Keys that indicate scatter / correlation data.
_SCATTER_KEYS = frozenset({"x", "y", "x_values", "y_values", "scatter"})


def _detect_figure_type(data: dict) -> str:
    """Choose the most appropriate figure type based on the keys present in *data*."""
    keys = frozenset(data.keys())
    if keys & _TREND_KEYS:
        return "line"
    if keys & _SCATTER_KEYS:
        return "scatter"
    if "table" in keys or "rows" in keys:
        return "table"
    # default: bar chart for categorical metrics
    return "bar"


class FigureAgent(AgentBase):
    name = "figure"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        topic = input_data.get("topic", "")
        section = input_data.get("section", "results")
        data = input_data.get("data", {})

        # Determine figure type from data keys
        figure_type = _detect_figure_type(data)

        # Optionally generate a richer description / caption via LLM
        caption = await self._llm_caption(topic, section, figure_type, data, db)

        figures = self._render_figures(topic, section, project_id, data, figure_type, caption)

        result = {"figures": figures}
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    # ------------------------------------------------------------------
    # LLM caption generation
    # ------------------------------------------------------------------

    async def _llm_caption(
        self, topic: str, section: str, figure_type: str, data: dict, db
    ) -> str:
        """Use the reasoning model to generate a descriptive figure caption."""
        if not is_llm_available():
            return ""
        try:
            prompt = (
                f"Write a concise, informative figure caption (1–2 sentences) for a {figure_type} chart "
                f"in the '{section}' section of an academic paper on '{topic}'. "
                f"Available data keys: {list(data.keys())}. "
                "Return only the caption text."
            )
            response = await timed_chat_completion(
                prompt,
                db=db,
                agent_name=self.name,
                log_api_call_fn=self._log_api_call,
                model=AGENT_MODELS["figure"]["reasoning"],
                temperature=0.3,
                max_tokens=120,
            )
            return (response or "").strip()
        except Exception as exc:
            logger.debug("FigureAgent caption generation failed (non-fatal): %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Figure rendering
    # ------------------------------------------------------------------

    def _render_figures(
        self,
        topic: str,
        section: str,
        project_id: str,
        data: dict,
        figure_type: str,
        caption: str,
    ) -> list[dict]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed — returning descriptive placeholder instead of rendered figure."
            )
            return [
                {
                    "title": f"Figure for {topic}",
                    "path": "",
                    "url": "",
                    "section": section,
                    "description": caption or f"Data-driven {figure_type} chart for {topic}.",
                    "data": data,
                    "figure_type": figure_type,
                }
            ]

        figures = []
        try:
            os.makedirs(FIGURES_DIR, exist_ok=True)

            if figure_type == "line":
                figures.append(self._render_line(topic, section, project_id, data, caption, plt))
            elif figure_type == "scatter":
                figures.append(self._render_scatter(topic, section, project_id, data, caption, plt))
            elif figure_type == "table":
                figures.append(self._render_table(topic, section, project_id, data, caption, plt))
            else:
                figures.append(self._render_bar(topic, section, project_id, data, caption, plt))

        except Exception as exc:
            logger.warning("FigureAgent rendering failed (non-fatal): %s", exc)
            figures.append(
                {
                    "title": f"Figure for {topic}",
                    "path": "",
                    "url": "",
                    "section": section,
                    "description": caption or f"Figure generation encountered an error: {exc}",
                    "data": data,
                    "figure_type": figure_type,
                }
            )
        return figures

    def _save_fig(self, fig, project_id: str, suffix: str, plt) -> tuple[str, str]:
        fname = f"fig_{project_id[:8]}_{uuid.uuid4().hex[:6]}{suffix}.png"
        fpath = os.path.join(FIGURES_DIR, fname)
        plt.savefig(fpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return fpath, f"/static/figures/{fname}"

    def _render_bar(self, topic, section, project_id, data, caption, plt) -> dict:
        categories = data.get("categories", ["Accuracy", "Precision", "Recall", "F1-Score"])
        values = data.get("values", [0.85, 0.82, 0.88, 0.85])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(categories, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_title(f"Performance Metrics — {topic[:40]}", fontsize=12)
        ax.set_ylabel("Score")
        max_v = max(values) if values else 1
        ax.set_ylim(0, 1.0 if max_v <= 1.0 else max_v * 1.2)
        ax.set_xlabel("Metric")
        plt.tight_layout()
        fpath, url = self._save_fig(fig, project_id, "", plt)
        return {
            "title": f"Performance Metrics for {topic}",
            "path": fpath,
            "url": url,
            "section": section,
            "description": caption or "Bar chart showing key performance metrics.",
            "data": {"categories": categories, "values": values},
            "figure_type": "bar",
        }

    def _render_line(self, topic, section, project_id, data, caption, plt) -> dict:
        years = data.get("years", list(range(2018, 2025)))
        trend = data.get("trend", [i * 0.05 + 0.5 for i in range(len(years))])
        label = data.get("trend_label", "Relative Impact")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, trend, marker="o", color="#4C72B0", linewidth=2)
        ax.set_title(f"Research Trend — {topic[:40]}", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fpath, url = self._save_fig(fig, project_id, "_trend", plt)
        return {
            "title": f"Research Trend for {topic}",
            "path": fpath,
            "url": url,
            "section": section,
            "description": caption or "Line chart showing research trend over time.",
            "data": {"years": years, "trend": trend},
            "figure_type": "line",
        }

    def _render_scatter(self, topic, section, project_id, data, caption, plt) -> dict:
        x = data.get("x", data.get("x_values", [1, 2, 3, 4, 5]))
        y = data.get("y", data.get("y_values", [2, 4, 3, 5, 4]))
        x_label = data.get("x_label", "X")
        y_label = data.get("y_label", "Y")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="#4C72B0", alpha=0.7)
        ax.set_title(f"Correlation — {topic[:40]}", fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        fpath, url = self._save_fig(fig, project_id, "_scatter", plt)
        return {
            "title": f"Scatter Plot for {topic}",
            "path": fpath,
            "url": url,
            "section": section,
            "description": caption or "Scatter plot showing variable correlation.",
            "data": {"x": x, "y": y},
            "figure_type": "scatter",
        }

    def _render_table(self, topic, section, project_id, data, caption, plt) -> dict:
        rows = data.get("rows", [["A", "B", "C"], ["1", "2", "3"]])
        headers = data.get("headers", data.get("columns", [f"Col {i+1}" for i in range(len(rows[0]) if rows else 3)]))
        fig, ax = plt.subplots(figsize=(8, max(2, len(rows) * 0.5 + 1)))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title(f"Summary Table — {topic[:40]}", fontsize=12, pad=12)
        plt.tight_layout()
        fpath, url = self._save_fig(fig, project_id, "_table", plt)
        return {
            "title": f"Data Table for {topic}",
            "path": fpath,
            "url": url,
            "section": section,
            "description": caption or "Summary table of key data.",
            "data": {"headers": headers, "rows": rows},
            "figure_type": "table",
        }

