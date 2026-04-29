import json
import logging

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion
from app.routing.model_config import AGENT_MODELS

logger = logging.getLogger(__name__)

# Edge-case types where the rule-based formatter may produce malformed entries.
_LLM_ESCALATION_TYPES = frozenset({"conference", "chapter", "dataset", "proceedings", "report"})


class CitationAgent(AgentBase):
    name = "citation"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")
        sources = input_data.get("sources", [])
        style = input_data.get("style", "harvard")

        formatted = []
        in_text = {}
        bib_lines = []

        for i, source in enumerate(sources):
            citation, in_text_key = self._format_source(i + 1, source, style)

            # Optional LLM refinement for edge-case source types.
            # Substring match is intentional: "conference proceedings" ⊇ "proceedings".
            if is_llm_available():
                src_type = (source.get("type") or source.get("venue") or "").lower()
                if any(t in src_type for t in _LLM_ESCALATION_TYPES):
                    try:
                        citation = await self._llm_format(source, style, citation, db)
                    except Exception as exc:
                        logger.debug("LLM citation refinement failed (non-fatal): %s", exc)

            title = source.get("title", "Untitled")
            in_text[title] = in_text_key
            formatted.append(citation)
            bib_lines.append(citation)

        result = {
            "formatted_citations": formatted,
            "bibliography": "\n".join(bib_lines),
            "in_text_citations": in_text,
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result

    # ------------------------------------------------------------------
    # Rule-based formatters
    # ------------------------------------------------------------------

    def _format_source(self, number: int, source: dict, style: str) -> tuple[str, str]:
        """Return (formatted_citation, in_text_key) for a given style."""
        authors = source.get("authors") or ["Unknown"]
        year = source.get("year", "n.d.")
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        doi = source.get("doi", "")

        if style == "ieee":
            return self._format_ieee(number, authors, year, title, doi), f"[{number}]"
        if style == "apa":
            return self._format_apa(authors, year, title, doi, url), self._apa_in_text(authors, year)
        if style == "mla":
            return self._format_mla(authors, year, title, doi, url), self._mla_in_text(authors)
        # default: harvard
        return self._format_harvard(authors, year, title, doi, url), self._harvard_in_text(authors, year)

    # Harvard
    def _format_harvard(self, authors: list, year, title: str, doi: str, url: str) -> str:
        author_str = "; ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        citation = f"{author_str} ({year}) '{title}'."
        if doi:
            citation += f" Available at: https://doi.org/{doi}."
        elif url:
            citation += f" Available at: {url}."
        return citation

    def _harvard_in_text(self, authors: list, year) -> str:
        first = (authors[0].split(",")[0] if authors else "Unknown").strip()
        return f"({first}, {year})"

    # IEEE
    def _format_ieee(self, number: int, authors: list, year, title: str, doi: str) -> str:
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        citation = f"[{number}] {author_str}, \"{title},\" {year}."
        if doi:
            citation += f" DOI: {doi}."
        return citation

    # APA 7th edition
    def _format_apa(self, authors: list, year, title: str, doi: str, url: str) -> str:
        # Author format: Last, F. I., & Last, F. I.
        formatted_authors = []
        for a in authors[:6]:
            parts = [p.strip() for p in a.split(",") if p.strip()]
            if len(parts) >= 2:
                initials = " ".join(p[0].upper() + "." for p in parts[1:] if p)
                formatted_authors.append(f"{parts[0]}, {initials}".strip())
            else:
                formatted_authors.append(a)
        if len(authors) > 6:
            formatted_authors.append("...")
        if len(formatted_authors) > 1:
            author_str = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
        else:
            author_str = formatted_authors[0] if formatted_authors else "Unknown"
        citation = f"{author_str} ({year}). {title}."
        if doi:
            citation += f" https://doi.org/{doi}"
        elif url:
            citation += f" {url}"
        return citation

    def _apa_in_text(self, authors: list, year) -> str:
        if not authors:
            return f"(Unknown, {year})"
        last = (authors[0].split(",")[0] if "," in authors[0] else authors[0]).strip()
        if len(authors) == 2:
            last2 = (authors[1].split(",")[0] if "," in authors[1] else authors[1]).strip()
            return f"({last} & {last2}, {year})"
        if len(authors) > 2:
            return f"({last} et al., {year})"
        return f"({last}, {year})"

    # MLA 9th edition
    def _format_mla(self, authors: list, year, title: str, doi: str, url: str) -> str:
        if not authors:
            author_str = "Unknown"
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]}, and {authors[1]}"
        else:
            author_str = f"{authors[0]}, et al."
        citation = f'{author_str}. "{title}." {year}.'
        if doi:
            citation += f" DOI: {doi}."
        elif url:
            citation += f" {url}."
        return citation

    def _mla_in_text(self, authors: list) -> str:
        if not authors:
            return "(Unknown)"
        last = (authors[0].split(",")[0] if "," in authors[0] else authors[0]).strip()
        return f"({last})"

    # ------------------------------------------------------------------
    # LLM refinement for edge-case source types
    # ------------------------------------------------------------------

    async def _llm_format(self, source: dict, style: str, rule_based: str, db) -> str:
        """Ask the LLM to produce a corrected citation for unusual source types."""
        prompt = (
            f"You are a citation formatter. Produce a single correctly formatted {style.upper()} citation "
            f"for the following source. Return only the citation string, no explanation.\n\n"
            f"Source metadata: {json.dumps(source)}\n\n"
            f"Current rule-based attempt: {rule_based}"
        )
        response = await timed_chat_completion(
            prompt,
            db=db,
            agent_name=self.name,
            log_api_call_fn=self._log_api_call,
            model=AGENT_MODELS["citation"]["default"],
            temperature=0.0,
            max_tokens=200,
        )
        refined = (response or "").strip()
        return refined if refined else rule_based

