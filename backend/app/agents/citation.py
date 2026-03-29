from app.agents.base import AgentBase


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
            authors = source.get("authors", ["Unknown"])
            year = source.get("year", "n.d.")
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            doi = source.get("doi", "")

            if style == "ieee":
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                citation = f"[{i+1}] {author_str}, \"{title},\" {year}."
                if doi:
                    citation += f" DOI: {doi}."
                in_text[title] = f"[{i+1}]"
            else:  # harvard
                first_author = authors[0].split(",")[0] if authors else "Unknown"
                author_str = "; ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                citation = f"{author_str} ({year}) '{title}'."
                if doi:
                    citation += f" Available at: https://doi.org/{doi}."
                in_text[title] = f"({first_author}, {year})"

            formatted.append(citation)
            bib_lines.append(citation)

        result = {
            "formatted_citations": formatted,
            "bibliography": "\n".join(bib_lines),
            "in_text_citations": in_text,
        }
        await self._update_agent_state(db, project_id, "completed", result)
        return result
