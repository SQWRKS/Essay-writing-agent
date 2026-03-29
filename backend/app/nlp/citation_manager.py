"""Citation extraction, formatting, and validation — no LLM required.

CitationManager handles:
1. **Extraction** — find candidate references embedded in research source
   metadata (title, authors, year, DOI, venue).
2. **Formatting** — produce APA-7 or Harvard in-text + bibliography entries.
3. **Validation** — check that required fields (author, year, title) are
   present; reject sources with no usable citation data.

Design principles:
- Pure Python, no network calls.
- Tolerant input parsing (partial metadata is handled gracefully).
- Output is deterministic for the same input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Citation:
    """A validated, formatted citation record."""

    title: str
    authors: list[str]
    year: int
    doi: str = ""
    url: str = ""
    venue: str = ""
    source: str = ""

    # Populated by CitationManager.format()
    apa: str = ""
    harvard: str = ""
    in_text_apa: str = ""
    in_text_harvard: str = ""

    is_valid: bool = False
    validation_issues: list[str] = field(default_factory=list)


class CitationManager:
    """Extract, format, and validate citations from source metadata.

    Parameters
    ----------
    max_authors_display:
        Maximum number of authors to show before applying "et al.".
        APA-7 uses 20 (full list ≤ 20), then "et al."; Harvard typically 3.
    """

    def __init__(self, max_authors_display: int = 3) -> None:
        self.max_authors_display = max(1, max_authors_display)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_sources(self, sources: list[dict]) -> list[Citation]:
        """Extract, validate, and format citations from a list of source dicts.

        Each dict should have (at minimum) keys: ``title``, ``authors``,
        ``year``.  Optional: ``doi``, ``url``, ``venue``, ``source``.

        Returns a list of :class:`Citation` objects — one per input source,
        in the same order.
        """
        citations: list[Citation] = []
        for src in sources:
            cit = self._extract(src)
            cit = self._validate(cit)
            if cit.is_valid:
                cit = self._format(cit)
            citations.append(cit)
        return citations

    def bibliography(
        self,
        citations: list[Citation],
        style: str = "apa",
        include_invalid: bool = False,
    ) -> str:
        """Return a formatted bibliography string.

        Parameters
        ----------
        citations:
            List of :class:`Citation` objects (from :meth:`process_sources`).
        style:
            ``"apa"`` (default) or ``"harvard"``.
        include_invalid:
            When ``True``, invalid citations are listed with a warning marker.
        """
        valid = [c for c in citations if c.is_valid or include_invalid]
        if not valid:
            return ""

        style = style.lower()
        lines: list[str] = []
        for i, cit in enumerate(sorted(valid, key=lambda c: (c.authors[0] if c.authors else "", c.year)), 1):
            if not cit.is_valid:
                entry = f"[INVALID] {cit.title or 'Unknown title'}"
            elif style == "harvard":
                entry = cit.harvard or cit.apa
            else:
                entry = cit.apa or cit.harvard
            lines.append(entry)

        return "\n".join(lines)

    def validate_fields(self, sources: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split *sources* into (valid, invalid) lists based on citation fields.

        A source is considered **valid** if it has a non-empty title, at least
        one author, and a year in a plausible range (1000–2100).
        """
        valid, invalid = [], []
        for src in sources:
            cit = self._extract(src)
            cit = self._validate(cit)
            if cit.is_valid:
                valid.append(src)
            else:
                invalid.append(src)
        return valid, invalid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract(self, src: dict) -> Citation:
        """Build a :class:`Citation` from a raw source dict."""
        raw_authors = src.get("authors") or []
        if isinstance(raw_authors, str):
            raw_authors = [a.strip() for a in re.split(r"[,;]", raw_authors) if a.strip()]

        authors = [str(a).strip() for a in raw_authors if str(a).strip()]

        raw_year = src.get("year")
        try:
            year = int(raw_year) if raw_year else 0
        except (TypeError, ValueError):
            year = 0

        return Citation(
            title=(src.get("title") or "").strip(),
            authors=authors,
            year=year,
            doi=(src.get("doi") or "").strip(),
            url=(src.get("url") or "").strip(),
            venue=(src.get("venue") or "").strip(),
            source=(src.get("source") or "").strip(),
        )

    @staticmethod
    def _validate(cit: Citation) -> Citation:
        """Populate ``is_valid`` and ``validation_issues`` on *cit*."""
        issues: list[str] = []
        if not cit.title:
            issues.append("missing title")
        if not cit.authors:
            issues.append("missing authors")
        if not (1000 <= cit.year <= 2100):
            issues.append(f"implausible year: {cit.year!r}")
        if cit.doi and not re.match(r"^10\.\d{4,}/", cit.doi):
            issues.append(f"malformed DOI: {cit.doi!r}")

        cit.validation_issues = issues
        cit.is_valid = len(issues) == 0
        return cit

    def _format(self, cit: Citation) -> Citation:
        """Populate APA and Harvard formatted strings on *cit*."""
        cit.apa = self._format_apa(cit)
        cit.harvard = self._format_harvard(cit)
        cit.in_text_apa = self._in_text_apa(cit)
        cit.in_text_harvard = self._in_text_harvard(cit)
        return cit

    # ---- APA-7 formatting ----

    def _format_apa(self, cit: Citation) -> str:
        """APA 7th edition bibliography entry."""
        author_str = self._apa_authors(cit.authors)
        year_str = f"({cit.year})" if cit.year else "(n.d.)"
        title_str = cit.title if cit.title else "Untitled"

        parts = [f"{author_str} {year_str}. {title_str}."]

        if cit.venue:
            parts[0] = parts[0].rstrip(".")
            parts.append(f" *{cit.venue}*.")

        if cit.doi:
            parts.append(f" https://doi.org/{cit.doi}")
        elif cit.url:
            parts.append(f" {cit.url}")

        return "".join(parts).strip()

    def _apa_authors(self, authors: list[str]) -> str:
        """Format authors in APA style: Last, F. M., & Last, F. M."""
        if not authors:
            return "Anonymous"
        formatted: list[str] = []
        for author in authors[: self.max_authors_display]:
            formatted.append(self._apa_author_name(author))
        if len(authors) > self.max_authors_display:
            formatted.append("et al.")
            return ", ".join(formatted[:-1]) + ", " + formatted[-1]
        if len(formatted) == 1:
            return formatted[0]
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    @staticmethod
    def _apa_author_name(name: str) -> str:
        """Attempt to reformat 'First Last' → 'Last, F.' for APA."""
        name = name.strip()
        # Already in 'Last, F.' or 'Last, First' format
        if "," in name:
            parts = [p.strip() for p in name.split(",", 1)]
            last = parts[0]
            given = parts[1].strip() if len(parts) > 1 else ""
            initials = " ".join(f"{w[0]}." for w in given.split() if w) if given else ""
            return f"{last}, {initials}".rstrip(", ")
        # 'First Last' format
        tokens = name.split()
        if len(tokens) >= 2:
            last = tokens[-1]
            initials = " ".join(f"{w[0]}." for w in tokens[:-1])
            return f"{last}, {initials}"
        return name

    def _in_text_apa(self, cit: Citation) -> str:
        """APA in-text citation: (Last, Year)."""
        author_last = self._first_author_last(cit.authors)
        year = str(cit.year) if cit.year else "n.d."
        return f"({author_last}, {year})"

    # ---- Harvard formatting ----

    def _format_harvard(self, cit: Citation) -> str:
        """Harvard bibliography entry."""
        author_str = self._harvard_authors(cit.authors)
        year_str = str(cit.year) if cit.year else "n.d."
        title_str = f"'{cit.title}'" if cit.title else "'Untitled'"

        parts = [f"{author_str} ({year_str}) {title_str}"]

        if cit.venue:
            parts.append(f", *{cit.venue}*")

        if cit.doi:
            parts.append(f". doi:{cit.doi}")
        elif cit.url:
            parts.append(f". Available at: {cit.url}")

        return "".join(parts).strip() + "."

    def _harvard_authors(self, authors: list[str]) -> str:
        """Format authors in Harvard style: Last, F. and Last, F."""
        if not authors:
            return "Anon"
        formatted = [self._apa_author_name(a) for a in authors[: self.max_authors_display]]
        if len(authors) > self.max_authors_display:
            return ", ".join(formatted) + " et al."
        if len(formatted) == 1:
            return formatted[0]
        return " and ".join([", ".join(formatted[:-1]), formatted[-1]])

    def _in_text_harvard(self, cit: Citation) -> str:
        """Harvard in-text citation: (Last, Year)."""
        author_last = self._first_author_last(cit.authors)
        year = str(cit.year) if cit.year else "n.d."
        return f"({author_last}, {year})"

    # ---- Shared helper ----

    @staticmethod
    def _first_author_last(authors: list[str]) -> str:
        """Extract the last name of the first author."""
        if not authors:
            return "Anon"
        name = authors[0].strip()
        if "," in name:
            return name.split(",")[0].strip()
        tokens = name.split()
        return tokens[-1] if tokens else "Anon"
