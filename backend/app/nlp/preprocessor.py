"""Text preprocessing pipeline — no LLM, no heavy external deps.

Responsibilities
----------------
1. **clean_text** — remove noise: HTML tags, bracket citations, URLs, extra
   whitespace, junk characters, LaTeX artefacts.
2. **chunk_text** — split a long document into overlapping windows of
   approximately *chunk_size* tokens (words).
3. **detect_sections** — heuristically identify the section each sentence
   belongs to (abstract, introduction, methods, results, discussion, conclusion)
   based on keyword triggers in heading-like lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    """A chunk of text with positional metadata."""

    text: str
    index: int                     # chunk index within the document
    char_start: int                # character offset in the original document
    char_end: int
    section: Optional[str] = None  # detected section label, if any


@dataclass
class ProcessedDocument:
    """Holds the cleaned text, chunks, and section boundaries for one document."""

    original_text: str
    cleaned_text: str
    chunks: list[TextChunk] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)  # label -> text span


# Section trigger keywords (case-insensitive)
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background", "overview"],
    "methods": ["method", "methodology", "approach", "experimental", "procedure"],
    "results": ["result", "finding", "outcome", "experiment"],
    "discussion": ["discussion", "analysis", "interpretation"],
    "conclusion": ["conclusion", "summary", "future work", "closing"],
    "references": ["references", "bibliography", "works cited"],
}

# Compiled noise patterns
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_BRACKET_CITE = re.compile(r"\[\s*\d+(?:,\s*\d+)*\s*\]")  # [1], [1,2,3]
_RE_PAREN_CITE = re.compile(r"\(\s*[A-Z][a-z]+\s*(?:et\s+al\.?)?\s*,?\s*\d{4}\s*\)")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_LATEX = re.compile(r"\$[^$]+\$|\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_JUNK_CHARS = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF]")


class Preprocessor:
    """Rule-based text preprocessing pipeline.

    Parameters
    ----------
    chunk_size:
        Approximate number of words per chunk (default 150).
    chunk_overlap:
        Number of words to overlap between adjacent chunks (default 30).
    """

    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> ProcessedDocument:
        """Full preprocessing pipeline on a single document.

        Returns a :class:`ProcessedDocument` with cleaned text, chunks, and
        detected section labels.
        """
        cleaned = self.clean_text(text)
        chunks = self.chunk_text(cleaned)
        sections = self.detect_sections(cleaned)
        # Attach section labels to each chunk
        for chunk in chunks:
            chunk.section = self._section_for_chunk(chunk, sections, cleaned)
        return ProcessedDocument(
            original_text=text,
            cleaned_text=cleaned,
            chunks=chunks,
            sections=sections,
        )

    def clean_text(self, text: str) -> str:
        """Return a cleaned version of *text*.

        Steps (in order):
        1. Strip HTML tags
        2. Remove bracket and parenthetical citations
        3. Strip URLs
        4. Remove LaTeX markup
        5. Remove junk / non-printable characters
        6. Collapse excessive whitespace
        """
        if not text:
            return ""
        t = _RE_HTML_TAG.sub(" ", text)
        t = _RE_BRACKET_CITE.sub(" ", t)
        t = _RE_PAREN_CITE.sub(" ", t)
        t = _RE_URL.sub(" ", t)
        t = _RE_LATEX.sub(" ", t)
        t = _RE_JUNK_CHARS.sub("", t)
        t = _RE_MULTI_SPACE.sub(" ", t)
        t = _RE_MULTI_NEWLINE.sub("\n\n", t)
        return t.strip()

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split *text* into overlapping word-window chunks.

        Returns a list of :class:`TextChunk` objects ordered by position.
        """
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = max(1, self.chunk_size)

        idx = 0
        word_pos = 0
        while word_pos < len(words):
            window = words[word_pos: word_pos + self.chunk_size]
            chunk_text = " ".join(window)

            # Approximate character offsets via a forward search
            char_start = text.find(window[0], sum(len(w) + 1 for w in words[:word_pos]))
            char_end = char_start + len(chunk_text)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=idx,
                    char_start=max(0, char_start),
                    char_end=char_end,
                )
            )
            idx += 1
            word_pos += step

        return chunks

    def detect_sections(self, text: str) -> dict[str, str]:
        """Return a mapping of section_label → text span.

        Uses heuristic heading detection: lines that consist mostly of
        uppercase or title-case words and match a known section keyword are
        treated as heading boundaries.
        """
        lines = text.splitlines()
        sections: dict[str, str] = {}
        current_label: Optional[str] = None
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            heading_match = self._match_heading(stripped)
            if heading_match:
                if current_label and current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        # Append if section already started (can have multiple
                        # sub-headings map to same canonical label)
                        if current_label in sections:
                            sections[current_label] += "\n" + body
                        else:
                            sections[current_label] = body
                current_label = heading_match
                current_lines = []
            else:
                current_lines.append(line)

        # Flush last section
        if current_label and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                if current_label in sections:
                    sections[current_label] += "\n" + body
                else:
                    sections[current_label] = body

        return sections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_heading(line: str) -> Optional[str]:
        """Return a canonical section label if *line* looks like a section heading."""
        if not line or len(line) > 80 or len(line.split()) < 1:
            return None
        lower = line.lower().strip("# \t")
        for label, triggers in _SECTION_KEYWORDS.items():
            for trigger in triggers:
                if trigger in lower and len(lower) < 60:
                    return label
        return None

    def _section_for_chunk(
        self,
        chunk: TextChunk,
        sections: dict[str, str],
        full_text: str,
    ) -> Optional[str]:
        """Return the section label for a chunk by character-offset lookup."""
        if not sections:
            return None
        # Find which section span contains chunk's mid-point
        mid = (chunk.char_start + chunk.char_end) // 2
        running = 0
        for label, body in sections.items():
            start = full_text.find(body[:40])
            if start == -1:
                continue
            end = start + len(body)
            if start <= mid <= end:
                return label
            running = end
        return None
