import json
import re
from collections import Counter

from app.agents.base import AgentBase
from app.agents.llm_client import is_llm_available, timed_chat_completion


SECTION_GUIDANCE = {
    "introduction": "frame the engineering problem, define the thesis, and establish the stakes of the topic",
    "literature_review": "compare competing approaches, identify gaps, and synthesise the strongest evidence",
    "methodology": "justify methods, datasets, assumptions, and evaluation criteria with technical precision",
    "results": "report comparative findings, quantitative performance, and notable patterns from the evidence",
    "discussion": "interpret the implications of the evidence, trade-offs, and unresolved limitations",
    "conclusion": "summarise the argument, main technical contributions, and future work implied by the evidence",
}

PROMPT_LEAK_PATTERNS = [
    "requirements:",
    "return only",
    "you are writing",
    "target length",
    "section-specific",
    "use inline citation markers",
]

GENERIC_PHRASES = [
    "this paper examines",
    "this study explores",
    "has gained considerable attention",
    "plays an important role",
    "in today's world",
    "it is clear that",
    "the significance of this research",
]


class WriterAgent(AgentBase):
    name = "writer"

    async def execute(self, input_data: dict, project_id: str, db) -> dict:
        await self._update_agent_state(db, project_id, "running")

        payload = self._normalize_inputs(input_data)
        result = await self._generate_validated_output(payload, project_id, db)

        await self._update_agent_state(db, project_id, "completed", result)
        return result

    def _normalize_inputs(self, input_data: dict) -> dict:
        research_data = input_data.get("research_data", {})
        topic = input_data.get("topic", "the topic")
        section = input_data.get("section", "introduction")
        thesis = input_data.get("thesis") or research_data.get("thesis") or ""
        summaries = research_data.get("summaries", [])
        evidence_pack = research_data.get("evidence_pack", [])

        research_notes = input_data.get("research_notes") or self._notes_from_research(summaries, evidence_pack)
        domain_keywords = self._extract_domain_keywords(topic, thesis, research_notes)

        return {
            "topic": topic,
            "section": section,
            "word_count": int(input_data.get("word_count", 500)),
            "thesis": thesis,
            "research_notes": research_notes,
            "domain_keywords": domain_keywords,
            "feedback": input_data.get("feedback", ""),
            "section_queries": research_data.get("section_queries", []),
            "sources": research_data.get("sources", []),
            "research_summary": research_data.get("research_summary", ""),
        }

    def _notes_from_research(self, summaries: list[dict], evidence_pack: list[dict]) -> list[str]:
        notes = []
        for idx, summary in enumerate(summaries[:8], 1):
            source = summary.get("source", {})
            title = source.get("title", f"Source {idx}")
            finding = summary.get("key_findings", "")
            quantitative = summary.get("quantitative_data", [])
            quant_line = f" Quantitative evidence: {', '.join(quantitative[:3])}." if quantitative else ""
            notes.append(f"[{idx}] {title}: {finding}{quant_line}")

        if not notes:
            for idx, evidence in enumerate(evidence_pack[:8], 1):
                title = evidence.get("title", f"Source {idx}")
                excerpt = evidence.get("abstract_excerpt") or evidence.get("abstract") or ""
                notes.append(f"[{idx}] {title}: {excerpt}")
        return notes

    async def _generate_validated_output(self, payload: dict, project_id: str, db) -> dict:
        attempts = 3 if is_llm_available() else 1
        last_validation = None
        content = ""

        for attempt in range(attempts):
            if is_llm_available():
                content = await self._llm_write(payload, attempt, last_validation, project_id, db)
            else:
                content = self._grounded_write(payload)

            content = self.clean_output(content)
            validation = self.validate_output(content, payload)
            last_validation = validation
            if validation["valid"]:
                return {
                    "section": payload["section"],
                    "content": content,
                    "word_count": len(content.split()),
                    "validation": validation,
                }

        fallback = self.clean_output(self._grounded_write(payload))
        fallback_validation = self.validate_output(fallback, payload)
        return {
            "section": payload["section"],
            "content": fallback,
            "word_count": len(fallback.split()),
            "validation": fallback_validation,
        }

    def clean_output(self, text: str) -> str:
        cleaned = "\n".join(line.strip() for line in str(text).splitlines() if line.strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        for phrase in PROMPT_LEAK_PATTERNS:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

        for phrase in GENERIC_PHRASES:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        deduped_sentences = []
        seen = set()
        for sentence in sentences:
            normalized = " ".join(sentence.lower().split())
            if normalized and normalized not in seen:
                deduped_sentences.append(sentence.strip())
                seen.add(normalized)

        cleaned = " ".join(deduped_sentences)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        return cleaned.strip()

    def validate_output(self, content: str, payload: dict) -> dict:
        lowered = content.lower()
        repeated_ratio = self._repeated_phrase_ratio(content)
        domain_hits = [keyword for keyword in payload["domain_keywords"] if keyword in lowered]
        citations = len(re.findall(r"\[\d+\]", content))
        failures = []

        if any(pattern in lowered for pattern in PROMPT_LEAK_PATTERNS):
            failures.append("prompt leakage detected")
        if repeated_ratio > 0.08:
            failures.append("repeated phrases exceed threshold")
        if len(domain_hits) < max(2, min(5, len(payload["domain_keywords"]) // 4 or 2)):
            failures.append("insufficient domain-specific terminology")
        if not payload["research_notes"]:
            failures.append("missing research grounding")
        if citations == 0 and payload["research_notes"]:
            failures.append("missing inline citations")

        return {
            "valid": not failures,
            "failures": failures,
            "repeated_phrase_ratio": round(repeated_ratio, 3),
            "domain_keyword_hits": len(domain_hits),
            "citation_count": citations,
        }

    def _repeated_phrase_ratio(self, content: str, ngram_size: int = 4) -> float:
        words = re.findall(r"\b\w+\b", content.lower())
        if len(words) < ngram_size * 2:
            return 0.0
        grams = [" ".join(words[index:index + ngram_size]) for index in range(len(words) - ngram_size + 1)]
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / len(grams) if grams else 0.0

    def _extract_domain_keywords(self, topic: str, thesis: str, research_notes: list[str]) -> list[str]:
        corpus = " ".join([topic, thesis, *research_notes])
        keywords = []
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", corpus.lower()):
            if len(token) >= 5 and token not in keywords:
                keywords.append(token)
        return keywords[:20]

    async def _llm_write(self, payload: dict, attempt: int, last_validation: dict | None, project_id: str, db) -> str:
        stricter_feedback = ""
        if attempt > 0 and last_validation:
            stricter_feedback = (
                "The previous draft was rejected. Fix these issues explicitly: "
                f"{'; '.join(last_validation.get('failures', []))}. "
                "Increase specificity, reduce repeated wording, and make every paragraph evidence-led."
            )

        prompt = self.build_prompt(payload, stricter_feedback)
        return await timed_chat_completion(
            prompt,
            db=db,
            agent_name=self.name,
            log_api_call_fn=self._log_api_call,
            temperature=0.25 if attempt == 0 else 0.15,
            max_tokens=min(1600, max(900, payload["word_count"] * 3)),
        )

    def build_prompt(self, payload: dict, stricter_feedback: str = "") -> str:
        section_goal = SECTION_GUIDANCE.get(payload["section"], "advance the thesis with technical, evidence-based analysis")
        return (
            f"Write the {payload['section']} section of a final-year university paper on '{payload['topic']}'. "
            f"The section must {section_goal}.\n\n"
            f"Central thesis:\n{payload['thesis']}\n\n"
            f"Structured research notes:\n{json.dumps(payload['research_notes'][:8])}\n\n"
            f"Section queries:\n{json.dumps(payload['section_queries'][:5])}\n\n"
            f"Literature synthesis:\n{payload['research_summary']}\n\n"
            f"Revision feedback:\n{payload['feedback']}\n\n"
            f"Additional constraints:\n{stricter_feedback}\n\n"
            "Requirements:\n"
            f"- Write approximately {payload['word_count']} words in polished academic prose.\n"
            "- Use named studies, systems, datasets, methods, or case examples from the notes.\n"
            "- Include quantitative evidence whenever the notes provide it.\n"
            "- Use inline citations like [1], [2], [3] for factual claims.\n"
            "- Do not mention the prompt, instructions, target length, or that you are an AI.\n"
            "- Do not use generic filler such as 'this paper examines' or 'has gained considerable attention'.\n"
            "- Every paragraph must advance the thesis with concrete evidence or technical reasoning.\n"
            "Return only the final section text."
        )

    def _grounded_write(self, payload: dict) -> str:
        notes = payload["research_notes"][:6]
        section = payload["section"]
        topic = payload["topic"]
        thesis = payload["thesis"] or f"The literature on {topic} supports a specific, evidence-grounded argument."

        paragraphs = [
            f"{thesis} In the {section} context, the strongest evidence shows that {self._note_sentence(notes[0] if notes else topic, preserve_prefix=True)}",
        ]

        if len(notes) > 1:
            paragraphs.append(
                "Comparative evidence reinforces this position: "
                + " ".join(self._note_sentence(note, preserve_prefix=True) for note in notes[1:3])
            )
        if len(notes) > 3:
            paragraphs.append(
                "The technical implications become clearer when quantitative findings are considered: "
                + " ".join(self._note_sentence(note, preserve_prefix=True) for note in notes[3:5])
            )

        paragraphs.append(
            "Taken together, these findings indicate that the section should prioritise mechanism, trade-offs, and evidence quality rather than broad description."
        )
        return "\n\n".join(paragraphs)

    def _note_claim(self, note: str) -> str:
        cleaned = " ".join(str(note).split())
        cleaned = re.sub(r"^\[\d+\]\s*", "", cleaned)
        return cleaned[0].lower() + cleaned[1:] if cleaned else "the current evidence remains limited."

    def _note_sentence(self, note: str, preserve_prefix: bool = False) -> str:
        cleaned = " ".join(str(note).split()) if preserve_prefix else self._note_claim(note)
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned
