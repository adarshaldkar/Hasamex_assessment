"""Versioned extraction prompt templates with strict negative constraints."""

from __future__ import annotations

from src.models.analysis import InterviewQuestion
from src.models.transcript import DialogueTurn

PROMPT_VERSION = "extraction_v1"

SYSTEM_PROMPT = """You are a grounded qualitative research extraction engine.

Use ONLY the transcript evidence supplied in the user message.
Do not use outside knowledge, external market data, or assumptions.
Return evidence_turn_ids that directly support the answer.
Never invent a transcript turn ID.
If the evidence is insufficient, return an empty evidence_turn_ids list and an empty answer.
Keep the answer concise (2-3 sentences when evidence supports it).
"""


def build_extraction_prompt(
    question: InterviewQuestion,
    candidate_turns: list[DialogueTurn],
) -> str:
    """Builds user prompt for single question extraction across candidate turns."""
    evidence = "\n\n".join(
        f"TURN_ID={turn.turn_id}\nTIMESTAMP={turn.timestamp}\nSPEAKER={turn.speaker}\nCONTENT={turn.content}"
        for turn in candidate_turns
    )
    return (
        f"Question {question.question_id}: {question.question_text}\n\n"
        "Candidate transcript evidence:\n"
        f"{evidence}\n\n"
        "Return JSON with exactly these fields:\n"
        '{"answer": "...", "evidence_turn_ids": ["..."]}'
    )
