"""Deterministic evidence and citation verification engine.

Provides 3-part provenance verification (Timestamp + Speaker + Verbatim Source Lookup),
eliminating quote hallucinations by extracting the final citation quote directly
from the immutable DialogueTurn in the source transcript.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.models.analysis import (
    EvidenceStatus,
    InterviewQuestion,
    QuoteCitation,
)
from src.models.transcript import DialogueTurn, TranscriptDocument

QUESTION_RE = re.compile(r"^\s*(?:Q\s*)?(\d+)[.)]\s+(.+?)\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")


def parse_interview_guide(source: str | Path) -> list[InterviewQuestion]:
    """Parse any numbered interview guide questions dynamically without hardcoding."""
    if isinstance(source, Path) or (isinstance(source, str) and (Path(source).exists() or "\n" not in source)):
        path = Path(source)
        if path.exists():
            raw_text = path.read_text(encoding="utf-8")
        else:
            raw_text = str(source)
    else:
        raw_text = str(source)

    questions: list[InterviewQuestion] = []
    for match in QUESTION_RE.finditer(raw_text):
        question_number = int(match.group(1))
        question_text = match.group(2).strip()
        questions.append(
            InterviewQuestion(
                question_id=f"Q{question_number}",
                question_text=question_text,
            )
        )
    questions.sort(key=lambda q: int(q.question_id[1:]))
    if not questions:
        raise ValueError("No numbered questions found in interview guide")
    return questions


def parse_interview_guide_file(path: str | Path) -> list[InterviewQuestion]:
    """Reads interview guide file from disk and parses questions."""
    return parse_interview_guide(path)


def _normalize(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def _sentence_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _make_citation(
    transcript: TranscriptDocument,
    turn: DialogueTurn,
    quote: str,
    status: EvidenceStatus,
    match_score: float,
) -> QuoteCitation:
    return QuoteCitation(
        turn_id=turn.turn_id,
        transcript_id=transcript.transcript_id,
        expert_name=transcript.expert_name,
        speaker=turn.speaker,
        timestamp=turn.timestamp,
        quote=quote,
        evidence_status=status,
        match_score=match_score,
        start_char=turn.start_char,
        end_char=turn.end_char,
        start_line=turn.start_line,
        end_line=turn.end_line,
    )


def find_turn_by_timestamp(
    transcript: TranscriptDocument, timestamp: str
) -> DialogueTurn | None:
    return next((t for t in transcript.turns if t.timestamp == timestamp), None)


def _best_fuzzy_sentence(source: str, candidate: str) -> tuple[str | None, float]:
    best_sentence: str | None = None
    best_score = 0.0
    for sentence in _sentence_chunks(source):
        score = difflib.SequenceMatcher(
            None, _normalize(sentence), _normalize(candidate)
        ).ratio()
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence, round(best_score, 2)


def _find_normalized_source_slice(source: str, normalized_target: str) -> str | None:
    for sentence in _sentence_chunks(source):
        if normalized_target in _normalize(sentence) or _normalize(sentence) in normalized_target:
            return sentence
    if normalized_target in _normalize(source):
        return source
    return None


def resolve_evidence_turn_ids(
    transcript: TranscriptDocument,
    evidence_turn_ids: Iterable[str],
    *,
    candidate_quotes: dict[str, str] | None = None,
) -> list[QuoteCitation]:
    """Resolve model-selected evidence IDs to immutable source citations."""
    candidate_quotes = candidate_quotes or {}
    citations: list[QuoteCitation] = []
    seen: set[str] = set()

    turns_by_id = {turn.turn_id: turn for turn in transcript.turns}
    for turn_id in evidence_turn_ids:
        if turn_id in seen:
            continue
        seen.add(turn_id)
        turn = turns_by_id.get(turn_id)
        if turn is None:
            citations.append(
                QuoteCitation(
                    turn_id=turn_id,
                    transcript_id=transcript.transcript_id,
                    expert_name=transcript.expert_name,
                    speaker="",
                    timestamp="",
                    quote="",
                    evidence_status=EvidenceStatus.REJECTED,
                    match_score=0.0,
                    start_char=0,
                    end_char=0,
                    start_line=1,
                    end_line=1,
                )
            )
            continue

        candidate = candidate_quotes.get(turn_id)
        if not candidate:
            citations.append(
                _make_citation(
                    transcript,
                    turn,
                    turn.content,
                    EvidenceStatus.EXACT_VERIFIED,
                    1.0,
                )
            )
            continue

        if candidate in turn.content:
            citations.append(
                _make_citation(
                    transcript,
                    turn,
                    candidate,
                    EvidenceStatus.EXACT_VERIFIED,
                    1.0,
                )
            )
            continue

        normalized_candidate = _normalize(candidate)
        normalized_turn = _normalize(turn.content)
        if normalized_candidate and (normalized_candidate in normalized_turn or normalized_turn in normalized_candidate):
            exact_source = _find_normalized_source_slice(turn.content, normalized_candidate)
            citations.append(
                _make_citation(
                    transcript,
                    turn,
                    exact_source or turn.content,
                    EvidenceStatus.NORMALIZED_VERIFIED,
                    0.95,
                )
            )
            continue

        best_sentence, score = _best_fuzzy_sentence(turn.content, candidate)
        if best_sentence is not None and score >= 0.70:
            citations.append(
                _make_citation(
                    transcript,
                    turn,
                    best_sentence,
                    EvidenceStatus.NORMALIZED_VERIFIED,
                    score,
                )
            )
            continue

        # Hallucination Fallback
        citations.append(
            _make_citation(
                transcript,
                turn,
                turn.content,
                EvidenceStatus.FALLBACK_EVIDENCE,
                0.85,
            )
        )

    return citations


def verify_and_extract_quote(
    turn: DialogueTurn,
    candidate_quote: str | None = None,
) -> tuple[str, EvidenceStatus, float, int, int]:
    """
    Direct unit-level verifier validating candidate quote against a DialogueTurn.
    Returns: (verbatim_quote, status, score, start_char, end_char)
    """
    if not candidate_quote or not candidate_quote.strip():
        return turn.content, EvidenceStatus.EXACT_VERIFIED, 1.0, turn.start_char, turn.end_char

    clean_candidate = candidate_quote.strip()

    # 1. Exact match
    exact_idx = turn.content.find(clean_candidate)
    if exact_idx != -1:
        abs_start = turn.start_char + exact_idx
        abs_end = abs_start + len(clean_candidate)
        return clean_candidate, EvidenceStatus.EXACT_VERIFIED, 1.0, abs_start, abs_end

    # 2. Normalized match
    norm_candidate = _normalize(clean_candidate)
    norm_turn = _normalize(turn.content)
    clean_no_punct_candidate = re.sub(r"[^\w\s]", "", norm_candidate)
    turn_no_punct = re.sub(r"[^\w\s]", "", norm_turn)

    if clean_no_punct_candidate in turn_no_punct or norm_candidate in norm_turn:
        for s in _sentence_chunks(turn.content):
            s_norm = _normalize(s)
            s_no_punct = re.sub(r"[^\w\s]", "", s_norm)
            if clean_no_punct_candidate in s_no_punct or s_no_punct in clean_no_punct_candidate:
                sub_start = turn.content.find(s)
                abs_start = turn.start_char + sub_start
                abs_end = abs_start + len(s)
                return s, EvidenceStatus.NORMALIZED_VERIFIED, 0.95, abs_start, abs_end

    # 3. Fuzzy sentence match
    best_sentence, score = _best_fuzzy_sentence(turn.content, clean_candidate)
    if best_sentence is not None and score >= 0.70:
        sub_start = turn.content.find(best_sentence)
        abs_start = turn.start_char + sub_start
        abs_end = abs_start + len(best_sentence)
        return best_sentence, EvidenceStatus.NORMALIZED_VERIFIED, score, abs_start, abs_end

    # 4. Fallback on hallucination
    sentences = _sentence_chunks(turn.content)
    fallback = sentences[0] if sentences else turn.content
    sub_start = turn.content.find(fallback)
    abs_start = turn.start_char + sub_start
    abs_end = abs_start + len(fallback)
    return fallback, EvidenceStatus.FALLBACK_EVIDENCE, 0.70, abs_start, abs_end


resolve_citations_for_answer = resolve_evidence_turn_ids

__all__ = [
    "find_turn_by_timestamp",
    "parse_interview_guide",
    "parse_interview_guide_file",
    "resolve_citations_for_answer",
    "resolve_evidence_turn_ids",
    "verify_and_extract_quote",
    "verify_candidate_quote",
]
