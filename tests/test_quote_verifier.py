"""Automated unit tests for deterministic evidence & quote verifier."""

from pathlib import Path
import pytest

from src.core.parser import parse_transcript_file
from src.core.quote_verifier import (
    parse_interview_guide,
    resolve_citations_for_answer,
    verify_and_extract_quote,
)
from src.models.analysis import EvidenceStatus

DATA_DIR = Path("data")


def test_dynamic_interview_guide_parsing():
    """Validates that interview guide is parsed dynamically without hardcoding."""
    guide_path = DATA_DIR / "Interview_Guide.txt"
    questions = parse_interview_guide(guide_path)

    assert len(questions) == 6
    assert questions[0].question_id == "Q1"
    assert "current adoption" in questions[0].text.lower()
    assert questions[1].question_id == "Q2"
    assert "barriers" in questions[1].text.lower()
    assert questions[5].question_id == "Q6"
    assert "timeline" in questions[5].text.lower()


def test_exact_quote_verification():
    """Asserts that an exact verbatim substring matches with status EXACT_VERIFIED and score 1.0."""
    france_doc = parse_transcript_file(DATA_DIR / "Transcript_1_France.txt")
    turn_04 = next(t for t in france_doc.turns if t.timestamp == "01:20")

    exact_quote = "The biggest issue is still capital budget approval."
    extracted_quote, status, score, start_char, end_char = verify_and_extract_quote(
        turn_04, exact_quote
    )

    assert status == EvidenceStatus.EXACT_VERIFIED
    assert score == 1.0
    assert extracted_quote == exact_quote
    assert france_doc.raw_text[start_char:end_char] == exact_quote


def test_normalized_quote_verification_smart_quotes_and_punctuation():
    """Asserts that minor punctuation or smart-quote mutations are normalized and return source text."""
    germany_doc = parse_transcript_file(DATA_DIR / "Transcript_2_Germany.txt")
    turn_06 = next(t for t in germany_doc.turns if t.timestamp == "02:08")

    # Candidate with altered punctuation and smart quotes
    candidate = 'We look at total cost of ownership expected procedure volume maintenance'
    extracted_quote, status, score, start_char, end_char = verify_and_extract_quote(
        turn_06, candidate
    )

    assert status == EvidenceStatus.NORMALIZED_VERIFIED
    assert score >= 0.85
    # The returned quote must be the real sentence from the transcript, NOT the candidate
    assert "total cost of ownership" in extracted_quote.lower()
    assert germany_doc.raw_text[start_char:end_char] == extracted_quote


def test_multi_citation_resolution():
    """Asserts that answers referencing multiple timestamps resolve into distinct citations."""
    uk_doc = parse_transcript_file(DATA_DIR / "Transcript_3_UK.txt")
    turn_06 = next(t for t in uk_doc.turns if t.timestamp == "02:07")
    turn_08 = next(t for t in uk_doc.turns if t.timestamp == "03:10")

    citations = resolve_citations_for_answer(
        transcript=uk_doc,
        evidence_turn_ids=[turn_06.turn_id, turn_08.turn_id]
    )

    assert len(citations) == 2
    assert citations[0].timestamp == "02:07"
    assert citations[0].speaker == "Dr. Carter"
    assert citations[0].evidence_status == EvidenceStatus.EXACT_VERIFIED
    assert citations[1].timestamp == "03:10"
    assert citations[1].evidence_status == EvidenceStatus.EXACT_VERIFIED
    assert "balanced" in citations[1].quote.lower()


def test_hallucinated_quote_rejection_and_fallback():
    """
    Asserts that if the LLM hallucinates a quote, the fake text is REJECTED
    and replaced with the authentic source turn text with FALLBACK_EVIDENCE status.
    """
    france_doc = parse_transcript_file(DATA_DIR / "Transcript_1_France.txt")
    turn_06 = next(t for t in france_doc.turns if t.timestamp == "02:18")

    fake_quote = "Robotic systems are totally free and every French hospital already bought five."
    extracted_quote, status, score, start_char, end_char = verify_and_extract_quote(
        turn_06, fake_quote
    )

    assert status == EvidenceStatus.FALLBACK_EVIDENCE
    assert score < 0.80
    # Crucial: the returned quote is NOT the fake text!
    assert fake_quote not in extracted_quote
    assert "very important" in extracted_quote.lower()
    assert france_doc.raw_text[start_char:end_char] == extracted_quote


def test_rejected_turn_id():
    """Asserts that referencing a non-existent turn ID returns REJECTED status with score 0.0."""
    france_doc = parse_transcript_file(DATA_DIR / "Transcript_1_France.txt")

    citations = resolve_citations_for_answer(
        transcript=france_doc,
        evidence_turn_ids=["france_001_turn_999"]
    )

    assert len(citations) == 1
    assert citations[0].evidence_status == EvidenceStatus.REJECTED
    assert citations[0].match_score == 0.0
    assert citations[0].quote == ""


def test_pure_evidence_id_architecture_without_candidate_quote():
    """
    Validates our core architecture: LLM only passes evidence_turn_ids,
    and Python deterministically extracts the exact source quote.
    """
    germany_doc = parse_transcript_file(DATA_DIR / "Transcript_2_Germany.txt")
    tco_turn = next(t for t in germany_doc.turns if t.timestamp == "02:08")

    citations = resolve_citations_for_answer(
        transcript=germany_doc,
        evidence_turn_ids=[tco_turn.turn_id],
        candidate_quotes=None  # No quote provided by LLM
    )

    assert len(citations) == 1
    citation = citations[0]
    assert citation.evidence_status == EvidenceStatus.EXACT_VERIFIED
    assert citation.match_score == 1.0
    assert "total cost of ownership" in citation.quote.lower()
    assert germany_doc.raw_text[citation.start_char:citation.end_char] == citation.quote
