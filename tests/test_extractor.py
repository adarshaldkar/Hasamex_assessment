import json
from pathlib import Path

from src.core.extractor import (
    MockExtractionProvider,
    build_ground_truth_matrix,
    build_or_load_ground_truth_matrix,
    load_ground_truth_matrix,
    select_candidate_turns,
)
from src.core.parser import parse_transcript_file
from src.core.quote_verifier import parse_interview_guide
from src.models.analysis import EvidenceStatus

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def transcripts():
    specs = [
        ("Transcript_1_France.txt", "france_001"),
        ("Transcript_2_Germany.txt", "germany_001"),
        ("Transcript_3_UK.txt", "uk_001"),
    ]
    return [parse_transcript_file(DATA / filename, transcript_id) for filename, transcript_id in specs]


def guide_text():
    return (DATA / "Interview_Guide.txt").read_text(encoding="utf-8")


def test_candidate_selection_returns_expert_turns_only():
    doc = transcripts()[0]
    questions = parse_interview_guide(guide_text())
    candidates = select_candidate_turns(questions[0], doc)
    assert candidates
    assert all(t.speaker_type.value == "expert" for t in candidates)


def test_ground_truth_matrix_contains_all_18_cells():
    matrix = build_ground_truth_matrix(transcripts(), guide_text(), MockExtractionProvider())
    assert len(matrix.questions) == 6
    assert len(matrix.analyses) == 3
    assert all(len(analysis.answers) == 6 for analysis in matrix.analyses)
    assert sum(len(analysis.answers) for analysis in matrix.analyses) == 18


def test_all_generated_citations_point_to_existing_turns():
    matrix = build_ground_truth_matrix(transcripts(), guide_text(), MockExtractionProvider())
    valid_turn_ids = {
        turn.turn_id
        for transcript in transcripts()
        for turn in transcript.turns
    }
    for analysis in matrix.analyses:
        for answer in analysis.answers:
            assert answer.evidence_turn_ids
            for citation in answer.citations:
                assert citation.turn_id in valid_turn_ids
                assert citation.evidence_status != EvidenceStatus.REJECTED
                assert citation.quote


def test_cache_versioning_and_source_hashes(tmp_path):
    cache = tmp_path / "ground_truth_matrix.json"
    provider = MockExtractionProvider()
    docs = transcripts()
    matrix1 = build_or_load_ground_truth_matrix(docs, guide_text(), provider, cache)
    matrix2 = build_or_load_ground_truth_matrix(docs, guide_text(), provider, cache)
    assert matrix1.source_hashes == matrix2.source_hashes
    assert matrix2.model_name == provider.model_name
    assert matrix2.prompt_version == matrix1.prompt_version
    assert cache.exists()

    loaded = load_ground_truth_matrix(cache)
    assert loaded.model_dump() == matrix2.model_dump()


def test_cache_invalidates_when_source_hash_changes(tmp_path):
    cache = tmp_path / "ground_truth_matrix.json"
    provider = MockExtractionProvider()
    docs = transcripts()
    matrix1 = build_or_load_ground_truth_matrix(docs, guide_text(), provider, cache)
    altered = docs[0].model_copy(update={"file_hash": "changed-hash"})
    matrix2 = build_or_load_ground_truth_matrix([altered, *docs[1:]], guide_text(), provider, cache)
    assert matrix1.source_hashes != matrix2.source_hashes
