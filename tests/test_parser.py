from pathlib import Path

from src.core.parser import parse_transcript_file
from src.models.transcript import SpeakerType

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EXPECTED = {
    "Transcript_1_France.txt": {
        "transcript_id": "france_001",
        "expert_name": "Dr. Jean Martin",
        "role": "Head of Urology",
        "market": "France",
        "total_turns": 14,
        "first_timestamp": "00:00",
        "last_timestamp": "06:08",
        "expert_turns": 7,
        "interviewer_turns": 7,
        "check_timestamp": "02:18",
        "check_text": "Very important. The clinical argument may get surgeons interested",
    },
    "Transcript_2_Germany.txt": {
        "transcript_id": "germany_001",
        "expert_name": "Anna Keller",
        "role": "Former Hospital Procurement Director",
        "market": "Germany",
        "total_turns": 14,
        "first_timestamp": "00:00",
        "last_timestamp": "06:05",
        "expert_turns": 7,
        "interviewer_turns": 7,
        "check_timestamp": "02:08",
        "check_text": "We look at total cost of ownership",
    },
    "Transcript_3_UK.txt": {
        "transcript_id": "uk_001",
        "expert_name": "Dr. Emily Carter",
        "role": "Consultant Urologist",
        "market": "United Kingdom",
        "total_turns": 14,
        "first_timestamp": "00:00",
        "last_timestamp": "06:04",
        "expert_turns": 7,
        "interviewer_turns": 7,
        "check_timestamp": "01:05",
        "check_text": "Funding is important, but I would say training capacity",
    },
}


def _parse(filename: str):
    return parse_transcript_file(
        DATA_DIR / filename,
        EXPECTED[filename]["transcript_id"],
    )


def test_france_transcript_parsing():
    doc = _parse("Transcript_1_France.txt")
    expected = EXPECTED["Transcript_1_France.txt"]

    assert doc.expert_name == expected["expert_name"]
    assert doc.role == expected["role"]
    assert doc.market == expected["market"]
    assert doc.total_turns == expected["total_turns"]
    assert doc.turns[0].timestamp == expected["first_timestamp"]
    assert doc.turns[-1].timestamp == expected["last_timestamp"]
    assert doc.turns[0].speaker_type == SpeakerType.INTERVIEWER

    target = next(t for t in doc.turns if t.timestamp == expected["check_timestamp"])
    assert expected["check_text"] in target.content
    assert target.speaker_type == SpeakerType.EXPERT


def test_germany_transcript_parsing():
    doc = _parse("Transcript_2_Germany.txt")
    expected = EXPECTED["Transcript_2_Germany.txt"]

    assert doc.expert_name == expected["expert_name"]
    assert doc.role == expected["role"]
    assert doc.market == expected["market"]
    assert doc.total_turns == expected["total_turns"]
    assert doc.turns[0].timestamp == expected["first_timestamp"]
    assert doc.turns[-1].timestamp == expected["last_timestamp"]

    target = next(t for t in doc.turns if t.timestamp == expected["check_timestamp"])
    assert expected["check_text"] in target.content
    assert target.speaker == "Anna Keller"
    assert target.speaker_type == SpeakerType.EXPERT


def test_uk_transcript_parsing():
    doc = _parse("Transcript_3_UK.txt")
    expected = EXPECTED["Transcript_3_UK.txt"]

    assert doc.expert_name == expected["expert_name"]
    assert doc.role == expected["role"]
    assert doc.market == expected["market"]
    assert doc.total_turns == expected["total_turns"]
    assert doc.turns[0].timestamp == expected["first_timestamp"]
    assert doc.turns[-1].timestamp == expected["last_timestamp"]

    target = next(t for t in doc.turns if t.timestamp == expected["check_timestamp"])
    assert expected["check_text"] in target.content
    assert target.speaker == "Dr. Carter"
    assert target.speaker_type == SpeakerType.EXPERT


def test_all_transcripts_have_seven_alternating_pairs():
    for filename, expected in EXPECTED.items():
        doc = _parse(filename)
        assert doc.total_turns == 14
        assert sum(t.speaker_type == SpeakerType.INTERVIEWER for t in doc.turns) == 7
        assert sum(t.speaker_type == SpeakerType.EXPERT for t in doc.turns) == 7

        for index, turn in enumerate(doc.turns):
            expected_type = (
                SpeakerType.INTERVIEWER
                if index % 2 == 0
                else SpeakerType.EXPERT
            )
            assert turn.speaker_type == expected_type


def test_provenance_character_offset_invariant():
    for filename in EXPECTED:
        doc = _parse(filename)

        for turn in doc.turns:
            assert doc.raw_text[turn.start_char : turn.end_char] == turn.content
            assert 0 <= turn.start_char < turn.end_char <= len(doc.raw_text)
            assert turn.start_line <= turn.end_line
            assert turn.start_line >= 1
            assert turn.end_line >= 1


def test_speaker_type_classification():
    for filename in EXPECTED:
        doc = _parse(filename)
        for turn in doc.turns:
            if turn.speaker.casefold() == "interviewer":
                assert turn.speaker_type == SpeakerType.INTERVIEWER
            else:
                assert turn.speaker_type == SpeakerType.EXPERT


def test_character_offsets_target_speech_payload_not_speaker_prefix():
    doc = _parse("Transcript_1_France.txt")
    turn = next(t for t in doc.turns if t.timestamp == "02:18")

    source_slice = doc.raw_text[turn.start_char : turn.end_char]
    assert source_slice.startswith("Very important.")
    assert not source_slice.startswith("Dr. Martin:")
    assert "Dr. Martin:" not in source_slice


def test_sha256_is_recorded_and_stable():
    doc1 = _parse("Transcript_1_France.txt")
    doc2 = _parse("Transcript_1_France.txt")

    assert len(doc1.file_hash) == 64
    assert doc1.file_hash == doc2.file_hash
    assert doc1.file_hash.isalnum()
