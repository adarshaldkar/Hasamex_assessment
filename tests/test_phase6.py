from __future__ import annotations

from pathlib import Path

from src.core.parser import parse_transcript_file

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")


def test_phase6_sources_are_available():
    app = ROOT / "app" / "streamlit_app.py"
    phase_doc = ROOT / "docs" / "PHASE6.md"
    assert app.exists()
    assert phase_doc.exists()


def test_phase6_loads_all_reference_transcripts():
    specs = [
        ("Transcript_1_France.txt", "france_001"),
        ("Transcript_2_Germany.txt", "germany_001"),
        ("Transcript_3_UK.txt", "uk_001"),
    ]
    transcripts = [parse_transcript_file(RAW / name, transcript_id) for name, transcript_id in specs]
    assert [t.market for t in transcripts] == ["France", "Germany", "United Kingdom"]
    assert all(t.total_turns == 14 for t in transcripts)
