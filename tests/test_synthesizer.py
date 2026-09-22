from pathlib import Path

from src.core.extractor import MockExtractionProvider, build_ground_truth_matrix
from src.core.parser import parse_transcript_file
from src.core.synthesizer import (
    MockSynthesisProvider,
    build_or_load_synthesis_report,
    build_synthesis_report,
    load_synthesis_report,
    synthesis_cache_is_valid,
)
from src.models.synthesis import ConsensusTheme, DisagreementPoint, MarketGrowthSpectrum

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def transcripts():
    specs = [
        ("Transcript_1_France.txt", "france_001"),
        ("Transcript_2_Germany.txt", "germany_001"),
        ("Transcript_3_UK.txt", "uk_001"),
    ]
    return [parse_transcript_file(DATA / filename, transcript_id) for filename, transcript_id in specs]


def matrix():
    guide = (DATA / "Interview_Guide.txt").read_text(encoding="utf-8")
    return build_ground_truth_matrix(transcripts(), guide, MockExtractionProvider())


def test_synthesis_report_has_expected_sections():
    report = build_synthesis_report(matrix(), MockSynthesisProvider())

    assert report.consensus_themes
    assert report.disagreements
    assert report.growth_spectrum
    assert any("training" in theme.title.casefold() for theme in report.consensus_themes)

    assert isinstance(report.consensus_themes[0], ConsensusTheme)
    assert isinstance(report.disagreements[0], DisagreementPoint)
    assert isinstance(report.growth_spectrum, MarketGrowthSpectrum)


def test_consensus_uses_all_three_markets():
    report = build_synthesis_report(matrix(), MockSynthesisProvider())
    training = next(theme for theme in report.consensus_themes if "training" in theme.title.casefold())
    assert set(training.evidence_by_country) == {"France", "Germany", "United Kingdom"}


def test_disagreement_contains_stakeholder_positions():
    report = build_synthesis_report(matrix(), MockSynthesisProvider())
    finance = next(item for item in report.disagreements if "Financial primacy" in item.topic)
    assert set(finance.stances_by_stakeholder) == {"France", "Germany", "United Kingdom"}
    assert all(finance.stances_by_stakeholder.values())


def test_growth_spectrum_has_three_market_placements():
    report = build_synthesis_report(matrix(), MockSynthesisProvider())
    spectrum = report.growth_spectrum
    combined = spectrum.conservative + spectrum.moderate + spectrum.bullish
    assert len(combined) == 3
    assert any("Germany" in item for item in spectrum.conservative)
    assert any("France" in item for item in spectrum.moderate)
    assert any("United Kingdom" in item for item in spectrum.bullish)


def test_synthesis_cache_roundtrip(tmp_path):
    report_path = tmp_path / "synthesis_report.json"
    base = matrix()
    provider = MockSynthesisProvider()

    first = build_or_load_synthesis_report(base, provider, report_path)
    second = build_or_load_synthesis_report(base, provider, report_path)

    assert second.model_dump() == first.model_dump()
    assert synthesis_cache_is_valid(second, base)
    assert load_synthesis_report(report_path).model_dump() == first.model_dump()


def test_synthesis_cache_invalidates_when_matrix_changes(tmp_path):
    report_path = tmp_path / "synthesis_report.json"
    base = matrix()
    provider = MockSynthesisProvider()
    report = build_or_load_synthesis_report(base, provider, report_path)

    changed = base.model_copy(update={"matrix_version": "ground_truth_v2"})
    assert not synthesis_cache_is_valid(report, changed)
