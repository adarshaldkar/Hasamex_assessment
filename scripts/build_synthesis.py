from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.core.extractor import MockExtractionProvider, load_ground_truth_matrix
from src.core.synthesizer import (
    GeminiSynthesisProvider,
    MockSynthesisProvider,
    OpenAISynthesisProvider,
    build_or_load_synthesis_report,
)

SETTINGS = get_settings()
MATRIX_PATH = ROOT / "storage" / "processed" / "ground_truth_matrix.json"
REPORT_PATH = ROOT / "storage" / "processed" / "synthesis_report.json"


def main() -> None:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth matrix not found at {MATRIX_PATH}. Run Phase 3 first."
        )

    matrix = load_ground_truth_matrix(MATRIX_PATH)

    if SETTINGS.llm_provider == "openai" or (SETTINGS.openai_api_key and not SETTINGS.llm_provider == "gemini"):
        provider = OpenAISynthesisProvider(
            api_key=SETTINGS.openai_api_key,
            model_name=SETTINGS.openai_model,
            base_url=SETTINGS.openai_base_url,
        )
    elif SETTINGS.llm_provider == "gemini" and (SETTINGS.gemini_api_key or SETTINGS.google_api_key):
        provider = GeminiSynthesisProvider(
            api_key=SETTINGS.gemini_api_key or SETTINGS.google_api_key,
            model_name=SETTINGS.gemini_model,
        )
    else:
        provider = MockSynthesisProvider()

    report = build_or_load_synthesis_report(matrix, provider, REPORT_PATH)
    print(f"Synthesis report ready: {REPORT_PATH}")
    print(f"Provider: {provider.__class__.__name__} ({getattr(provider, 'model_name', 'mock')})")
    print(f"Consensus themes: {len(report.consensus_themes)}")
    print(f"Disagreements: {len(report.disagreements)}")
    print(
        "Growth spectrum: "
        f"{len(report.growth_spectrum.conservative)} conservative, "
        f"{len(report.growth_spectrum.moderate)} moderate, "
        f"{len(report.growth_spectrum.bullish)} bullish"
    )


if __name__ == "__main__":
    main()
