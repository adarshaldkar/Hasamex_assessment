from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.core.extractor import (
    GeminiExtractionProvider,
    MockExtractionProvider,
    OpenAIExtractionProvider,
    build_or_load_ground_truth_matrix,
)
from src.core.parser import parse_transcript_file

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = get_settings()
DATA = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")
CACHE = ROOT / "storage" / "processed" / "ground_truth_matrix.json"


def main() -> None:
    if SETTINGS.llm_provider == "openai" or (SETTINGS.openai_api_key and not SETTINGS.llm_provider == "gemini"):
        provider = OpenAIExtractionProvider(
            api_key=SETTINGS.openai_api_key,
            model_name=SETTINGS.openai_model,
            base_url=SETTINGS.openai_base_url,
        )
    elif SETTINGS.llm_provider == "gemini" and (SETTINGS.gemini_api_key or SETTINGS.google_api_key):
        provider = GeminiExtractionProvider(
            api_key=SETTINGS.gemini_api_key or SETTINGS.google_api_key,
            model_name=SETTINGS.gemini_model,
        )
    else:
        provider = MockExtractionProvider()

    transcript_specs = [
        ("Transcript_1_France.txt", "france_001"),
        ("Transcript_2_Germany.txt", "germany_001"),
        ("Transcript_3_UK.txt", "uk_001"),
    ]
    transcripts = [
        parse_transcript_file(DATA / filename, transcript_id)
        for filename, transcript_id in transcript_specs
    ]
    guide = (DATA / "Interview_Guide.txt").read_text(encoding="utf-8")
    matrix = build_or_load_ground_truth_matrix(
        transcripts,
        guide,
        provider,
        CACHE,
    )
    print(
        f"Ground-truth matrix ready: {len(matrix.analyses)} transcripts x "
        f"{len(matrix.questions)} questions = "
        f"{sum(len(a.answers) for a in matrix.analyses)} cells"
    )
    print(f"Provider: {provider.__class__.__name__} ({getattr(provider, 'model_name', 'mock')})")
    print(f"Cache: {CACHE}")


if __name__ == "__main__":
    main()
