from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models.analysis import GroundTruthMatrix
from src.models.synthesis import (
    ConsensusTheme,
    DisagreementPoint,
    MarketGrowthSpectrum,
    SynthesisReport,
)
from src.prompts.synthesis import PROMPT_VERSION, SYSTEM_PROMPT, build_synthesis_prompt


class LLMSynthesis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    consensus_themes: list[ConsensusTheme] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    growth_spectrum: MarketGrowthSpectrum


class SynthesisProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def synthesize(self, matrix: GroundTruthMatrix) -> LLMSynthesis:
        raise NotImplementedError


class MockSynthesisProvider(SynthesisProvider):
    model_name = "mock-synthesis-v1"

    def synthesize(self, matrix: GroundTruthMatrix) -> LLMSynthesis:
        markets = {a.market: a for a in matrix.analyses}
        return LLMSynthesis(
            consensus_themes=[
                ConsensusTheme(
                    title="Training and utilization are critical to adoption",
                    description=(
                        "All three experts describe sufficient training and procedure volume as "
                        "important to making robotic surgery programmes sustainable."
                    ),
                    evidence_by_country={
                        market: [_answer_text(analysis, "Q4")]
                        for market, analysis in markets.items()
                        if _answer_text(analysis, "Q4")
                    },
                ),
                ConsensusTheme(
                    title="Adoption is stronger in larger centres",
                    description=(
                        "All three markets show stronger adoption in larger or better-resourced "
                        "centres, with smaller hospitals lagging."
                    ),
                    evidence_by_country={
                        market: [_answer_text(analysis, "Q1")]
                        for market, analysis in markets.items()
                        if _answer_text(analysis, "Q1")
                    },
                ),
                ConsensusTheme(
                    title="Capital and budget cycles affect purchasing",
                    description=(
                        "Each expert identifies funding, capital approval, or budget-cycle timing "
                        "as an important practical factor in purchasing."
                    ),
                    evidence_by_country={
                        market: [_answer_text(analysis, "Q2")]
                        for market, analysis in markets.items()
                        if _answer_text(analysis, "Q2")
                    },
                ),
            ],
            disagreements=[
                DisagreementPoint(
                    topic="Financial primacy vs clinical strategy",
                    description=(
                        "France and Germany emphasize economic justification strongly, while the UK "
                        "expert describes economics and clinical strategy as balanced."
                    ),
                    stances_by_stakeholder={
                        "France": _answer_text(markets["France"], "Q3"),
                        "Germany": _answer_text(markets["Germany"], "Q3"),
                        "United Kingdom": _answer_text(markets["United Kingdom"], "Q3"),
                    },
                ),
                DisagreementPoint(
                    topic="Three-to-five-year growth outlook",
                    description=(
                        "The experts differ in the expected pace of growth, ranging from gradual "
                        "growth to stronger acceleration if constraints ease."
                    ),
                    stances_by_stakeholder={
                        "France": _answer_text(markets["France"], "Q5"),
                        "Germany": _answer_text(markets["Germany"], "Q5"),
                        "United Kingdom": _answer_text(markets["United Kingdom"], "Q5"),
                    },
                ),
                DisagreementPoint(
                    topic="Purchase decision timelines",
                    description=(
                        "The experts describe different purchasing windows, with Germany indicating "
                        "a longer process and the UK describing a shorter route when funding is already available."
                    ),
                    stances_by_stakeholder={
                        "France": _answer_text(markets["France"], "Q6"),
                        "Germany": _answer_text(markets["Germany"], "Q6"),
                        "United Kingdom": _answer_text(markets["United Kingdom"], "Q6"),
                    },
                ),
            ],
            growth_spectrum=MarketGrowthSpectrum(
                conservative=["Germany — Anna Keller: gradual growth; high single digits to low double digits"],
                moderate=["France — Dr. Jean Martin: steady 15–20% annual procedure growth in stronger centres"],
                bullish=["United Kingdom — Dr. Emily Carter: above 15% annual growth in some areas if constraints ease"],
            ),
        )


class OpenAISynthesisProvider(SynthesisProvider):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import os
        from dotenv import load_dotenv

        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        if not self.base_url and self.api_key.startswith("sk-or-v1-"):
            self.base_url = "https://openrouter.ai/api/v1"

        default_model = "openai/gpt-4o-mini" if "openrouter.ai" in self.base_url else "gpt-4o-mini"
        self.model_name = model_name or os.getenv("OPENAI_MODEL", default_model)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAISynthesisProvider")

    def synthesize(self, matrix: GroundTruthMatrix) -> LLMSynthesis:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the official openai package to use OpenAISynthesisProvider") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_synthesis_prompt(matrix)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            return LLMSynthesis.model_validate(json.loads(content))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON") from exc


class GeminiSynthesisProvider(SynthesisProvider):
    def __init__(self, *, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiSynthesisProvider")

    def synthesize(self, matrix: GroundTruthMatrix) -> LLMSynthesis:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the google-genai package to use GeminiSynthesisProvider") from exc

        client = genai.Client(api_key=self.api_key)
        prompt = f"{SYSTEM_PROMPT}\n\n{build_synthesis_prompt(matrix)}"
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        content = response.text or "{}"
        try:
            return LLMSynthesis.model_validate(json.loads(content))
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid JSON") from exc


def _answer_text(analysis: Any, question_id: str) -> str:
    for answer in analysis.answers:
        if answer.question_id == question_id:
            return answer.answer.strip()
    return ""


def _matrix_fingerprint(matrix: GroundTruthMatrix) -> str:
    import hashlib

    payload = matrix.model_dump(mode="json", exclude={"generated_at"})
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_synthesis_report(
    matrix: GroundTruthMatrix,
    provider: SynthesisProvider,
    *,
    report_version: str = "synthesis_v1",
) -> SynthesisReport:
    result = provider.synthesize(matrix)
    return SynthesisReport(
        report_version=report_version,
        source_matrix_version=f"{matrix.matrix_version}:{_matrix_fingerprint(matrix)}",
        consensus_themes=result.consensus_themes,
        disagreements=result.disagreements,
        growth_spectrum=result.growth_spectrum,
    )


def save_synthesis_report(report: SynthesisReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_synthesis_report(path: str | Path) -> SynthesisReport:
    return SynthesisReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def synthesis_cache_is_valid(
    report: SynthesisReport,
    matrix: GroundTruthMatrix,
    *,
    report_version: str = "synthesis_v1",
) -> bool:
    expected = f"{matrix.matrix_version}:{_matrix_fingerprint(matrix)}"
    return report.report_version == report_version and report.source_matrix_version == expected


def build_or_load_synthesis_report(
    matrix: GroundTruthMatrix,
    provider: SynthesisProvider,
    cache_path: str | Path,
    *,
    report_version: str = "synthesis_v1",
    force: bool = False,
) -> SynthesisReport:
    cache = Path(cache_path)
    if cache.exists() and not force:
        cached = load_synthesis_report(cache)
        if synthesis_cache_is_valid(cached, matrix, report_version=report_version):
            return cached

    report = build_synthesis_report(matrix, provider, report_version=report_version)
    save_synthesis_report(report, cache)
    return report


__all__ = [
    "LLMSynthesis",
    "SynthesisProvider",
    "MockSynthesisProvider",
    "OpenAISynthesisProvider",
    "build_synthesis_report",
    "build_or_load_synthesis_report",
    "load_synthesis_report",
    "save_synthesis_report",
    "synthesis_cache_is_valid",
]
