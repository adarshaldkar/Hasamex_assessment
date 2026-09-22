"""Structured extraction engine for the standardized interview guide questions."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.quote_verifier import parse_interview_guide, resolve_evidence_turn_ids
from src.models.analysis import (
    GroundTruthMatrix,
    InterviewQuestion,
    QuestionAnswer,
    TranscriptAnalysis,
)
from src.models.transcript import DialogueTurn, TranscriptDocument
from src.prompts.extraction import PROMPT_VERSION, SYSTEM_PROMPT, build_extraction_prompt

WORD_RE = re.compile(r"[A-Za-z0-9]+")


class LLMExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    evidence_turn_ids: list[str] = Field(default_factory=list)


class ExtractionProvider(ABC):
    """Provider contract for structured question extraction."""

    model_name: str = "unknown"

    @abstractmethod
    def extract(
        self,
        *,
        question: InterviewQuestion,
        candidate_turns: list[DialogueTurn],
    ) -> LLMExtraction:
        raise NotImplementedError


class MockExtractionProvider(ExtractionProvider):
    """Deterministic provider used by tests and offline evaluation."""

    model_name = "mock-extractor-v1"

    def extract(
        self,
        *,
        question: InterviewQuestion,
        candidate_turns: list[DialogueTurn],
    ) -> LLMExtraction:
        if not candidate_turns:
            return LLMExtraction(answer="", evidence_turn_ids=[])
        ranked = sorted(
            candidate_turns,
            key=lambda turn: _overlap_score(question.question_text, turn.content),
            reverse=True,
        )
        best = ranked[0]
        score = _overlap_score(question.question_text, best.content)
        if score <= 0:
            return LLMExtraction(answer="", evidence_turn_ids=[])
        answer = _deterministic_answer(best.content)
        return LLMExtraction(answer=answer, evidence_turn_ids=[best.turn_id])


class OpenAIExtractionProvider(ExtractionProvider):
    """Direct official OpenAI SDK adapter with strict JSON + Pydantic validation."""

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
            raise ValueError("OPENAI_API_KEY is required for OpenAIExtractionProvider")

    def extract(
        self,
        *,
        question: InterviewQuestion,
        candidate_turns: list[DialogueTurn],
    ) -> LLMExtraction:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_extraction_prompt(question, candidate_turns),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON") from exc
        return LLMExtraction.model_validate(parsed)


class GeminiExtractionProvider(ExtractionProvider):
    """Google Gemini SDK adapter with structured JSON mode + Pydantic validation."""

    def __init__(self, *, model_name: str | None = None, api_key: str | None = None):
        import os

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiExtractionProvider")

    def extract(
        self,
        *,
        question: InterviewQuestion,
        candidate_turns: list[DialogueTurn],
    ) -> LLMExtraction:
        import time
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = f"{SYSTEM_PROMPT}\n\n{build_extraction_prompt(question, candidate_turns)}"
        
        last_exc = None
        for attempt in range(5):
            try:
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
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError("Gemini returned invalid JSON") from exc
                return LLMExtraction.model_validate(parsed)
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    time.sleep(7.0 * (attempt + 1))
                else:
                    raise
        raise RuntimeError(f"Gemini extraction failed after 5 attempts: {last_exc}")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in WORD_RE.findall(text)}


def _overlap_score(question: str, content: str) -> float:
    q = _tokens(question)
    c = _tokens(content)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def _deterministic_answer(content: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    useful = [sentence.strip() for sentence in sentences if sentence.strip()]
    return " ".join(useful[:2]) if useful else content.strip()


def select_candidate_turns(
    question: InterviewQuestion,
    transcript: TranscriptDocument,
    *,
    top_n: int = 4,
) -> list[DialogueTurn]:
    """Select candidate expert evidence without hardcoding country or turn IDs."""
    expert_turns = [turn for turn in transcript.turns if turn.speaker_type.value == "expert"]
    if not expert_turns:
        return []

    ranked = sorted(
        expert_turns,
        key=lambda turn: (_overlap_score(question.question_text, turn.content), -turn.seconds),
        reverse=True,
    )
    positive = [turn for turn in ranked if _overlap_score(question.question_text, turn.content) > 0]

    try:
        q_index = int(question.question_id.removeprefix("Q")) - 1
    except ValueError:
        q_index = -1
    fallback = expert_turns[q_index] if 0 <= q_index < len(expert_turns) else None

    selected: list[DialogueTurn] = []
    for turn in positive:
        if turn not in selected:
            selected.append(turn)
        if len(selected) >= top_n:
            break
    if fallback is not None and fallback not in selected:
        if len(selected) >= top_n:
            selected[-1] = fallback
        else:
            selected.append(fallback)
    return selected[:top_n]


def _question_confidence(citations: list[Any]) -> str:
    if not citations:
        return "INSUFFICIENT"
    statuses = {citation.evidence_status.value for citation in citations}
    if "EXACT_VERIFIED" in statuses and len(citations) >= 1:
        return "HIGH"
    if "REJECTED" in statuses:
        return "LOW"
    return "MEDIUM"


def build_ground_truth_matrix(
    transcripts: list[TranscriptDocument],
    interview_guide_text_or_path: str | Path,
    provider: ExtractionProvider,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> GroundTruthMatrix:
    """Extracts all guide questions for all transcripts into a GroundTruthMatrix."""
    questions = parse_interview_guide(interview_guide_text_or_path)
    analyses: list[TranscriptAnalysis] = []

    for transcript in transcripts:
        answers: list[QuestionAnswer] = []
        for question in questions:
            candidate_turns = select_candidate_turns(question, transcript)
            extraction = provider.extract(
                question=question,
                candidate_turns=candidate_turns,
            )
            citations = resolve_evidence_turn_ids(
                transcript,
                extraction.evidence_turn_ids,
            )
            confidence = _question_confidence(citations)
            answers.append(
                QuestionAnswer(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    answer=extraction.answer.strip(),
                    evidence_turn_ids=list(extraction.evidence_turn_ids),
                    citations=citations,
                    confidence=confidence,  # type: ignore[arg-type]
                )
            )

        analyses.append(
            TranscriptAnalysis(
                transcript_id=transcript.transcript_id,
                expert_name=transcript.expert_name,
                market=transcript.market,
                answers=answers,
            )
        )

    return GroundTruthMatrix(
        source_hashes={t.transcript_id: t.file_hash for t in transcripts},
        model_name=provider.model_name,
        prompt_version=prompt_version,
        generated_at=GroundTruthMatrix.now_timestamp(),
        questions=questions,
        analyses=analyses,
    )


def save_ground_truth_matrix(matrix: GroundTruthMatrix, path: str | Path) -> None:
    """Persists ground-truth matrix to disk as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(matrix.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_ground_truth_matrix(path: str | Path) -> GroundTruthMatrix:
    """Loads ground-truth matrix from cached JSON."""
    return GroundTruthMatrix.model_validate_json(Path(path).read_text(encoding="utf-8"))


def cache_is_valid(
    matrix: GroundTruthMatrix,
    transcripts: list[TranscriptDocument],
    *,
    model_name: str,
    prompt_version: str,
) -> bool:
    """Validates if cache matches current transcript file hashes, model, and prompt."""
    expected_hashes = {t.transcript_id: t.file_hash for t in transcripts}
    return (
        matrix.source_hashes == expected_hashes
        and matrix.model_name == model_name
        and matrix.prompt_version == prompt_version
    )


def build_or_load_ground_truth_matrix(
    transcripts: list[TranscriptDocument],
    interview_guide_text_or_path: str | Path,
    provider: ExtractionProvider,
    cache_path: str | Path,
    *,
    prompt_version: str = PROMPT_VERSION,
    force: bool = False,
) -> GroundTruthMatrix:
    """Returns cached matrix if valid; otherwise generates and caches."""
    cache = Path(cache_path)
    if cache.exists() and not force:
        cached = load_ground_truth_matrix(cache)
        if cache_is_valid(
            cached,
            transcripts,
            model_name=provider.model_name,
            prompt_version=prompt_version,
        ):
            return cached

    matrix = build_ground_truth_matrix(
        transcripts,
        interview_guide_text_or_path,
        provider,
        prompt_version=prompt_version,
    )
    save_ground_truth_matrix(matrix, cache)
    return matrix


__all__ = [
    "ExtractionProvider",
    "LLMExtraction",
    "MockExtractionProvider",
    "OpenAIExtractionProvider",
    "build_ground_truth_matrix",
    "build_or_load_ground_truth_matrix",
    "cache_is_valid",
    "load_ground_truth_matrix",
    "save_ground_truth_matrix",
    "select_candidate_turns",
]
