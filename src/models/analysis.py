"""Analysis and evaluation schemas with evidence states, multi-citation, and cache versioning."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(str, Enum):
    """Audited provenance state of a citation."""
    EXACT_VERIFIED = "EXACT_VERIFIED"
    NORMALIZED_VERIFIED = "NORMALIZED_VERIFIED"
    FALLBACK_EVIDENCE = "FALLBACK_EVIDENCE"
    REJECTED = "REJECTED"


class QuoteCitation(BaseModel):
    """Structured citation object linking a claim directly to source transcript coordinates."""
    model_config = ConfigDict(frozen=True)

    turn_id: str
    transcript_id: str
    expert_name: str
    speaker: str
    timestamp: str
    quote: str
    evidence_status: EvidenceStatus
    match_score: float = Field(ge=0.0, le=1.0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class QuestionAnswer(BaseModel):
    """Structured answer to a specific interview guide question for one expert."""
    model_config = ConfigDict(frozen=True)

    question_id: str
    question_text: str
    answer: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    citations: list[QuoteCitation] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT"]


class InterviewQuestion(BaseModel):
    """Dynamic representation of an interview guide question."""
    model_config = ConfigDict(frozen=True)

    question_id: str
    question_text: str

    @property
    def text(self) -> str:
        return self.question_text


class TranscriptAnalysis(BaseModel):
    """All answered questions for a specific expert transcript."""
    model_config = ConfigDict(frozen=True)

    transcript_id: str
    expert_name: str
    market: str
    answers: list[QuestionAnswer]


class GroundTruthMatrix(BaseModel):
    """Precomputed ground-truth matrix across all experts and questions with cache versioning."""
    model_config = ConfigDict(frozen=True)

    matrix_version: str = "ground_truth_v1"
    source_hashes: dict[str, str]
    model_name: str
    prompt_version: str
    generated_at: str
    questions: list[InterviewQuestion]
    analyses: list[TranscriptAnalysis]

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat()
