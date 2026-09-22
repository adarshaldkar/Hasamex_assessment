from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievalSource(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    RRF = "rrf"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    transcript_id: str
    expert_name: str
    role: str
    market: str
    turn_id: str
    speaker: str
    speaker_type: str
    timestamp_start: str
    timestamp_end: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    normalized_rrf_score: float = 0.0
    token_overlap: float = 0.0
    fused_relevance: float = 0.0


class EvidenceGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str
    evidence: list[RetrievalResult] = Field(default_factory=list)


class RAGResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    abstained: bool = False


class ProviderError(RuntimeError):
    pass


__all__ = [
    "Chunk",
    "EvidenceGateResult",
    "ProviderError",
    "RAGResponse",
    "RetrievalResult",
    "RetrievalSource",
]
