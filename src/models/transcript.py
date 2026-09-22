"""Transcript data models with character-level and line-level provenance."""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class SpeakerType(str, Enum):
    EXPERT = "expert"
    INTERVIEWER = "interviewer"


class DialogueTurn(BaseModel):
    """A single timestamp-bounded dialogue turn with strict source provenance."""

    model_config = ConfigDict(frozen=True)

    turn_id: str = Field(description="Unique turn identifier, e.g. france_001_turn_04")
    transcript_id: str = Field(description="Identifier of parent transcript, e.g. france_001")
    timestamp: str = Field(description="Original timestamp string, e.g. 02:18")
    seconds: int = Field(description="Timestamp converted to total seconds from start")
    speaker: str = Field(description="Cleaned speaker name, e.g. Dr. Martin")
    speaker_type: SpeakerType = Field(description="Classification: expert vs interviewer")
    role: str = Field(description="Professional role / title of the expert")
    market: str = Field(description="Country or jurisdiction, e.g. France")
    content: str = Field(description="Verbatim speech text of this turn")
    start_char: int = Field(ge=0, description="0-indexed character start in raw source file")
    end_char: int = Field(ge=0, description="0-indexed character end in raw source file")
    start_line: int = Field(ge=1, description="1-indexed line number where speech starts")
    end_line: int = Field(ge=1, description="1-indexed line number where speech ends")


class TranscriptDocument(BaseModel):
    """Immutable parsed representation of one source transcript."""

    model_config = ConfigDict(frozen=True)

    transcript_id: str = Field(description="Stable document ID, e.g. france_001")
    file_hash: str = Field(description="SHA-256 hash of original raw transcript file")
    filename: str = Field(description="Source file name, e.g. Transcript_1_France.txt")
    expert_name: str = Field(description="Full name of expert, e.g. Dr. Jean Martin")
    role: str = Field(description="Expert's organizational role, e.g. Head of Urology")
    market: str = Field(description="Geographic market, e.g. France")
    raw_text: str = Field(description="Immutable original file content")
    turns: list[DialogueTurn] = Field(default_factory=list, description="Ordered dialogue turns")
    total_turns: int = Field(default=0, description="Total number of turns extracted")
    duration_str: str = Field(default="00:00", description="Final timestamp in transcript")
