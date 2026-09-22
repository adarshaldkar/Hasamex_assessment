from __future__ import annotations

from src.models.retrieval import Chunk
from src.models.transcript import DialogueTurn, TranscriptDocument


def chunk_transcript(transcript: TranscriptDocument) -> list[Chunk]:
    """Create one retrieval chunk per timestamp-bounded dialogue turn.

    Keeping chunk boundaries aligned to DialogueTurn preserves speaker, timestamp,
    and source-offset provenance for every retrieved result.
    """
    chunks: list[Chunk] = []
    for index, turn in enumerate(transcript.turns, start=1):
        chunks.append(
            Chunk(
                chunk_id=f"{transcript.transcript_id}_chunk_{index:03d}",
                transcript_id=transcript.transcript_id,
                expert_name=transcript.expert_name,
                role=transcript.role,
                market=transcript.market,
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                speaker_type=turn.speaker_type.value,
                timestamp_start=turn.timestamp,
                timestamp_end=turn.timestamp,
                text=turn.content,
                start_char=turn.start_char,
                end_char=turn.end_char,
                start_line=turn.start_line,
                end_line=turn.end_line,
            )
        )
    return chunks


def chunk_transcripts(transcripts: list[TranscriptDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for transcript in transcripts:
        chunks.extend(chunk_transcript(transcript))
    return chunks


__all__ = ["chunk_transcript", "chunk_transcripts"]
