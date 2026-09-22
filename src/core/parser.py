"""Position-preserving state-machine transcript parser with character and line provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src.models.transcript import DialogueTurn, SpeakerType, TranscriptDocument


TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2})\s*$")
SPEAKER_RE = re.compile(r"^([^:\n\r]+):\s*(.*)$")
EXPERT_HEADER_RE = re.compile(r"^Expert\s+\d+\s*[-–—]\s*(.+?)\s*$")
ROLE_RE = re.compile(r"^Role:\s*(.+?)\s*$")
MARKET_RE = re.compile(r"^Market:\s*(.+?)\s*$")


@dataclass(frozen=True)
class _RawLine:
    """Raw line plus its absolute character offsets in the preserved raw text."""
    text: str
    start: int
    content_end: int
    line_number: int


@dataclass
class _TurnBuffer:
    timestamp: str
    seconds: int
    speaker: str
    speaker_type: SpeakerType
    content_start: int
    content_end: int
    start_line: int
    end_line: int


def compute_sha256(raw_bytes: bytes) -> str:
    """Computes SHA-256 hash of raw bytes for idempotent ingestion."""
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_file_hash(text_or_bytes: str | bytes) -> str:
    """Computes SHA-256 hash of string or bytes."""
    if isinstance(text_or_bytes, str):
        return hashlib.sha256(text_or_bytes.encode("utf-8")).hexdigest()
    return hashlib.sha256(text_or_bytes).hexdigest()


def generate_transcript_id(filename: str, market: str) -> str:
    """Generates a canonical transcript ID from filename or market."""
    clean_name = Path(filename).stem.lower()
    if "france" in clean_name or "france" in market.lower():
        return "france_001"
    elif "germany" in clean_name or "germany" in market.lower():
        return "germany_001"
    elif "uk" in clean_name or "united kingdom" in market.lower():
        return "uk_001"
    sanitized = re.sub(r"[^a-z0-9]+", "_", clean_name).strip("_")
    return f"{sanitized}_001" if sanitized else "transcript_001"


def _split_raw_lines(raw_text: str) -> list[_RawLine]:
    lines: list[_RawLine] = []
    cursor = 0

    for line_number, line in enumerate(raw_text.splitlines(keepends=True), start=1):
        start = cursor
        cursor += len(line)
        content_end = start + len(line.rstrip("\r\n"))
        lines.append(
            _RawLine(
                text=line,
                start=start,
                content_end=content_end,
                line_number=line_number,
            )
        )

    return lines


def _clean_header_match(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.match(line.strip("\r\n"))
    return match.group(1).strip() if match else None


def _parse_timestamp(line: str) -> tuple[str, int] | None:
    match = TIMESTAMP_RE.match(line.strip("\r\n").strip())
    if not match:
        return None

    minutes = int(match.group(1))
    seconds = int(match.group(2))
    if seconds >= 60:
        raise ValueError(f"Invalid timestamp seconds value: {line!r}")

    return f"{minutes:02d}:{seconds:02d}", minutes * 60 + seconds


def _classify_speaker(speaker: str) -> SpeakerType:
    return (
        SpeakerType.INTERVIEWER
        if "interviewer" in speaker.casefold()
        else SpeakerType.EXPERT
    )


def _iter_nonempty_after_timestamp(
    lines: list[_RawLine],
    start_index: int,
) -> tuple[int, _RawLine] | None:
    for index in range(start_index, len(lines)):
        if lines[index].text.strip():
            return index, lines[index]
        if _parse_timestamp(lines[index].text) is not None:
            break
    return None


def _finalize_turn(
    *,
    buffer: _TurnBuffer,
    transcript_id: str,
    role: str,
    market: str,
    raw_text: str,
    turn_number: int,
) -> DialogueTurn:
    content = raw_text[buffer.content_start : buffer.content_end]
    if not content:
        raise ValueError(
            f"Empty dialogue payload for {transcript_id} at {buffer.timestamp}"
        )

    start_line = 1 + raw_text[: buffer.content_start].count("\n")
    end_line = 1 + raw_text[: buffer.content_end].count("\n")

    return DialogueTurn(
        turn_id=f"{transcript_id}_turn_{turn_number:02d}",
        transcript_id=transcript_id,
        timestamp=buffer.timestamp,
        seconds=buffer.seconds,
        speaker=buffer.speaker,
        speaker_type=buffer.speaker_type,
        role=role,
        market=market,
        content=content,
        start_char=buffer.content_start,
        end_char=buffer.content_end,
        start_line=start_line,
        end_line=end_line,
    )


def parse_transcript_text(
    raw_text: str,
    *,
    filename: str = "transcript.txt",
    transcript_id: str | None = None,
    file_hash: str | None = None,
) -> TranscriptDocument:
    """Parse one timestamped transcript while preserving exact source offsets."""
    lines = _split_raw_lines(raw_text)
    if not lines:
        raise ValueError("Transcript is empty")

    expert_name: str | None = None
    role: str | None = None
    market: str | None = None

    for raw_line in lines:
        stripped = raw_line.text.strip("\r\n")
        if expert_name is None:
            expert_name = _clean_header_match(EXPERT_HEADER_RE, stripped)
        if role is None:
            role = _clean_header_match(ROLE_RE, stripped)
        if market is None:
            market = _clean_header_match(MARKET_RE, stripped)

    missing = [
        name
        for name, value in (
            ("expert_name", expert_name),
            ("role", role),
            ("market", market),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required transcript metadata: {', '.join(missing)}")

    if transcript_id is None:
        transcript_id = generate_transcript_id(filename, market or "")

    turns: list[DialogueTurn] = []
    active: _TurnBuffer | None = None
    current_cursor = 0
    index = 0

    while index < len(lines):
        raw_line = lines[index]
        timestamp_info = _parse_timestamp(raw_line.text)

        if timestamp_info is None:
            index += 1
            continue

        # Finalize the previous turn before starting next
        if active is not None:
            turns.append(
                _finalize_turn(
                    buffer=active,
                    transcript_id=transcript_id,
                    role=role,  # type: ignore[arg-type]
                    market=market,  # type: ignore[arg-type]
                    raw_text=raw_text,
                    turn_number=len(turns) + 1,
                )
            )
            current_cursor = active.content_end
            active = None

        timestamp, seconds = timestamp_info
        speaker_info = _iter_nonempty_after_timestamp(lines, index + 1)
        if speaker_info is None:
            raise ValueError(f"Timestamp {timestamp} has no dialogue content")

        speaker_index, speaker_line = speaker_info
        speaker_match = SPEAKER_RE.match(speaker_line.text.rstrip("\r\n"))
        if not speaker_match:
            raise ValueError(
                f"Could not parse speaker prefix after timestamp {timestamp}: "
                f"{speaker_line.text!r}"
            )

        speaker = speaker_match.group(1).strip()
        first_content = speaker_match.group(2)

        first_line_content_start = speaker_line.start + speaker_line.text.index(":") + 1
        while (
            first_line_content_start < speaker_line.content_end
            and raw_text[first_line_content_start] in " \t"
        ):
            first_line_content_start += 1

        expected_first_content = raw_text[
            first_line_content_start : speaker_line.content_end
        ]
        if expected_first_content != first_content:
            raise AssertionError(
                f"Speaker content mismatch at {timestamp}: "
                f"{expected_first_content!r} != {first_content!r}"
            )

        if first_line_content_start < current_cursor:
            raise AssertionError("Parser cursor moved backwards")

        content_end = speaker_line.content_end
        end_line = speaker_line.line_number
        scan_index = speaker_index + 1

        while scan_index < len(lines):
            continuation = lines[scan_index]
            if _parse_timestamp(continuation.text) is not None:
                break

            if continuation.text.strip():
                content_end = continuation.content_end
                end_line = continuation.line_number
            else:
                if continuation.line_number < len(lines):
                    content_end = continuation.content_end
                    end_line = continuation.line_number
            scan_index += 1

        while content_end > first_line_content_start and raw_text[content_end - 1] in "\r\n":
            content_end -= 1

        active = _TurnBuffer(
            timestamp=timestamp,
            seconds=seconds,
            speaker=speaker,
            speaker_type=_classify_speaker(speaker),
            content_start=first_line_content_start,
            content_end=content_end,
            start_line=speaker_line.line_number,
            end_line=end_line,
        )
        index = scan_index

    if active is not None:
        turns.append(
            _finalize_turn(
                buffer=active,
                transcript_id=transcript_id,
                role=role,  # type: ignore[arg-type]
                market=market,  # type: ignore[arg-type]
                raw_text=raw_text,
                turn_number=len(turns) + 1,
            )
        )

    if not turns:
        raise ValueError("No timestamped dialogue turns found")

    # Final provenance invariant audit
    for turn in turns:
        if raw_text[turn.start_char : turn.end_char] != turn.content:
            raise AssertionError(
                f"Provenance invariant failed for {turn.turn_id} ({turn.timestamp})"
            )

    hash_value = file_hash or hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    return TranscriptDocument(
        transcript_id=transcript_id,
        file_hash=hash_value,
        filename=filename,
        expert_name=expert_name,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        market=market,  # type: ignore[arg-type]
        raw_text=raw_text,
        turns=turns,
        total_turns=len(turns),
        duration_str=turns[-1].timestamp,
    )


def parse_transcript_file(
    file_path: str | Path,
    transcript_id: str | None = None,
) -> TranscriptDocument:
    """Read a UTF-8 transcript from disk and parse it."""
    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {source_path}")

    raw_bytes = source_path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")
    file_hash = compute_sha256(raw_bytes)

    if transcript_id is None:
        transcript_id = generate_transcript_id(source_path.name, "")

    return parse_transcript_text(
        raw_text=raw_text,
        filename=source_path.name,
        transcript_id=transcript_id,
        file_hash=file_hash,
    )
