# Phase 1: Position-Preserving Data Models & Dialogue Parser

**Document**: `docs/PHASE1.md`  
**Phase Objective**: Transform raw, unstructured transcript `.txt` files into immutable, strictly typed Pydantic models with character-level and line-level provenance, speaker-type classification, and SHA-256 idempotency.  
**Deliverable Files**:
1. `src/models/transcript.py` (Domain Data Schemas)
2. `src/core/parser.py` (State-Machine Regex Parser)
3. `tests/test_parser.py` (Automated Unit Test Suite)

---

## 1. Architectural Role of Phase 1

Phase 1 is the **provenance bedrock** of the entire platform. 

In commercial due diligence, every AI-synthesized insight must trace back to the exact words spoken by an expert. If the parser introduces off-by-one errors in line numbers, loses character offsets, or merges interviewer questions with expert statements, all downstream verification, RAG, and citation jumping will fail.

```mermaid
flowchart TD
    RAW[Raw .txt Transcript] --> SHA[Compute SHA-256 Hash]
    RAW --> HEADER[Header Parser<br>Regex: Expert, Role, Market]
    RAW --> STATE_MACHINE[Dialogue State Machine<br>Regex: ^MM:SS$]
    
    subgraph StateMachine ["State Machine Turn Extraction"]
        DETECT_TS[Detect Timestamp Anchor]
        SPLIT_SPK[Identify Speaker Prefix & Type]
        BUFFER[Buffer Multi-Line Speech Payload]
        CALC_POS[Calculate start_char, end_char, start_line, end_line]
    end
    
    STATE_MACHINE --> StateMachine
    StateMachine --> TURNS[List of DialogueTurn Objects]
    HEADER & SHA & TURNS --> DOC[TranscriptDocument]
    DOC --> AUDIT{Integrity Audit<br>raw_text[start_char:end_char] == content}
    AUDIT -- 100% Match --> STORE[Cache to storage/processed/]
```

---

## 2. Detailed Data Schema Specification (`src/models/transcript.py`)

### 2.1 Enumerations
```python
from enum import Enum

class SpeakerType(str, Enum):
    EXPERT = "expert"
    INTERVIEWER = "interviewer"
```

### 2.2 `DialogueTurn`
Represents a single atomic conversational turn bounded by a timestamp:

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `turn_id` | `str` | Stable hierarchical identifier | `"france_001_turn_04"` |
| `transcript_id` | `str` | Identifier of parent transcript | `"france_001"` |
| `timestamp` | `str` | Original `MM:SS` string from header | `"02:18"` |
| `seconds` | `int` | Integer seconds from start of interview | `138` (2 min * 60 + 18 sec) |
| `speaker` | `str` | Cleaned speaker label | `"Dr. Martin"` |
| `speaker_type` | `SpeakerType` | Semantic classification | `SpeakerType.EXPERT` |
| `role` | `str` | Professional title of the expert | `"Head of Urology"` |
| `market` | `str` | Country jurisdiction | `"France"` |
| `content` | `str` | Verbatim speech payload | `"Very important. The clinical argument..."` |
| `start_char` | `int` | Exact 0-indexed character start in raw text | `482` |
| `end_char` | `int` | Exact 0-indexed character end in raw text | `698` |
| `start_line` | `int` | Exact 1-indexed starting line number | `20` |
| `end_line` | `int` | Exact 1-indexed ending line number | `22` |

### 2.3 `TranscriptDocument`
Represents the complete parsed transcript with file-level metadata:

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `transcript_id` | `str` | Unique document ID | `"france_001"` |
| `file_hash` | `str` | SHA-256 hash of original raw content | `"e3b0c44298fc1c149afbf4c8996fb924..."` |
| `filename` | `str` | Name of source file on disk | `"Transcript_1_France.txt"` |
| `expert_name` | `str` | Extracted expert full name | `"Dr. Jean Martin"` |
| `role` | `str` | Expert title / function | `"Head of Urology"` |
| `market` | `str` | Geographic market / country | `"France"` |
| `raw_text` | `str` | Immutable, original raw file string | `"Expert 1 – Dr. Jean Martin..."` |
| `turns` | `list[DialogueTurn]` | Chronological sequence of turns | `[turn_01, turn_02, ...]` |
| `total_turns` | `int` | Total count of turns | `8` |
| `duration_str` | `str` | Timestamp of final turn | `"06:08"` |

---

## 3. Parser State Machine Implementation Details (`src/core/parser.py`)

### 3.1 Step 1: Metadata Header Extraction
The parser expects header blocks formatted like:
```text
Expert 1 – Dr. Jean Martin
Role: Head of Urology
Market: France
```
* **Regex Patterns**:
  - `expert_name`: `r"^Expert\s+\d+\s*[-–—]\s*(.+)$"`
  - `role`: `r"^Role:\s*(.+)$"`
  - `market`: `r"^Market:\s*(.+)$"`

### 3.2 Step 2: Timestamp-Anchored State Machine
Transcripts alternate between timestamp markers and dialogue:
```text
02:14
Interviewer: So ROI is important?

02:18
Dr. Martin: Very important. The clinical argument may get surgeons interested, but...
```
* **Anchor Regex**: `^(\d{2}):(\d{2})\s*$`
  - Group 1: Minutes (`int(m.group(1))`)
  - Group 2: Seconds (`int(m.group(2))`)
  - Total seconds: `minutes * 60 + seconds`
* **Speaker Separation**:
  - Regex: `^([^:\n]+):\s*(.*)$`
  - Group 1: Speaker name (e.g. `Interviewer`, `Dr. Martin`, `Anna Keller`)
  - Group 2: First line of dialogue speech.
* **Speaker Classification**:
  - If `"interviewer"` in speaker name (case-insensitive) $\rightarrow$ `SpeakerType.INTERVIEWER`.
  - Otherwise $\rightarrow$ `SpeakerType.EXPERT`.
* **Multi-Line Speech Buffering**:
  - Many speeches span 2–5 lines before the next timestamp anchor appears.
  - The state machine buffers all lines until a new timestamp anchor is matched or EOF is reached.

### 3.3 Step 3: Exact Character & Line Provenance Tracking
To guarantee that `start_char` and `end_char` map with 100% accuracy:
```python
# Locate content in raw text starting from the current search cursor
content_start = raw_text.find(turn_content, current_cursor)
content_end = content_start + len(turn_content)

# Assert invariant
assert raw_text[content_start:content_end] == turn_content
```
Line numbers are computed by tracking `1 + raw_text[:content_start].count('\n')`.

---

## 4. Ground-Truth Analysis of the 3 Assessment Transcripts

The parser must yield these exact turn counts and distributions:

### Transcript 1: France (`Transcript_1_France.txt`)
* **Expert**: Dr. Jean Martin | **Role**: Head of Urology | **Market**: France
* **Total Turns**: 8 turns (4 Interviewer, 4 Expert)
* **Turn Sequence**:
  - `turn_01` (00:00): Interviewer (Lines 5–6)
  - `turn_02` (00:18): Dr. Martin (Lines 8–9) $\rightarrow$ *Current adoption*
  - `turn_03` (01:12): Interviewer (Lines 11–12)
  - `turn_04` (01:20): Dr. Martin (Lines 14–15) $\rightarrow$ *Barriers (capital approval)*
  - `turn_05` (02:14): Interviewer (Lines 17–18)
  - `turn_06` (02:18): Dr. Martin (Lines 20–21) $\rightarrow$ *ROI & payback*
  - `turn_07` (03:05): Interviewer (Lines 23–24)
  - `turn_08` (03:10): Dr. Martin (Lines 26–27) $\rightarrow$ *Training multi-surgeons*
  - *(Continues to turn_14 at 06:08)*

### Transcript 2: Germany (`Transcript_2_Germany.txt`)
* **Expert**: Anna Keller | **Role**: Former Hospital Procurement Director | **Market**: Germany
* **Total Turns**: 14 turns across 46 lines
* **First Expert Turn**: `00:16` (Anna Keller)
* **Final Expert Turn**: `06:05` (Anna Keller: 9–18 month purchase timeline)

### Transcript 3: UK (`Transcript_3_UK.txt`)
* **Expert**: Dr. Emily Carter | **Role**: Consultant Urologist | **Market**: United Kingdom
* **Total Turns**: 14 turns across 46 lines
* **First Expert Turn**: `00:14` (Dr. Carter)
* **Final Expert Turn**: `06:04` (Dr. Carter: Sustainable training & volume)

---

## 5. Automated Test Suite Specification (`tests/test_parser.py`)

The unit test suite validates 5 critical invariants:

1. **`test_france_transcript_parsing()`**:
   - Asserts metadata: `Dr. Jean Martin`, `Head of Urology`, `France`.
   - Asserts first turn is at `00:00` with `speaker_type == SpeakerType.INTERVIEWER`.
   - Asserts turn at `02:18` contains `"Very important. The clinical argument..."`.
2. **`test_germany_transcript_parsing()`**:
   - Asserts metadata: `Anna Keller`, `Former Hospital Procurement Director`, `Germany`.
   - Asserts turn at `02:08` contains `"We look at total cost of ownership..."`.
3. **`test_uk_transcript_parsing()`**:
   - Asserts metadata: `Dr. Emily Carter`, `Consultant Urologist`, `United Kingdom`.
   - Asserts turn at `01:05` contains `"Funding is important, but I would say training capacity..."`.
4. **`test_provenance_character_offset_invariant()`**:
   - Loops over every single parsed turn across all 3 files.
   - Asserts `raw_text[turn.start_char:turn.end_char] == turn.content`.
   - Fails if even a single character or whitespace offset is misaligned.
5. **`test_speaker_type_classification()`**:
   - Asserts all Interviewer turns have `SpeakerType.INTERVIEWER`.
   - Asserts all clinician/procurement turns have `SpeakerType.EXPERT`.

---

## 6. Execution Command Sequence

To implement Phase 1:
```bash
# 1. Create data schemas in src/models/transcript.py
# 2. Implement parser in src/core/parser.py
# 3. Create unit tests in tests/test_parser.py
# 4. Run pytest
pytest tests/test_parser.py -v
```

Upon passing, Phase 1 is **100% complete and verified**, enabling Phase 2 (Deterministic Evidence & Citation Engine).
