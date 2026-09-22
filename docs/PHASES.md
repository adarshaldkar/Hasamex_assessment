# Implementation Phases & Execution Roadmap (Refined Architecture)

**Assessment**: Hasamex AI Engineer – Round 2 Technical Case  
**Domain**: European Robotic Surgery Market (Commercial Due Diligence)  
**Core Paradigm**: Closed-Loop Deterministic Evidence Extraction & Verification  
**Submission Deadline**: 23 September 2026, 11:00 AM IST  

---

## 1. Five Mandatory Architectural Corrections Incorporated

Based on rigorous assessment review, this roadmap incorporates five non-negotiable architectural principles:

1. **Deterministic Source Lookup over LLM Quote Generation**:
   - The LLM's role is to answer and **identify evidence turn IDs** (`evidence_turn_ids: list[str]`).
   - Python code deterministically fetches the verbatim source text, timestamps, speaker, character offsets (`start_char`, `end_char`), and line numbers (`start_line`, `end_line`) directly from the immutable transcript turn.
2. **Multi-Citation Support (`citations: list[QuoteCitation]`)**:
   - Answers frequently span multiple interview moments (e.g., Dr. Carter on ROI at `02:07` and `03:10`). Citations are modeled as lists of typed evidence objects.
3. **Transparent Evidence States**:
   - Replaces binary "100% verified" with explicit audited states:
     - `EXACT_VERIFIED` (100% verbatim substring match in source turn).
     - `NORMALIZED_VERIFIED` (whitespace/punctuation normalized match).
     - `FALLBACK_EVIDENCE` (LLM altered text; backend substituted verbatim turn text).
     - `REJECTED` (Turn ID or timestamp does not exist).
4. **Configurable Evidence Gate over Hardcoded Magic Thresholds**:
   - Replaces hardcoded distance numbers with an adaptive **Evidence Gate** evaluating fused RRF rank and token overlap before passing chunks to the LLM or triggering honest abstention.
5. **Comprehensive Evaluation Suite**:
   - Extends evaluation beyond Hit@K to: Citation Validity, Timestamp Exactness, Quote Verbatim Accuracy, Abstention Precision, and Schema Conformance.

---

## 2. End-to-End System Data Flow

```text
                 RAW TRANSCRIPT (data/raw/)
                       │
                       ↓
              Deterministic Parser
                       │
                       ↓
                 DialogueTurn
              ┌────────┼────────┐
              │        │        │
           Speaker  Timestamp  Offsets & Lines
        (expert/interviewer)    (start_char, start_line)
              │        │        │
              └────────┼────────┘
                       ↓
              Turn-based chunks
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
           BM25              Embeddings
          (Lexical)        (Sentence-Transformers)
             ↓                   ↓
             └─────────┬─────────┘
                       ↓
                      RRF (k=60)
                       ↓
                 Evidence Gate
                       │
              ┌────────┴────────┐
              ↓                 ↓
        Evidence Turns       NONE / WEAK
              ↓                 ↓
             LLM             Abstain
              ↓
       Structured Response
    (Answer + evidence_turn_ids)
              ↓
      Deterministic Source Lookup
              ↓
      Exact Source Quotes & Offsets
              ↓
       Verified Ground Truth
              ↓
     ┌────────┴────────┐
     ↓                 ↓
  Matrix           Synthesis
 (Cached JSON)   (Consensus & Divergence)
     │                 │
     └────────┬────────┘
              ↓
      5-Screen Streamlit UI
              ↓
    Evidence Drawer & Inspector
```

---

## 3. The 9 Phased Implementation Steps

### Phase 0: Environment & Single-Provider Configuration
* **Files**:
  - `src/config.py`
  - `.env` & `.env.example`
* **Specifications**:
  - Choose **one primary LLM provider** (e.g. Gemini 2.0 Flash or Groq Llama 3.3 70B via official SDK) + a deterministic `mock` mode for instant offline test suite execution.
  - Local CPU embeddings via `all-MiniLM-L6-v2` (Sentence-Transformers) for zero-cost, 10ms embedding.
  - Configurable evidence thresholds: `EVIDENCE_GATE_THRESHOLD = 0.50`, `RRF_K = 60`.
* **Verification**: Config imports cleanly; mock and live provider toggles function without error.

---

### Phase 1: Position-Preserving Parser & Immutable Schemas
* **Files**:
  - `src/models/transcript.py`
  - `src/core/parser.py`
  - `tests/test_parser.py`
* **Data Model**:
  ```python
  class DialogueTurn(BaseModel):
      turn_id: str                      # e.g., "france_turn_04"
      transcript_id: str                # e.g., "france_001"
      timestamp: str                    # e.g., "02:18"
      seconds: int                      # 138
      speaker: str                      # "Dr. Martin"
      speaker_type: Literal["expert", "interviewer"]
      role: str                         # "Head of Urology"
      market: str                       # "France"
      content: str                      # Raw speech content
      start_char: int                   # Exact character start in raw file
      end_char: int                     # Exact character end in raw file
      start_line: int                   # Exact line number start in file
      end_line: int                     # Exact line number end in file

  class TranscriptDocument(BaseModel):
      transcript_id: str
      file_hash: str                    # SHA-256 for idempotent ingestion
      expert_name: str
      role: str
      market: str
      raw_text: str
      turns: list[DialogueTurn]
      total_turns: int
  ```
* **Parser Logic**:
  - Regex header extraction (`Expert:`, `Role:`, `Market:`).
  - State-machine multi-line dialogue buffering bounded by `^(\d{2}:\d{2})\s*$`.
  - Classifies `speaker_type` (expert vs interviewer).
  - Character offsets via `raw_text.find(content)` and exact 1-indexed line numbers.
* **Verification**: `pytest tests/test_parser.py` asserts 100% turn recall for France, Germany, and UK; asserts `raw_text[start_char:end_char]` matches `turn.content` verbatim.

---

### Phase 2: Deterministic Evidence & Citation Engine
* **Files**:
  - `src/models/analysis.py`
  - `src/core/quote_verifier.py`
  - `tests/test_quote_verifier.py`
* **Data Model**:
  ```python
  class EvidenceStatus(str, Enum):
      EXACT_VERIFIED = "EXACT_VERIFIED"
      NORMALIZED_VERIFIED = "NORMALIZED_VERIFIED"
      FALLBACK_EVIDENCE = "FALLBACK_EVIDENCE"
      REJECTED = "REJECTED"

  class QuoteCitation(BaseModel):
      turn_id: str
      transcript_id: str
      expert_name: str
      speaker: str
      timestamp: str
      quote: str                        # Verbatim source text extracted by Python
      evidence_status: EvidenceStatus
      match_score: float
      start_char: int
      end_char: int
      start_line: int
      end_line: int

  class QuestionAnswer(BaseModel):
      question_id: str                  # "Q1" to "Q6"
      question_text: str
      answer: str
      citations: list[QuoteCitation]    # Multi-citation list
      confidence: Literal["HIGH", "MEDIUM", "LOW"]
  ```
* **Deterministic Lookup & Verification Logic**:
  1. LLM proposes `evidence_turn_ids: list[str]` and optional candidate quotes.
  2. Backend looks up target turn in `TranscriptDocument`.
  3. If candidate quote matches verbatim $\rightarrow$ `EXACT_VERIFIED` (score 1.0).
  4. If candidate quote has minor punctuation/whitespace variances $\rightarrow$ extracts exact slice from source turn $\rightarrow$ `NORMALIZED_VERIFIED` (score 0.95).
  5. If candidate quote is altered or missing $\rightarrow$ backend extracts the target sentence or full turn content directly $\rightarrow$ `FALLBACK_EVIDENCE` (score 0.85).
  6. If turn ID does not exist $\rightarrow$ `REJECTED`.
* **Verification**: `pytest tests/test_quote_verifier.py` validates all 4 evidence states.

---

### Phase 3: Question-Aware Retrieval & Structured Extraction
* **Files**:
  - `src/prompts/extraction.py`
  - `src/core/extractor.py`
  - `storage/processed/ground_truth_matrix.json`
  - `tests/test_extractor.py`
* **Logic**:
  - For each of the 6 interview guide questions per transcript:
    1. Filter dialogue turns to relevant candidate turns (reducing noise & token waste).
    2. Prompt LLM to return `answer` and `evidence_turn_ids`.
    3. Pass through Phase 2 deterministic lookup to assemble `list[QuoteCitation]`.
  - Assemble the $6 \times 3$ Ground-Truth Matrix.
  - Persist to `storage/processed/ground_truth_matrix.json` with SHA-256 file hash.
* **Verification**: `pytest tests/test_extractor.py` asserts all 18 cells populated, all citations have valid turn IDs, and zero unverified quotes.

---

### Phase 4: Ground-Truth Matrix & Cross-Expert Synthesis
* **Files**:
  - `src/models/synthesis.py`
  - `src/prompts/synthesis.py`
  - `src/core/synthesizer.py`
* **Data Model**:
  ```python
  class ConsensusTheme(BaseModel):
      title: str
      description: str
      evidence_by_country: dict[str, list[str]]

  class DisagreementPoint(BaseModel):
      topic: str
      description: str
      stances_by_stakeholder: dict[str, str]

  class MarketGrowthSpectrum(BaseModel):
      conservative: list[str]  # Germany (Anna Keller): High single digits
      moderate: list[str]      # France (Dr. Martin): 15-20% in top centres
      bullish: list[str]       # UK (Dr. Carter): >15% if training bottlenecks ease

  class SynthesisReport(BaseModel):
      consensus_themes: list[ConsensusTheme]
      disagreements: list[DisagreementPoint]
      growth_spectrum: MarketGrowthSpectrum
  ```
* **Logic**: Runs strictly over the verified Ground-Truth Matrix. Caches output to `storage/processed/synthesis_report.json`.
* **Verification**: Inspect generated report; ensure distinct separation of consensus vs disagreements.

---

### Phase 5: Hybrid RAG Engine with Evidence Gate & Abstention
* **Files**:
  - `src/core/chunker.py` (Dialogue-turn chunker preserving speaker & lines).
  - `src/core/embedder.py` (Sentence-Transformers `all-MiniLM-L6-v2` singleton).
  - `src/core/retriever.py` (BM25 + ChromaDB + RRF $k=60$).
  - `src/core/rag_engine.py` (Attributed Q&A with Evidence Gate).
  - `src/prompts/rag.py`
  - `tests/test_rag.py`
* **Evidence Gate Logic**:
  ```python
  def evidence_gate(retrieved_turns: list[DialogueTurn], query: str) -> tuple[bool, list[DialogueTurn]]:
      # Filter to expert turns (avoid interviewer questions as facts)
      expert_turns = [t for t in retrieved_turns if t.speaker_type == "expert"]
      if not expert_turns:
          return False, []
      
      # Combined relevance check (RRF score threshold)
      top_score = expert_turns[0].rrf_score
      if top_score < settings.evidence_gate_threshold:
          return False, []
      
      return True, expert_turns[:4]
  ```
* **Honest Abstention**: If `evidence_gate` returns `False`, RAG returns:
  > *"The provided transcripts do not contain sufficient evidence to answer this question."*
* **Verification**: `pytest tests/test_rag.py` tests retrieval on clinical acronyms (`NHS`, `TCO`) and asserts abstention on out-of-scope questions.

---

### Phase 6: 5-Screen Interactive Streamlit Dashboard & Evidence Drawer
* **Files**:
  - `app/streamlit_app.py`
  - `app/api.py` (FastAPI REST service)
* **Screen Specifications**:
  1. **Screen 1: Comparative Matrix ($6 \times 3$)**:
     - Interactive grid with question rows and country columns.
     - Each cell shows answer, green `✓ Exact Verified` badge, and timestamp pill buttons.
     - **Evidence Drawer**: Clicking a timestamp pill opens a slide-over modal showing the verbatim quote, speaker, and character offset, with a button: *"Open in Full Transcript Inspector"*.
  2. **Screen 2: Synthesis & Insights**:
     - Left: Shared Consensus Cards with multi-expert quotes.
     - Right: Strategic Disagreements (Financial Primacy vs Clinical Strategy).
     - Bottom: Visual Market Growth Spectrum (Conservative $\leftrightarrow$ Moderate $\leftrightarrow$ Bullish).
  3. **Screen 3: Conversational Cross-Transcript RAG**:
     - Filter sidebar: Filter by Market (`All`, `France`, `Germany`, `UK`) or Role (`All`, `Clinician`, `Procurement`).
     - Chat input with grounded streaming answers and interactive citation cards.
     - Demonstrates honest refusal when asked out-of-scope questions.
  4. **Screen 4: Raw Transcript Inspector**:
     - Full transcript reader with line numbers.
     - Auto-scrolls and highlights the target line when navigated from an Evidence Drawer.
  5. **Screen 5: Document Upload & Dynamic Ingestion**:
     - Drag-and-drop `.txt` uploader with metadata input fallback if headers are missing.
     - Computes SHA-256 to prevent duplicate indexing (idempotent).
     - Dynamically indexes and appends the new expert column to the matrix.

---

### Phase 7: Comprehensive Evaluation & Quality Harness
* **Files**:
  - `eval/eval.py`
  - `eval/eval_dataset.json`
* **Test Metrics**:
  1. **Retrieval Hit@K**: Expected turn appears in top-$k$.
  2. **Citation Validity**: Citation points to the correct speaker and timestamp.
  3. **Quote Exactness**: Verbatim match ratio between citation and source text.
  4. **Abstention Precision**: System correctly refuses out-of-domain queries.
  5. **Schema Conformance**: 100% valid Pydantic outputs across runs.
* **Verification**: Run `python eval/eval.py` and display a clean metric report table.

---

### Phase 8: Production README, Limitations & PiP Video Demo
* **Files**:
  - `README.md` (Quickstart, architecture, tech stack, evaluation results, and **Limitations** section).
  - `docs/INTERVIEW_DEMO_SCRIPT.md`
* **Video Recording (7–8 Minutes)**:
  - Setup: OBS Studio / Loom with webcam in picture-in-picture mode.
  - Follow the 7–8 minute script:
    1. Problem & Context (00:00–00:40)
    2. Architecture & Closed-Loop Design (00:40–01:30)
    3. Tech Stack & Engineering Decisions (01:30–02:20)
    4. **Mandatory AI Usage Disclosure** (02:20–03:00)
    5. Key Engineering Challenges (03:00–04:00)
    6. Live Working Demo of the 5 Screens + Abstention Guardrail (04:00–07:20)
    7. Scalability & Limitations Wrap-Up (07:20–08:00)

---

## 4. Execution Order for Tonight

```text
Step 1: Phase 0 (Config) + Phase 1 (Parser & DialogueTurn models + pytest)
Step 2: Phase 2 (Quote Verifier & Evidence model + pytest)
Step 3: Phase 3 (Structured Extractor & Ground-Truth Matrix cache)
Step 4: Phase 4 (Cross-Expert Synthesis Engine)
Step 5: Phase 5 (Hybrid RAG: BM25 + Chroma + RRF + Evidence Gate)
Step 6: Phase 6 (5-Screen Streamlit Dashboard + Evidence Drawer + Inspector)
Step 7: Phase 7 (Evaluation harness eval/eval.py)
Step 8: Phase 8 (README with Limitations + Video rehearsal)
```
