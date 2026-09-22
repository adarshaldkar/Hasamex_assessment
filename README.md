# Transcript Intelligence Pro — Hasamex AI Engineer Technical Case

A closed-loop qualitative document intelligence application for qualitative commercial due diligence on the European robotic surgery market.

---

## 1. Executive Summary & Core Capabilities

The application ingests timestamped expert transcripts across **France** (Dr. Jean Martin), **Germany** (Anna Keller), and the **UK** (Dr. Emily Carter), answers the 6 standardized interview-guide questions, verifies evidence against the original transcript source, synthesizes cross-expert themes and disagreements, and exposes a conversational RAG interface with deterministic citations.

---

## 2. System Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion ["Stage 1: Position-Preserving Ingestion & Indexing"]
        TXT["📄 Raw .txt Transcripts\n(France, Germany, UK)"] --> PARSER["Deterministic Parser\n(Regex + SHA-256)"]
        PARSER --> TURNS["DialogueTurn Models\n[start_char, end_char, start_line, end_line]"]
        TURNS --> CHUNKER["Contextual Dialogue Chunker"]
        CHUNKER --> BM25["BM25 Lexical Index"]
        CHUNKER --> VEC["Chroma Dense Vector Index\n(all-MiniLM-L6-v2)"]
    end

    subgraph ExtractionVerification ["Stage 2: Closed-Loop Verification & Matrix Extraction"]
        TURNS --> EXTRACT["6-Question Extractor\n(Candidate Answers + Turn IDs)"]
        EXTRACT --> VERIFIER{"Deterministic Quote Verifier\n(Exact / Normalized / Fallback)"}
        VERIFIER -->|"Verified Citations"| MATRIX["Ground-Truth Matrix (6x3 Grid)\n(storage/processed/ground_truth_matrix.json)"]
    end

    subgraph CrossExpertSynthesis ["Stage 3: Cross-Expert Synthesis"]
        MATRIX --> SYNTH["Synthesis Engine"]
        SYNTH --> REPORT["Synthesis Report\n• Shared Consensus Themes\n• Strategic Disagreements\n• 3-Year Growth Spectrum"]
    end

    subgraph HybridRAG ["Stage 4: Hybrid RAG & Guardrailed QA"]
        QUERY["User / CDD Analyst Query"] --> HYBRID["Hybrid Retriever\n(BM25 + Dense)"]
        BM25 & VEC --> HYBRID
        HYBRID --> RRF["Reciprocal Rank Fusion\n(RRF k=60)"]
        RRF --> GATE{"Evidence Gate\n(Max RRF Score >= 0.50)"}
        GATE -->|"Sufficient Evidence"| LLM["LLM Response Generator\n(Extracts Evidence Turn IDs)"]
        GATE -->|"Insufficient Evidence"| ABSTAIN["Deterministic Abstention Guardrail\n('Insufficient evidence in transcripts')"]
        LLM --> LOOKUP["Deterministic Source Lookup\n(Injects Verbatim Quotes & Lines)"]
    end

    subgraph UI ["Stage 5: Presentation & Interactive UI"]
        MATRIX --> ST_MATRIX["1. Matrix View\n(6x3 Grid + Evidence Drawer)"]
        REPORT --> ST_SYNTH["2. Synthesis View\n(Consensus & Conflicts)"]
        LOOKUP & ABSTAIN --> ST_CHAT["3. AI Chat View\n(Interactive RAG + Deep Citations)"]
        TURNS --> ST_INSPECT["4. Transcript Inspector\n(Line-Numbered + Auto-Scroll)"]
        UPLOAD["Drag-and-Drop Upload"] --> ST_UPLOAD["5. Upload View\n(Dynamic Session Rebuild)"]
    end
```

---

## 3. Data Model & Schema Diagram

```mermaid
classDiagram
    class TranscriptDocument {
        +string transcript_id
        +string file_hash (SHA-256)
        +string filename
        +string expert_name
        +string role
        +string market
        +string raw_text
        +list~DialogueTurn~ turns
        +int total_turns
        +string duration_str
    }

    class DialogueTurn {
        +string turn_id
        +string transcript_id
        +string timestamp
        +int seconds
        +string speaker
        +SpeakerType speaker_type
        +string role
        +string market
        +string content
        +int start_char
        +int end_char
        +int start_line
        +int end_line
    }

    class QuoteCitation {
        +string turn_id
        +string transcript_id
        +string expert_name
        +string speaker
        +string timestamp
        +string quote
        +EvidenceStatus evidence_status
        +float match_score
        +int start_char
        +int end_char
        +int start_line
        +int end_line
    }

    class QuestionAnswer {
        +string question_id
        +string question_text
        +string answer
        +list~string~ evidence_turn_ids
        +list~QuoteCitation~ citations
        +string confidence
    }

    class GroundTruthMatrix {
        +list~TranscriptAnalysis~ analyses
        +string generated_at
        +string schema_version
        +get_answer(expert, question_id)
        +get_column(expert)
        +get_row(question_id)
    }

    class SynthesisReport {
        +list~ConsensusTheme~ consensus_themes
        +list~DisagreementPoint~ disagreements
        +MarketGrowthSpectrum market_growth_spectrum
        +string generated_at
    }

    class RAGResponse {
        +string query
        +string answer
        +list~QuoteCitation~ citations
        +list~RetrievalResult~ retrieved_chunks
        +EvidenceGateResult evidence_gate
        +bool abstained
    }

    TranscriptDocument "1" *-- "many" DialogueTurn : contains
    GroundTruthMatrix "1" *-- "many" QuestionAnswer : stores 6xN answers
    QuestionAnswer "1" *-- "many" QuoteCitation : backed by verified
    DialogueTurn "1" ..> "1" QuoteCitation : exact coordinates
    SynthesisReport ..> GroundTruthMatrix : synthesizes
    RAGResponse "1" *-- "many" QuoteCitation : cites
```

---

## 4. Evidence Contract & Closed-Loop Verification

The LLM does not define the final citation text. It returns `evidence_turn_ids`; the application resolves those IDs against the canonical `TranscriptDocument` and extracts the exact source quote, timestamp, and line/character provenance:

1. **`EXACT_VERIFIED` (Score: 1.00)**: Verbatim substring match `raw_text[start:end] == quote`.
2. **`NORMALIZED_VERIFIED` (Score: 0.95)**: Normalized match across smart quotes, dashes, whitespace, and punctuation.
3. **`FALLBACK_EVIDENCE` (Score: 0.85)**: Deterministic fuzzy recovery via `thefuzz` / Levenshtein distance.
4. **`REJECTED` (Score: 0.00)**: Candidate citation hallucinated or unsupported; rejected from verified citations.

---

## 5. Five-Screen Interactive Streamlit Application

1. **MATRIX** — Dynamic $6 \times 3$ comparative question matrix. Cached answers load with zero token spend; timestamp buttons open the Evidence Drawer.
2. **SYNTHESIS** — Shared Consensus Themes, Strategic Disagreements with stakeholder positions, and the 3-Year Market Growth Spectrum (Conservative, Moderate, Bullish).
3. **AI CHAT** — Hybrid RAG with Market/Role filtering, progressive streaming, evidence gate abstention, and expandable citation cards with 1-click line jump.
4. **INSPECTOR** — Complete line-numbered transcript viewer with speaker styling, `.line-target` highlighting on citation jump, and auto-scroll.
5. **UPLOAD** — Drag-and-drop `.txt` ingestion with metadata/timestamp validation, SHA-256 idempotency, and dynamic session rebuild (matrix extraction $\to$ synthesis $\to$ in-memory RAG index).

---

## 6. Main Codebase Structure

```text
src/
├── config.py               # Pydantic settings and environment management
├── models/
│   ├── transcript.py       # TranscriptDocument, Turn, SpeakerType, Metadata
│   ├── analysis.py         # QuestionAnswer, QuoteCitation, TranscriptAnalysis, GroundTruthMatrix
│   ├── synthesis.py        # ConsensusTheme, DisagreementPoint, MarketGrowthSpectrum, SynthesisReport
│   └── retrieval.py        # Chunk, RetrievalResult, EvidenceGateResult, RAGResponse
├── core/
│   ├── parser.py           # Position-preserving regex parser with SHA-256 idempotency
│   ├── quote_verifier.py   # Closed-loop deterministic citation verifier
│   ├── extractor.py        # 6-Question structured extractor with prompt versioning
│   ├── synthesizer.py      # Cross-expert synthesis engine
│   ├── chunker.py          # Turn-level provenance-preserving chunker
│   ├── embedder.py         # Sentence-Transformers (all-MiniLM-L6-v2) + deterministic hash embedder
│   ├── retriever.py        # BM25 + Vector + Reciprocal Rank Fusion (k=60) + Evidence Gate
│   └── rag_engine.py       # Conversational RAG engine with abstention guardrails
app/
└── streamlit_app.py        # 5-Screen interactive dashboard & Evidence Inspector
eval/
├── eval_dataset.json       # Canonical 8-case benchmark dataset
├── eval.py                 # Evaluation benchmark runner
└── evaluation_report.json  # Machine-readable evaluation report
scripts/
├── build_ground_truth.py   # CLI extraction utility
├── build_synthesis.py      # CLI cross-expert synthesis utility
└── build_rag_index.py      # CLI persistent Chroma indexing utility
tests/
├── test_parser.py          # 8 tests (42/42 turn offset invariant, speaker classification, SHA-256)
├── test_quote_verifier.py  # 7 tests (exact, normalized, multi-citation, hallucinated rejection)
├── test_extractor.py       # 5 tests (18-cell grid, cache invalidation, candidate turns)
├── test_synthesizer.py     # 6 tests (consensus themes, disagreements, growth spectrum)
├── test_rag.py             # 8 tests (turn chunking, hybrid retrieval, evidence gating, abstention)
├── test_phase6.py          # 2 tests (UI source availability, transcript counts)
└── test_phase7.py          # 2 tests (evaluation harness and deterministic offline scoring)
```

---

## 7. Local Setup & Quickstart

### Prerequisites
- Python 3.11+
- Windows PowerShell, macOS, or Linux

### 1. Create and Activate Virtual Environment

**Windows PowerShell**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
*(Optional: Provide `OPENAI_API_KEY` for live GPT-4o extractions; system defaults to deterministic offline mock/hash paths without credentials).*

---

## 8. Verification Commands

### Run Full Automated Test Suite (38 Tests)
```bash
pytest -v
```

### Run Evaluation Harness Benchmark
```bash
python eval/eval.py
```

### Launch Interactive Streamlit Pro Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 9. Deep Dive Documentation & Architecture

For additional technical blueprints, benchmarks, and scaling discussions:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Comprehensive System Architecture & Deep Pipeline Analysis
- [docs/SCALABILITY.md](docs/SCALABILITY.md) — Production Scale & Latency Optimization
- [docs/BENCHMARK_AND_REFERENCES.md](docs/BENCHMARK_AND_REFERENCES.md) — Literature & Industry Benchmark Grounding

---

## 10. Interview Demonstration & Presentation

See [`docs/INTERVIEW_DEMO_SCRIPT.md`](docs/INTERVIEW_DEMO_SCRIPT.md) for a complete 5–10 minute live walkthrough guide with minute-by-minute talking points, UI navigation steps, and production scaling answers.
