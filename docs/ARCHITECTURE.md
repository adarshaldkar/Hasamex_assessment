# System Architecture: Trustworthy Transcript Intelligence Platform

---

## 1. High-Level Architectural Vision

This system is engineered as an **Enterprise Document Intelligence & Verification Pipeline** specifically designed for Qualitative Commercial Due Diligence and Expert Call Synthesis.

Unlike standard open-loop RAG systems, this platform enforces **closed-loop deterministic verification**: every extracted fact, quote, and timestamp is verified against raw source text before being displayed, cached, or synthesized.

```mermaid
flowchart TD
    subgraph Client ["Client & Presentation Layer"]
        ST[Streamlit Pro UI / React Dashboard]
        API[FastAPI REST & SSE Gateway]
        ST <--> API
    end

    subgraph Parsing ["Stage 1: Position-Preserving Ingestion"]
        TXT[Raw .txt Transcripts] --> PARSER[Regex & Header State Machine]
        PARSER --> TURNS[DialogueTurn Models<br>start_char, end_char, timestamp, speaker]
        TURNS --> CHUNKER[Contextual Chunker]
    end

    subgraph Retrieval ["Stage 2: Hybrid Indexing & Retrieval"]
        CHUNKER --> BM25[BM25 Lexical Index]
        CHUNKER --> VEC[Dense Vector Embeddings]
        BM25 & VEC --> RRF[Reciprocal Rank Fusion RRF]
        RRF --> RAG[RAG Conversational Engine]
    end

    subgraph Verification ["Stage 3: Extraction & Deterministic Verification"]
        TURNS --> EXTRACT[6-Question Structured Extractor]
        EXTRACT --> CANDIDATE[Candidate: Answer + Quote + Timestamp]
        CANDIDATE --> VERIFIER{Deterministic Quote Verifier}
        VERIFIER -- Exact / Fuzzy Match --> ACCEPT[Verified Ground-Truth Matrix]
        VERIFIER -- Mismatch / Hallucination --> FALLBACK[Verbatim Turn Fallback]
        FALLBACK --> ACCEPT
    end

    subgraph Synthesis ["Stage 4: Cross-Expert Synthesis"]
        ACCEPT --> SYNTHESIZER[Consensus vs Disagreement Engine]
        SYNTHESIZER --> THEMES[Shared Themes & Stance Divergences]
    end

    API --> ACCEPT
    API --> THEMES
    API --> RAG
```

---

## 2. Component-by-Component Specifications

### 2.1 Ingestion & Parsing (`src/core/parser.py`)
* **Input**: Raw `.txt` files containing conversational transcripts with header metadata and timestamped dialogue.
* **Header Extraction**: Regex patterns extract `Expert Name`, `Role`, and `Market`.
* **Turn Segmentation**: Regex `^(\d{2}:\d{2})\s*$` identifies timestamp anchors.
* **Offset Tracking**: Every turn records its exact `start_char` and `end_char` index within the original file text, enabling 1-click jumps in the UI.

### 2.2 Contextual Chunking (`src/core/chunker.py`)
* Rather than arbitrary token slicing that splits sentences across speakers, chunks are bounded by **logical dialogue turns**.
* Chunks carry rich metadata:
  ```python
  {
      "chunk_id": "uk_chunk_03",
      "transcript_id": "uk-001",
      "expert_name": "Dr. Emily Carter",
      "market": "United Kingdom",
      "role": "Consultant Urologist",
      "timestamp_start": "01:05",
      "timestamp_end": "02:07",
      "text": "Dr. Carter: Funding is important, but I would say training capacity...",
      "start_char": 420,
      "end_char": 890
  }
  ```

### 2.3 Hybrid Retrieval Engine (`src/core/retriever.py`)
Standard vector search alone struggles with medical and procurement jargon (`TCO`, `NHS trusts`, `Urology`, `6–12 months`).
* **Dense Vector Search**: Captures semantic concepts (e.g., *"financial payback timeframes"*).
* **BM25 Lexical Search**: Captures exact keywords, acronyms, and names.
* **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score(d) = \sum_{m \in \{Dense, BM25\}} \frac{1}{60 + rank_m(d)}$$
  Top chunks from RRF are fed into the conversational RAG engine.

### 2.4 Deterministic Quote Verifier (`src/core/quote_verifier.py`)
This is the core trustworthiness anchor of the entire system:
1. When an answer is extracted, the LLM provides an exact quote and timestamp.
2. The verifier queries the transcript turn at that timestamp.
3. It performs:
   - **Exact Substring Inclusion**: `if quote in turn.content: return (True, quote, 1.0)`
   - **Normalized Match**: Strips whitespace, quotes, and punctuation variations.
   - **Fuzzy Token Matching**: Levenshtein distance ratio $> 0.90$.
4. **Failure Strategy**: If the quote does not exist in the source transcript, it is rejected and replaced with the verbatim dialogue turn text from that timestamp.

### 2.5 Structured Extractor & Ground-Truth Matrix (`src/core/extractor.py`)
* Extracts responses for the 6 fixed interview guide questions per transcript.
* Outputs validated Pydantic models.
* Results are persisted to `storage/processed/ground_truth_matrix.json`.
* **Zero Token Waste**: The matrix is computed once upon ingestion; viewing the dashboard requires zero repeated LLM calls.

### 2.6 Cross-Expert Synthesis Engine (`src/core/synthesizer.py`)
* Operates strictly on top of the verified Ground-Truth Matrix.
* Analyzes:
  1. **Consensus Themes**: Shared operational and clinical patterns (e.g., single-surgeon utilization risk, tiered hospital adoption).
  2. **Disagreement / Divergence Points**: Strategic conflicts (e.g., German procurement's strict TCO focus vs UK NHS's balanced length-of-stay focus; growth forecasts).

### 2.7 Conversational RAG Engine (`src/core/rag_engine.py`)
* Allows free-form interactive questions across all transcripts.
* Enforces strict citation tags: `[Expert Name | MM:SS]`.
* Output citations link directly to character offsets in the Raw Transcript Inspector.

---

## 3. Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Evaluator / Analyst
    participant UI as Streamlit / Web UI
    participant API as FastAPI Gateway
    participant Engine as Extraction & Verification
    participant Storage as Storage (JSON / Chroma)

    Note over User, Storage: Phase 1: Ingestion & Precomputation (Runs Once)
    User->>UI: Upload or Select Transcripts
    UI->>API: POST /api/analyze
    API->>Engine: Parse Turns & Extract Character Offsets
    API->>Engine: Run 6-Question Extraction
    Engine->>Engine: Deterministic Quote & Timestamp Verification
    Engine->>Storage: Save Ground-Truth Matrix & Chroma Embeddings
    Storage-->>API: Ready

    Note over User, Storage: Phase 2: Instant Matrix & Insights Loading
    User->>UI: View Comparative Matrix
    UI->>API: GET /api/matrix
    API->>Storage: Read Cached JSON Matrix
    Storage-->>UI: Instant 0ms Render with Verified Badges

    Note over User, Storage: Phase 3: Conversational Hybrid RAG
    User->>UI: Ask: "What are opinions on surgeon training?"
    UI->>API: POST /api/chat
    API->>Engine: Hybrid BM25 + Vector Search (RRF)
    Engine->>Engine: Generate Attributed Answer with [MM:SS] tags
    Engine-->>UI: Stream Attributed Answer + Clickable Citation Cards
    User->>UI: Click [03:10] Citation Card
    UI->>UI: Jump to Raw Transcript Inspector & Highlight Line
```
