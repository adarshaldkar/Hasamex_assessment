# Comprehensive Discussion Summary & Architectural Decisions

**Assessment**: Hasamex AI Engineer – Round 2 Technical Case  
**Domain**: European Robotic Surgery Market (Commercial Due Diligence & Primary Research)  
**Submission Deadline**: 23 September 2026, 11:00 AM IST  
**Document Status**: Fully Aligned Discussion Record & Blueprint  

---

## Table of Contents
1. [Assessment Context & Core Challenge](#1-assessment-context--core-challenge)
2. [Input Data Analysis: 3 Experts, 3 Jurisdictions, 6 Questions](#2-input-data-analysis)
3. [The Core Architectural Paradigm: Closed-Loop Document Intelligence](#3-the-core-architectural-paradigm)
4. [Pragmatic Tech Stack Decisions: Anti-Overengineering](#4-pragmatic-tech-stack-decisions)
5. [The 20 Critical Engineering Safeguards & Refinements](#5-the-20-critical-engineering-safeguards--refinements)
6. [The 5-Screen User Interface Experience](#6-the-5-screen-user-interface-experience)
7. [The 7–8 Minute PiP Video Strategy & Mandatory AI Disclosure](#7-the-video-strategy--mandatory-ai-disclosure)
8. [Scalability Narrative: 3 → 30 → 300+ Transcripts](#8-scalability-narrative)
9. [Project Directory & Module Mapping](#9-project-directory--module-mapping)
10. [Step-by-Step Implementation Execution Plan](#10-step-by-step-implementation-execution-plan)

---

## 1. Assessment Context & Core Challenge

### 1.1 What the Case Study Is
This assessment simulates a **Qualitative Due Diligence & Market Intelligence Platform** used in management consulting (McKinsey, BCG, Bain), private equity diligence, and expert networks (Tegus, AlphaSights, GLG). 

Analysts conduct 60-minute in-depth interviews with Key Opinion Leaders (KOLs), hospital executives, and procurement directors. The engineering challenge is to transform raw timestamped conversational transcripts into **audit-ready, verified structured intelligence** without hallucinations.

### 1.2 The 3 Mandatory Submission Deliverables
1. **Working Application**: An interactive, runnable application fulfilling all case requirements.
2. **Clean Source Code & README**: Pristine Git repository with `.env.example`, automated tests, and clear run instructions.
3. **5–10 Minute PiP Video**: A webcam-visible screen recording demonstrating the solution, explaining architecture and tech stack, disclosing AI usage, addressing challenges, and showcasing live execution.

---

## 2. Input Data Analysis

### 2.1 The 3 Expert Call Transcripts

| Transcript | Expert Name | Role | Market | Core Perspective & Findings |
| :--- | :--- | :--- | :--- | :--- |
| **Transcript 1** | Dr. Jean Martin | Head of Urology | **France** | Concentrated in large academic & private centres; capital budget approval is the main hurdle; finance team requires utilization and payback modeling; training multiple surgeons in year one is vital; expects steady 15–20% growth in top centres; 6–12 month purchase cycle. |
| **Transcript 2** | Anna Keller | Former Hospital Procurement Director | **Germany** | Uneven adoption; large university hospitals lead while regional wait; capital cost & business case proving utilization are primary barriers; procurement focuses on Total Cost of Ownership (TCO) and service contracts; single-surgeon utilization destroys ROI; conservative high-single/low-double digit growth; 9–18 month purchase cycle. |
| **Transcript 3** | Dr. Emily Carter | Consultant Urologist | **UK (NHS)** | Selected NHS trusts standardizing; funding is critical but training capacity for surgeons and theatre staff is equally crucial; ROI balanced with clinical strategy, length of stay, and recruitment; bullish outlook (>15% growth if training expands and costs fall); 6–9 month timeline if pre-funded. |

### 2.2 The 6 Standardized Interview Guide Questions
1. *How would you describe current adoption of robotic surgery in your market?*
2. *What are the main barriers to adoption?*
3. *How important are hospital budgets and ROI in purchasing decisions?*
4. *How important are surgeon training and clinical outcomes?*
5. *What adoption trend do you expect over the next 3–5 years?*
6. *What is the typical hospital decision-making timeline for purchasing a new robotic system?*

---

## 3. The Core Architectural Paradigm: Closed-Loop Document Intelligence

### 3.1 Why Naive RAG Fails
Standard open-loop RAG follows: `Query → Vector Search → Chunks → LLM → Output`.
In commercial due diligence, this fails because:
* LLMs can subtly mutate quotes, misattribute numbers, or invent timestamps.
* Non-deterministic repeated extraction on page loads creates latency and token waste.
* Weak semantic search fails on domain acronyms like `NHS`, `TCO`, `Urology`.

### 3.2 Our Closed-Loop Design
```text
Raw Transcript (.txt)
        ↓
Structured Parser (State machine, timestamps, speakers, byte offsets)
        ↓
Precomputed 6-Question Extraction (Strict Pydantic JSON schema)
        ↓
[DETERMINISTIC VERIFICATION GUARDRAIL]
   ├── 1. Timestamp exists in transcript?
   ├── 2. Turn belongs to cited expert?
   └── 3. Quote exists verbatim in turn content?
        ↓
Verified Ground-Truth Matrix (Cached JSON, 0ms dashboard load)
        ↓
Two-Stage Cross-Expert Synthesis (Consensus & Divergence Engine)
        ↓
Hybrid Retrieval RAG (BM25 + Vectors + RRF) with Answer Abstention
```

---

## 4. Pragmatic Tech Stack Decisions: Anti-Overengineering

We explicitly evaluated and eliminated unnecessary infrastructure overhead for version 1:

| Component | Choice | Rejection of Alternatives & Interview Rationale |
| :--- | :--- | :--- |
| **Runtime & Core** | Python 3.11+ | Industry standard for AI engineering, Pydantic, and async inference. |
| **Backend API** | FastAPI + Pydantic v2 | High-performance async REST endpoints, automated Swagger documentation, Server-Sent Events (SSE) for streaming. |
| **Vector Store** | **ChromaDB (Local)** | *No Postgres/pgvector or Qdrant for v1.* ChromaDB runs locally with zero setup friction for reviewers. We discuss pgvector/Qdrant as the natural upgrade at 300+ scale. |
| **Lexical Search** | **Rank-BM25** | Vital for domain acronyms (`NHS`, `TCO`, `Urology`, `6–12 months`). |
| **Fusion Algorithm**| **Reciprocal Rank Fusion (RRF)** | Merges BM25 and vector rankings mathematically ($k=60$). |
| **LLM Provider** | Direct Official SDK (OpenAI / Gemini) | *No OpenRouter or Groq.* Direct SDKs eliminate third-party hops and give direct access to native structured JSON outputs and streaming. |
| **Temperature** | `temperature = 0.0` | Eliminates extraction drift; ensures deterministic, audit-ready extraction. |
| **Frontend UI** | **Streamlit Pro** | Pure Python, zero Node.js/npm version issues for the evaluator. Delivers all 5 screens in a cohesive, interactive dark dashboard. |
| **Worker Queue** | Synchronous Ingestion | *No Redis/Celery for v1.* 3 transcripts take < 5 seconds to process; background workers are discussed for 30+ project scale. |

---

## 5. The 20 Critical Engineering Safeguards & Refinements

1. **Speaker-State Parser**: Handles multi-line dialogue without assuming every newline has a speaker tag. Tracks `current_timestamp`, `current_speaker`, and text buffer.
2. **Immutable Raw Data**: Raw transcripts in `data/` are strictly read-only. Processed artifacts reside in `storage/processed/`.
3. **Deterministic Source IDs**: Stable hierarchical keys: `transcript_id` (`france_001`), `turn_id` (`france_001_turn_04`), and `chunk_id`.
4. **Citations as Typed Data Objects**: Citations are structured payloads (`transcript_id`, `turn_id`, `timestamp`, `quote`, `start_char`, `end_char`), not unstructured strings.
5. **2-Factor Verification (Quote + Timestamp + Speaker)**: Confirms the timestamp exists, belongs to the cited speaker, and the quote exists verbatim in that turn.
6. **Principled Answer Abstention**: If retrieved evidence is below similarity/lexical thresholds, the system explicitly returns *"Insufficient evidence in transcripts to answer this question"* rather than hallucinating.
7. **Evidence-Based Confidence Scoring**: Objective ratings: **HIGH** (exact verbatim quote + direct timestamp match), **MEDIUM** (multi-turn synthesis required), **LOW** (inferred claim).
8. **Stance Spectrum**: Visual synthesis spectrum for market growth: Conservative (Germany) $\leftrightarrow$ Moderate (France) $\leftrightarrow$ Bullish (UK).
9. **Role-Aware Analysis**: Differentiates Clinicians (prioritizing patient outcomes, training, and length of stay) from Procurement Directors (prioritizing TCO and maintenance contracts).
10. **Question-Aware Tagging**: Dialogue turns and extractions are mapped to `question_id` (`Q1`–`Q6`) for instant matrix assembly.
11. **Chunk Deduplication**: Removes overlapping text across top RRF retrieved chunks before prompt assembly.
12. **Automated Evaluation Set**: Tests verifying question answering accuracy, quote validity, and timestamp recall.
13. **Observability & Logging**: Tracks `request_id`, query string, latency, token usage, and verification pass/fail rates.
14. **Graceful Error Handling**: User-facing informative errors instead of raw Python tracebacks.
15. **Upload Validation**: Enforces `.txt` format, size limits, header syntax, and timestamp regex integrity.
16. **Idempotent Ingestion via SHA-256**: Prevents duplicate embeddings or redundant LLM calls on re-upload.
17. **Prompt Versioning**: Prompts stored in isolated files with explicit versions (`extraction_v1.py`, `rag_v1.py`, `synthesis_v1.py`).
18. **Model Provider Abstraction**: Base `LLMProvider` class with swappable adapters (`GeminiProvider`, `OpenAIProvider`).
19. **Strict Extraction vs. Chat Separation**: Precomputed ground-truth matrix for fixed questions; hybrid RAG only for ad-hoc chat.
20. **Security Hygiene**: `.env` git-ignored, path sanitization, and no arbitrary file execution.

---

## 6. The 5-Screen User Interface Experience

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        TRANSCRIPT INTELLIGENCE                         │
├───────────────┬───────────────┬───────────────┬───────────┬────────────┤
│  1. MATRIX    │ 2. SYNTHESIS  │  3. AI CHAT   │4.INSPECTOR│ 5. UPLOAD  │
└───────────────┴───────────────┴───────────────┴───────────┴────────────┘
```

1. **Screen 1: Comparative Matrix**:
   - $6 \times 3$ interactive grid (Adoption, Barriers, ROI, Training, Trend, Timeline).
   - Expandable cells with synthesized answer, verbatim quote, verified green badge, and clickable timestamp pill (`[02:18]`).
2. **Screen 2: Synthesis & Insights**:
   - Executive breakdown: Consensus themes (multi-surgeon requirement, hospital tiering) vs. Disagreements (financial primacy vs. clinical balance, growth velocity).
   - Visual Stance Spectrum (Conservative $\leftrightarrow$ Bullish).
3. **Screen 3: Cross-Transcript Conversational RAG**:
   - Free-form chat querying across all transcripts.
   - Outputs answers with clickable citation cards linked to source turns.
4. **Screen 4: Raw Transcript Inspector**:
   - Full transcript reader. Clicking any citation anywhere in the app jumps directly to that timestamp with line highlighting.
5. **Screen 5: Document Upload**:
   - Drag-and-drop `.txt` file uploader with automated validation, parsing, indexing, and dynamic matrix expansion.

---

## 7. The 7–8 Minute PiP Video Strategy & Mandatory AI Disclosure

### 7.1 Recording Specifications
* **Format**: Picture-in-Picture (PiP) screen recording with webcam face clearly visible at all times.
* **Duration**: Target 7–8 minutes (strict limit: 5–10 minutes).
* **Tool**: OBS Studio or Loom (clean screen + webcam, no editing or fancy effects required).

### 7.2 Timing Breakdown
```text
00:00 – 00:40 | Problem Understanding (Commercial due diligence transcript synthesis)
00:40 – 01:30 | Approach & Architecture (Closed-loop verification pipeline)
01:30 – 02:20 | Tech Stack & Engineering Rationale (Why FastAPI, Pydantic, BM25+RRF, Chroma)
02:20 – 03:00 | MANDATORY AI USAGE DISCLOSURE
03:00 – 04:00 | Key Engineering Challenges (Quote hallucinations, acronym retrieval, scaling)
04:00 – 07:20 | LIVE WORKING DEMO:
              ├── 1. Show the 6x3 Ground-Truth Matrix
              ├── 2. Click a citation [02:18] -> Jump to Inspector line
              ├── 3. Show Consensus vs. Disagreements & Stance Spectrum
              ├── 4. Ask a cross-transcript RAG question
              └── 5. Demonstrate Abstention (Out-of-scope question guardrail)
07:20 – 08:00 | Scalability Roadmap & Final Wrap-Up
```

### 7.3 Mandatory AI Disclosure Script (To Speak at 02:20)
> *"In accordance with the assessment guidelines, I want to fully disclose my use of AI tools. During development, I utilized AI coding assistants for architectural brainstorming, boilerplate generation, and debugging. However, I designed the overall system architecture, implemented the deterministic quote verification algorithm, validated all data schemas, and take full engineering responsibility for the solution presented."*

---

## 8. Scalability Narrative: 3 → 30 → 300+ Transcripts

| Dimension | 3 Transcripts (Case Prototype) | 30 Transcripts (Project Scale) | 300+ Transcripts (Enterprise Scale) |
| :--- | :--- | :--- | :--- |
| **Ingestion** | Synchronous in-memory parsing. | Async Celery / multiprocessing workers. | Distributed Kafka/SQS event bus with auto-scaling workers. |
| **Storage** | Local JSON cache (`storage/processed/`). | PostgreSQL + Redis cache. | Distributed PostgreSQL (RDS/Supabase) + S3 blob store. |
| **Vector DB** | In-memory ChromaDB. | Persistent ChromaDB with metadata filters. | Distributed Qdrant or pgvector cluster with market partitioning. |
| **Retrieval** | In-memory BM25 + Vector + RRF. | Hybrid RRF + Cohere Cross-Encoder Reranker. | Multi-stage: Metadata pre-filter $\rightarrow$ Hybrid RRF $\rightarrow$ Reranker. |
| **Synthesis** | Single-pass prompt over extracted matrix. | Hierarchical Map-Reduce by cohort/country. | GraphRAG / Knowledge Graph + UMAP/HDBSCAN opinion clustering. |
| **Token Cost** | Direct API calls (< $0.05). | Prompt Caching (Anthropic/Gemini) (>80% hit rate). | Fine-tuned SLMs for extraction; frontier model for synthesis. |

---

## 9. Project Directory & Module Mapping

```text
harness-assessment/
│
├── data/                                 # Raw input data (immutable)
│   ├── Interview_Guide.txt
│   ├── README_CASE.md
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   └── Transcript_3_UK.txt
│
├── docs/                                 # Full technical documentation suite
│   ├── ASSESSMENT_ANALYSIS_AND_BLUEPRINT.md
│   ├── ARCHITECTURE.md
│   ├── SCALABILITY.md
│   ├── INTERVIEW_DEMO_SCRIPT.md
│   └── COMPREHENSIVE_DISCUSSION_SUMMARY.md  <-- This complete record
│
├── app/                                  # Presentation & API layer
│   ├── main.py                           # FastAPI entrypoint
│   ├── api.py                            # REST & SSE endpoints
│   └── streamlit_app.py                  # 5-screen interactive dashboard
│
├── src/                                  # Core AI engine
│   ├── config.py                         # Settings & API keys
│   ├── models/                           # Pydantic v2 schemas
│   │   ├── transcript.py                 # DialogueTurn, TranscriptDocument
│   │   ├── analysis.py                   # QuestionAnswer, GroundTruthMatrix, Citation
│   │   └── synthesis.py                  # ConsensusTheme, DisagreementPoint
│   ├── core/                             # Algorithmic logic
│   │   ├── parser.py                     # Position-preserving dialogue parser
│   │   ├── chunker.py                    # Metadata-preserving turn chunker
│   │   ├── embedder.py                   # Dense vector embeddings
│   │   ├── retriever.py                  # Hybrid BM25 + Vector + RRF
│   │   ├── quote_verifier.py             # Deterministic verbatim & fuzzy verifier
│   │   ├── extractor.py                  # 6-question structured extraction
│   │   ├── synthesizer.py                # Cross-expert consensus & disagreement
│   │   └── rag_engine.py                 # Attributed conversational RAG
│   ├── prompts/                          # Versioned prompt templates
│   │   ├── extraction.py
│   │   ├── synthesis.py
│   │   └── rag.py
│   └── utils/                            # Helper functions
│       ├── text_utils.py
│       └── citation_utils.py
│
├── storage/                              # Local persistence
│   ├── chroma/                           # Vector index
│   └── processed/                        # Cached Ground-Truth Matrix
│
├── tests/                                # Automated test suite
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_quote_verifier.py
│   ├── test_extractor.py
│   └── test_rag.py
│
├── requirements.txt                      # Project dependencies
├── .env.example                          # Environment template
└── .gitignore                            # Secrets & cache rules
```

---

## 10. Step-by-Step Implementation Execution Plan

With the deadline approaching on **23 September at 11:00 AM IST**, the implementation order is:

1. **Step 1: Data Models & Parser (`src/models/`, `src/core/parser.py`)**:
   - Define Pydantic schemas preserving character offsets.
   - Implement regex speaker-state parser and write `tests/test_parser.py`.
2. **Step 2: Deterministic Quote Verifier (`src/core/quote_verifier.py`)**:
   - Build the 2-factor verifier (timestamp + speaker + verbatim quote check) and write `tests/test_quote_verifier.py`.
3. **Step 3: Structured Extractor & Matrix Precomputation (`src/core/extractor.py`)**:
   - Extract the 6 interview guide questions per transcript.
   - Run candidate quotes through the verifier and cache the precomputed Ground-Truth Matrix to `storage/processed/ground_truth_matrix.json`.
4. **Step 4: Hybrid RAG & Synthesis Engine (`src/core/retriever.py`, `synthesizer.py`, `rag_engine.py`)**:
   - Build BM25 + ChromaDB + RRF retriever with chunk deduplication.
   - Implement consensus, disagreement, and stance spectrum synthesis.
   - Implement conversational RAG with answer abstention and citation objects.
5. **Step 5: 5-Screen Interactive Streamlit Dashboard (`app/streamlit_app.py`)**:
   - Build Matrix view, Synthesis view, Chat view, Transcript Inspector view (with auto-scroll to line), and Upload view.
6. **Step 6: Final Testing, README & Video Preparation**:
   - Run `pytest tests/` to confirm 100% test pass.
   - Finalize `README.md` with setup instructions.
   - Conduct the 7–8 minute PiP screen recording following the script.
