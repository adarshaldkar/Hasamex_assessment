# Industry Benchmarks & Reference Systems

A curated analysis of open-source architectures and commercial platforms that inform the design of the **Hasamex AI Engineer Transcript Intelligence Platform**.

---

## 1. Direct Open-Source Architectural Benchmarks

### 1. Mosaic — The Multi-Document RAG Standard
* **Repository**: [`abdmaherag/mosaic`](https://github.com/abdmaherag/mosaic)
* **Relevance**: Closest technical architecture to our pipeline.
* **Core Mechanisms to Adopt**:
  - **Hybrid Retrieval**: Rank-BM25 + ChromaDB (HNSW cosine similarity) fused via Reciprocal Rank Fusion (RRF $k=60$).
  - **Multi-Query Expansion**: Generating query variants for broader recall.
  - **Honest Refusal**: Returning explicit abstention when retrieval distance exceeds threshold ($> 0.80$).
  - **Streaming**: Server-Sent Events (SSE) for token and citation streaming.
  - **Evaluation Harness**: `eval/eval.py` calculating `hit@k` and `keyword@k`.

### 2. TubeIQ — Transcript-to-Timestamp Navigation
* **Repository**: [`dhruvWorkss/TubeIQ`](https://github.com/dhruvWorkss/TubeIQ)
* **Relevance**: Demonstrates the exact UX pattern for transcript Q&A with clickable timestamps.
* **Core Mechanisms to Adopt**:
  - Timestamped dialogue chunking.
  - Interactive clickable timestamp pills (`[MM:SS]`) that anchor to specific temporal positions.
  - Streamlit UI simplicity: clean dark mode, responsive tabbed navigation.
  - High-efficiency inference with **Groq Llama 3.3 70B** + **`all-MiniLM-L6-v2`** CPU embeddings.

### 3. Lenny's Podcast Knowledge Base — Multi-Expert Attribution
* **Repository**: [`flowiesb/lenny`](https://github.com/flowiesb/lenny)
* **Relevance**: Multi-interview RAG across 300+ expert interviews.
* **Core Mechanisms to Adopt**:
  - Clean separation of metadata (`expert_name`, `role`, `specialty`, `market`) from raw dialogue content.
  - Expert attribution badges ensuring users know which clinician or procurement lead made each statement.

### 4. IBM Qux360 — Validation-First Qualitative Analysis
* **Repository**: [`IBM/qux360`](https://github.com/IBM/qux360)
* **Relevance**: Philosophical foundation for trustworthy AI qualitative analysis.
* **Core Mechanisms to Adopt**:
  - First-class validation states: `VERIFIED`, `CHECK`, `REJECTED`.
  - Scrutinizable evidence trail: every claim links back to immutable raw interview text.

### 5. VideoRAG — Production Architecture & Evidence Gating
* **Repository**: [`moneshrallapalli/multimodal-video-rag`](https://github.com/moneshrallapalli/multimodal-video-rag)
* **Relevance**: Architectural reference for enterprise-scale retrieval.
* **Core Mechanisms to Adopt**:
  - Evidence gating: rejecting weak retrieval candidates before invoking the LLM.
  - Scalability blueprint for asynchronous processing and reranking.

### 6. Interview Analysis — Auditable Evidence Modeling
* **Repository**: [`DennisSchulmeister/interview-analysis`](https://github.com/DennisSchulmeister/interview-analysis)
* **Relevance**: Data schema modeling for qualitative transcripts.
* **Core Mechanisms to Adopt**:
  - Stable turn identifiers: `transcript_id`, `turn_id`, `chunk_id`.
  - Preserving `start_char` and `end_char` index offsets for 100% provenance.

### 7. Poverty Action — Qualitative Theme Coding
* **Repository**: [`PovertyAction/llm-quali-coding`](https://github.com/PovertyAction/llm-quali-coding)
* **Relevance**: Grounded thematic extraction and qualitative inductive synthesis.
* **Core Mechanisms to Adopt**:
  - Separating structured code extraction from cross-interview thematic synthesis.

---

## 2. Commercial Diligence Platforms (Product Inspiration)

### 8. Tegus & AlphaSense Expert Transcript Library
* **Product**: Commercial Primary Research & Expert Call Diligence.
* **Core Workflow to Model**:
  $$\text{Search} \longrightarrow \text{Cited Synthesis} \longrightarrow \text{Perspective Comparison Matrix} \longrightarrow \text{Direct Transcript Jump}$$
* **Takeaway**: Diligence analysts do not want a chatty bot; they want an **audit-ready comparison grid** with instant access to the raw call transcript.

### 9. Dovetail — Qualitative Research & Transcript Workspace
* **Product**: Modern Qualitative Research Workspace.
* **Core UX to Model**:
  - Two-pane layout: High-level insight cards on the left; scrollable raw transcript on the right with animated line highlighting on citation click.

---

## 3. The Hasamex Synthesis: Taking It Beyond Existing Systems

| Dimension | Normal RAG (Mosaic / TubeIQ) | **Hasamex AI Engineer Platform** |
| :--- | :--- | :--- |
| **Pipeline Flow** | Retrieve $\rightarrow$ LLM $\rightarrow$ Answer (Trusting LLM) | **Retrieve $\rightarrow$ LLM $\rightarrow$ Deterministic Verification $\rightarrow$ Verified Answer $\rightarrow$ Jump** |
| **Quote Reliability** | May generate hallucinations or subtle mutations | **Algorithmic check (`quote in turn.content`) guarantees verbatim fidelity** |
| **Timestamp Grounding** | Trusts generated string | **2-factor validation (Timestamp exists AND matches speaker and quote)** |
| **Matrix Generation** | Ad-hoc query on every load | **Precomputed Ground-Truth Matrix ($6 \times 3$) with 0ms load time** |
| **Synthesis Layer** | Generic summarization | **Consensus vs. Disagreements + Visual Stance Spectrum (Bullish $\leftrightarrow$ Conservative)** |
| **Proven Auditability**| None | **`start_char` and `end_char` offset tracking enables 1-click jump to raw transcript** |

---

## 4. Key Interview Talking Points

1. *"My design synthesizes the hybrid BM25 + Vector RRF retrieval seen in Mosaic with the timestamp-grounded navigation seen in TubeIQ."*
2. *"However, public RAG implementations suffer from a critical flaw: they trust the LLM's generated citations. In commercial due diligence (like Tegus or AlphaSense), misattribution can invalidate a multi-million-dollar thesis. I introduced a deterministic verification layer (inspired by IBM Qux360) that algorithmically confirms the quote and timestamp exist verbatim in the source file before rendering."*
3. *"The UI interaction borrows from Dovetail and Tegus: an executive comparative matrix paired with a raw transcript inspector that highlights the exact dialogue turn upon clicking any citation."*
