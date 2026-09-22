# Phase 5: Hybrid RAG Engine with Evidence Gate & Abstention

**Document**: `docs/PHASE5.md`  
**Phase Objective**: Implement the conversational cross-transcript retrieval and generation engine, combining dense semantic search (Sentence-Transformers / ChromaDB) and lexical term matching (Rank-BM25) via Reciprocal Rank Fusion (RRF $k=60$). Enforce strict negative constraints via an **Evidence Gate** to trigger honest abstention on out-of-scope or unsupported queries.  
**Deliverable Files**:
1. `docs/PHASE5.md` (This Specification)
2. `src/models/retrieval.py` (Chunk, RetrievalResult, EvidenceGateResult, RAGResponse)
3. `src/prompts/rag.py` (Versioned RAG Prompt with Negative Constraints)
4. `src/core/chunker.py` (Logical Dialogue-Turn Chunker)
5. `src/core/embedder.py` (Embedding Provider: `all-MiniLM-L6-v2` + Offline Hash Fallback)
6. `src/core/retriever.py` (Hybrid BM25 + ChromaDB + RRF $k=60$ + Evidence Gate)
7. `src/core/rag_engine.py` (Attributed Q&A Engine with Citation Resolution & Abstention)
8. `tests/test_rag.py` (Automated RAG Unit Test Suite)

---

## 1. Architectural Role of Phase 5

Phase 5 delivers the **free-form ad-hoc exploration engine** of the platform.

While the precomputed Ground-Truth Matrix answers the 6 standard interview guide questions with zero latency, diligence analysts frequently need to ask custom questions (e.g., *"What did Dr. Carter say about theatre staff?"* or *"Who raised concerns about Total Cost of Ownership?"*).

```mermaid
flowchart TD
    QUERY[User Query] --> ALIAS[Lexical Query Expander<br>TCO -> Total Cost of Ownership]
    ALIAS --> BM25[BM25 Lexical Search]
    ALIAS --> VEC[ChromaDB / Vector Search]
    BM25 & VEC --> RRF[Reciprocal Rank Fusion RRF k=60]
    RRF --> DEDUP[Deduplicated & Filtered Top Chunks]
    DEDUP --> GATE{Evidence Gate<br>Score >= 0.50 & Expert Turn?}
    
    GATE -- Pass --> LLM[LLM Generation with Turn IDs]
    LLM --> VERIFY[Citation Resolver -> QuoteCitation]
    VERIFY --> ANSWER[Grounded Attributed Answer with [MM:SS] Pills]
    
    GATE -- Fail / Out-of-Scope --> ABSTAIN["Honest Abstention:<br>'Insufficient evidence in transcripts'"]
```

---

## 2. Core Invariants & Engineering Safeguards

1. **Dialogue-Bounded Chunking**: Chunks are never sliced arbitrarily by character/token counts across sentence boundaries. Each chunk is bounded by a complete `DialogueTurn`, preserving `speaker`, `role`, `market`, `start_char`, `end_char`, `start_line`, and `end_line`.
2. **Domain Acronym Expansion**: Acronyms common in hospital procurement (`TCO`, `ROI`, `NHS`) are automatically expanded in query representations without replacing the user's original query.
3. **Reciprocal Rank Fusion (RRF)**:
   $$RRF(d) = \sum_{m \in \{BM25, Vector\}} \frac{1}{60 + rank_m(d)}$$
4. **Evidence Gate**:
   - Rejects interviewer turns as factual evidence (interviewer questions cannot be cited as clinician claims).
   - Computes fused score (70% normalized RRF + 30% lexical token overlap). If below threshold (0.50), the system refuses to invoke the LLM and triggers honest abstention.
   - Rejects queries containing unsupported proper nouns or acronyms not found anywhere in the corpus text.
5. **Principled Answer Abstention**: When evidence is weak or absent, returns:
   > *"The provided transcripts do not contain sufficient evidence to answer this question."*

---

## 3. Data Models (`src/models/retrieval.py`)

```python
class Chunk(BaseModel):
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
    start_char: int
    end_char: int
    start_line: int
    end_line: int

class RetrievalResult(BaseModel):
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
    allowed: bool
    reason: str
    evidence: list[RetrievalResult] = Field(default_factory=list)

class RAGResponse(BaseModel):
    answer: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    abstained: bool = False
```

---

## 4. Automated Test Specifications (`tests/test_rag.py`)

1. `test_chunker_preserves_turn_provenance`: Asserts chunk character and line boundaries match source dialogue turns.
2. `test_hybrid_retrieval_finds_tco_expert_turn`: Asserts querying *"TCO and service contracts"* ranks Anna Keller's German turn top.
3. `test_evidence_gate_rejects_empty_or_weak_results`: Asserts evidence gate stops low-relevance candidates.
4. `test_evidence_gate_ignores_interviewer_turns`: Asserts interviewer turns are filtered out of evidence candidates.
5. `test_rag_engine_returns_grounded_mock_answer`: Asserts valid queries return grounded answers with valid turn IDs.
6. `test_rag_engine_abstains_out_of_scope`: Asserts out-of-scope query (*"What is the market share in Japan?"*) triggers honest abstention with `abstained=True`.
7. `test_market_and_role_filters`: Asserts metadata filtering by market (`France`) and role (`Procurement`).
8. `test_rag_citations_resolve_to_source_text`: Asserts RAG response citations resolve to exact verbatim source text slices.
