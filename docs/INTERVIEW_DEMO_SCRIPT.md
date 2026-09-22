# Hasamex Interview Demo Script — 5 to 10 Minutes

## 0:00–0:45 — What the system does

“This application turns three expert transcripts into an audit-ready commercial diligence workspace. The key design principle is closed-loop grounding: the model can propose evidence IDs, but the source transcript remains the authority for the final quote and timestamp.”

## 0:45–1:30 — Architecture

Show the repository tree and explain:

```text
Raw transcripts
   ↓
Deterministic parser + provenance
   ↓
Turn-level chunks
   ↓
BM25 + vector retrieval
   ↓
RRF fusion (k=60)
   ↓
Evidence gate
   ↓
LLM answer + evidence turn IDs
   ↓
Deterministic citation resolution
   ↓
Streamlit dashboard
```

Mention that ChromaDB is available as the persistent vector-store implementation, while the offline tests use the deterministic hash embedding fallback when semantic-model dependencies are unavailable.

## 1:30–3:00 — Matrix and evidence

Open **MATRIX**.

- Show Q1–Q6 across France, Germany, and the UK.
- Open a timestamp citation (e.g. `[02:18]`).
- Show the exact source quote in the Evidence Drawer.
- Click **Jump to Line in Inspector**.
- Explain that the quote is resolved from the stored transcript turn rather than copied from the model output.

## 3:00–4:15 — Synthesis

Open **SYNTHESIS**.

Explain that consensus and disagreement cards are generated from the structured expert matrix. Point out that the three experts represent different stakeholder perspectives, so differences are retained rather than averaged away.

## 4:15–6:00 — Conversational RAG

Open **AI CHAT**.

Ask a grounded question such as:

> What does procurement in Germany look at when evaluating a robotic surgery system?

Show the answer and citation cards.

Then ask:

> What is the robotic surgery market size in Japan?

Explain that the system should abstain because Japan is outside the evidence corpus.

## 6:00–7:15 — Upload / ingestion

Open **UPLOAD**.

Drop a valid `.txt` transcript and explain:

- UTF-8 parsing
- metadata validation
- timestamp validation
- SHA-256 idempotency
- rebuild of matrix, synthesis, and RAG session state

## 7:15–8:15 — Evaluation

Run:

```bash
python eval/eval.py
```

Show the retrieval, expert-only, citation-resolution, and abstention metrics. State clearly that this is a small assessment benchmark, not a claim of production statistical coverage.

## 8:15–9:00 — Production scaling

Explain the natural scale-up path:

```text
3 transcripts
   ↓
30+ transcripts
   ↓
persistent vector DB / pgvector / Qdrant
   + background ingestion jobs
   + Redis-backed cache/queue
   + FastAPI service boundary
   + richer evaluation set
```
