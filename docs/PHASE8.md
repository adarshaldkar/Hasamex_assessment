# Phase 8 — Documentation, Reproducible Demo & Submission Package

## Objective

Produce a reviewer-friendly project package that explains the architecture, evidence contract, local setup, evaluation workflow, and interview demonstration path without overstating capabilities.

## Deliverables

- `README.md` — project overview, architecture, setup, run commands, and design decisions.
- `docs/INTERVIEW_DEMO_SCRIPT.md` — a 5–10 minute demonstration script aligned with the Hasamex case requirements.
- `docs/walkthrough.md` — concise implementation walkthrough for reviewers.

## Documentation Requirements

The final package should make the following easy to verify:

1. The application reads the three supplied transcripts and interview guide.
2. Transcript parsing preserves exact source offsets, timestamps, speaker type, and SHA-256 provenance.
3. Structured answers retain evidence turn IDs and verified quotes.
4. Cross-expert synthesis is generated from the structured matrix rather than free-form transcript guessing.
5. Conversational RAG uses BM25 + vector retrieval + RRF, followed by an evidence gate and deterministic citation resolution.
6. The UI exposes matrix, synthesis, chat, inspector, and upload workflows.
7. The evaluation harness includes in-scope retrieval tests and an out-of-scope abstention test.

## Reproducibility

The local run commands should work from the repository root after installing `requirements.txt`. API-backed LLM behavior requires `OPENAI_API_KEY`; the repository also contains deterministic mock/fallback paths used by the offline test suite.
