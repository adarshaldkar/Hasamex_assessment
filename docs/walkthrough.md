# Implementation Walkthrough

## Source of truth

`data/` contains the input transcripts. Parsed `TranscriptDocument` objects retain SHA-256, turn IDs, timestamps, speakers, roles, markets, and exact character/line offsets.

## Structured analysis

The extraction layer produces one analysis per expert for each interview-guide question. Each answer stores evidence turn IDs and verified `QuoteCitation` objects.

## Synthesis

The synthesis layer consumes the structured matrix and produces consensus/disagreement/growth-spectrum artifacts under `storage/processed/`.

## RAG

The retrieval layer creates turn-level chunks and combines BM25 and vector ranking with Reciprocal Rank Fusion. The evidence gate rejects weak or unsupported evidence. `RAGEngine` returns evidence turn IDs; citations are resolved against canonical transcript turns.

## UI

`app/streamlit_app.py` exposes five workflows: matrix, synthesis, chat, inspector, and upload. Uploaded documents are parsed and incorporated into the current Streamlit session.

## Evaluation

`eval/eval.py` runs a deterministic offline benchmark covering retrieval, expert-only evidence, citation resolution, and an explicit out-of-scope abstention case.

## Known assessment limitation

The bundled processed matrix may be generated with a mock extraction provider when no API key is configured. Passing structural tests does not prove semantic correctness of every generated answer. A final live-provider run and human review should be completed before presenting generated analytical content as final research output.
