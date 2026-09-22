# Phase 7 — Evaluation Harness & Grounding Benchmark

## Objective

Measure the retrieval and grounding behavior of the transcript intelligence system with a small, deterministic benchmark that runs without paid LLM calls.

## Deliverables

- `eval/eval_dataset.json` — benchmark questions and expected supporting turn IDs.
- `eval/eval.py` — executable evaluation runner.
- `tests/test_phase7.py` — automated checks for evaluation assets and guardrails.
- `eval/evaluation_report.json` — generated report after running the benchmark.

## Metrics

### Retrieval Hit@K
For in-scope questions, the benchmark checks whether at least one expected supporting expert turn appears in the retrieved top-K results.

### Expert-only Retrieval
The RAG retriever should return expert evidence rather than interviewer prompts. This checks the evidence-gate contract at retrieval time.

### Citation Resolution
Returned evidence turn IDs must resolve through the canonical transcript source to deterministic citation objects. The model is never treated as the source of truth for quote text.

### Abstention Accuracy
The benchmark includes an out-of-scope query about Japan. The expected behavior is abstention because the provided corpus contains no Japan market evidence.

## Run

```bash
python eval/eval.py
```

The command prints case-level status and writes:

```text
eval/evaluation_report.json
```

Run the complete automated suite with:

```bash
pytest -v
```

## Interpretation

This benchmark is intentionally small because the technical case contains only three transcripts. It demonstrates the evaluation structure that should scale to a larger annotated set. It does not claim that eight cases are statistically representative of production behavior.

For a production system, extend the dataset with manually reviewed question-answer pairs, expected evidence spans, adversarial paraphrases, and known abstention cases. Track retrieval recall, citation precision, grounded-answer rate, latency, and failure categories across model/provider versions.
