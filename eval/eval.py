"""Deterministic evaluation harness for the Hasamex transcript RAG pipeline."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.chunker import chunk_transcripts
from src.core.embedder import HashEmbeddingProvider
from src.core.parser import parse_transcript_file
from src.core.rag_engine import MockRAGProvider, RAGEngine
from src.core.retriever import HybridRetriever

RAW_DIR = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")
DATASET_PATH = Path(__file__).resolve().with_name("eval_dataset.json")


@dataclass
class CaseResult:
    case_id: str
    retrieved_turn_ids: list[str]
    expected_turn_ids: list[str]
    hit_at_k: bool
    expert_only: bool
    abstained: bool
    abstention_expected: bool
    citation_resolvable: bool
    answer_nonempty: bool


def load_transcripts():
    specs = [
        ("Transcript_1_France.txt", "france_001"),
        ("Transcript_2_Germany.txt", "germany_001"),
        ("Transcript_3_UK.txt", "uk_001"),
    ]
    return [parse_transcript_file(RAW_DIR / filename, transcript_id) for filename, transcript_id in specs]


def build_engine():
    transcripts = load_transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(
        chunks,
        embedding_provider=HashEmbeddingProvider(),
        top_k_each=8,
        rrf_k=60,
    )
    engine = RAGEngine(
        retriever=retriever,
        transcripts=transcripts,
        provider=MockRAGProvider(),
        evidence_threshold=0.50,
        top_k=4,
        max_evidence=4,
    )
    return engine


def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def evaluate_case(engine: RAGEngine, case: dict[str, Any]) -> CaseResult:
    response = engine.answer(
        case["query"],
        market=case.get("market"),
    )
    retrieved_turn_ids = [item.chunk.turn_id for item in response.retrieved_chunks]
    expected_turn_ids = list(case.get("expected_turn_ids", []))
    expected_abstention = bool(case.get("expect_abstention", False))

    hit_at_k = any(turn_id in retrieved_turn_ids for turn_id in expected_turn_ids) if expected_turn_ids else False
    expert_only = all(
        item.chunk.speaker_type.casefold() == "expert"
        for item in response.retrieved_chunks
    )

    citation_resolvable = True
    if response.evidence_turn_ids:
        try:
            citations = engine.citations_for_response(response)
            citation_resolvable = len(citations) >= 1 and all(
                citation.turn_id in response.evidence_turn_ids for citation in citations
            )
        except (KeyError, ValueError):
            citation_resolvable = False

    return CaseResult(
        case_id=case["id"],
        retrieved_turn_ids=retrieved_turn_ids,
        expected_turn_ids=expected_turn_ids,
        hit_at_k=hit_at_k,
        expert_only=expert_only,
        abstained=response.abstained,
        abstention_expected=expected_abstention,
        citation_resolvable=citation_resolvable,
        answer_nonempty=bool(response.answer.strip()),
    )


def run_evaluation() -> dict[str, Any]:
    dataset = load_dataset()
    engine = build_engine()
    results = [evaluate_case(engine, case) for case in dataset["cases"]]

    retrieval_cases = [result for result in results if result.expected_turn_ids]
    abstention_cases = [result for result in results if result.abstention_expected]

    retrieval_hit_rate = (
        sum(result.hit_at_k for result in retrieval_cases) / len(retrieval_cases)
        if retrieval_cases else 0.0
    )
    expert_only_rate = (
        sum(result.expert_only for result in results) / len(results)
        if results else 0.0
    )
    citation_resolution_rate = (
        sum(result.citation_resolvable for result in results) / len(results)
        if results else 0.0
    )
    abstention_accuracy = (
        sum(result.abstained == result.abstention_expected for result in abstention_cases) / len(abstention_cases)
        if abstention_cases else 1.0
    )

    return {
        "dataset_version": dataset.get("version", "unknown"),
        "metrics": {
            "retrieval_hit_at_k": round(retrieval_hit_rate, 4),
            "expert_only_rate": round(expert_only_rate, 4),
            "citation_resolution_rate": round(citation_resolution_rate, 4),
            "abstention_accuracy": round(abstention_accuracy, 4),
        },
        "cases": [asdict(result) for result in results],
    }


def main() -> int:
    report = run_evaluation()
    print("Hasamex RAG Evaluation")
    print("=======================")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.2%}")
    print()
    for case in report["cases"]:
        status = "PASS"
        if case["expected_turn_ids"] and not case["hit_at_k"]:
            status = "FAIL"
        if case["abstention_expected"] and not case["abstained"]:
            status = "FAIL"
        if not case["citation_resolvable"]:
            status = "FAIL"
        print(f"[{status}] {case['case_id']}")
        print(f"  retrieved: {', '.join(case['retrieved_turn_ids']) or 'none'}")
        print(f"  expected:  {', '.join(case['expected_turn_ids']) or 'none'}")

    report_path = Path(__file__).resolve().with_name("evaluation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_path}")

    failed = [
        case for case in report["cases"]
        if (case["expected_turn_ids"] and not case["hit_at_k"])
        or (case["abstention_expected"] and not case["abstained"])
        or not case["citation_resolvable"]
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
