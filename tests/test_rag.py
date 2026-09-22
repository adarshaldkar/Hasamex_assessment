from __future__ import annotations

from pathlib import Path

from src.core.chunker import chunk_transcripts
from src.core.embedder import HashEmbeddingProvider
from src.core.parser import parse_transcript_file
from src.core.quote_verifier import resolve_evidence_turn_ids
from src.core.rag_engine import MockRAGProvider, RAGEngine
from src.core.retriever import HybridRetriever, InMemoryVectorIndex, evidence_gate
from src.models.retrieval import RetrievalResult

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"


def _transcripts():
    paths = sorted(RAW.glob("Transcript_*.txt"))
    return [parse_transcript_file(path, path.stem.lower().replace("transcript_1_", "france_").replace("transcript_2_", "germany_").replace("transcript_3_", "uk_")) for path in paths]


def test_chunker_preserves_turn_provenance():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    assert len(chunks) == 42
    assert all(chunk.turn_id for chunk in chunks)
    assert all(chunk.start_char < chunk.end_char for chunk in chunks)


def test_hybrid_retrieval_finds_tco_expert_turn():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(
        chunks,
        embedding_provider=HashEmbeddingProvider(),
        top_k_each=8,
    )
    results = retriever.retrieve("What did procurement say about TCO?")
    assert results
    assert any(
        result.chunk.market == "Germany"
        and result.chunk.speaker_type == "expert"
        and "cost of ownership" in result.chunk.text.casefold()
        for result in results[:5]
    )


def test_evidence_gate_rejects_empty_or_weak_results():
    gate = evidence_gate([], "What is the market size in Japan?")
    assert gate.allowed is False
    assert gate.evidence == []


def test_evidence_gate_ignores_interviewer_turns():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    results = [
        RetrievalResult(
            chunk=chunk,
            rrf_score=0.03,
            normalized_rrf_score=0.95,
            token_overlap=0.9,
            fused_relevance=0.9,
        )
        for chunk in chunks
        if chunk.speaker_type == "interviewer"
    ][:2]
    gate = evidence_gate(results, "question", threshold=0.5)
    assert gate.allowed is False


def test_rag_engine_returns_grounded_mock_answer():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(chunks, embedding_provider=HashEmbeddingProvider())
    engine = RAGEngine(
        retriever=retriever,
        transcripts=transcripts,
        provider=MockRAGProvider(),
        evidence_threshold=0.50,
    )
    response = engine.answer("What did procurement say about TCO?")
    assert response.abstained is False
    assert response.evidence_turn_ids
    citations = engine.citations_for_response(response)
    assert citations
    assert all(citation.turn_id in response.evidence_turn_ids for citation in citations)


def test_rag_engine_abstains_out_of_scope():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(chunks, embedding_provider=HashEmbeddingProvider())
    engine = RAGEngine(
        retriever=retriever,
        transcripts=transcripts,
        provider=MockRAGProvider(),
        evidence_threshold=0.50,
    )
    response = engine.answer("What is the market size in Japan?")
    assert response.abstained is True
    assert "sufficient evidence" in response.answer


def test_market_and_role_filters():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(chunks, embedding_provider=HashEmbeddingProvider())
    results = retriever.retrieve("ROI and purchasing decisions", market="Germany", role="Former Hospital Procurement Director")
    assert results
    assert all(result.chunk.market == "Germany" for result in results)


def test_rag_citations_resolve_to_source_text():
    transcripts = _transcripts()
    chunks = chunk_transcripts(transcripts)
    retriever = HybridRetriever(chunks, embedding_provider=HashEmbeddingProvider())
    engine = RAGEngine(
        retriever=retriever,
        transcripts=transcripts,
        provider=MockRAGProvider(),
    )
    response = engine.answer("What did procurement say about TCO?")
    citations = engine.citations_for_response(response)
    assert citations
    turns = {turn.turn_id: turn for transcript in transcripts for turn in transcript.turns}
    for citation in citations:
        turn = turns[citation.turn_id]
        assert turn.content == citation.quote
