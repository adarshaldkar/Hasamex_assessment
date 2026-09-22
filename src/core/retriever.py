from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.core.embedder import EmbeddingProvider, HashEmbeddingProvider
from src.models.retrieval import Chunk, EvidenceGateResult, RetrievalResult

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-–][A-Za-z0-9]+)*")


@dataclass
class _RankedItem:
    chunk: Chunk
    rank: int
    score: float


class BM25Index:
    """Rank-BM25 wrapper with a standard-library fallback for offline use."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._rank_bm25 = None
        self._tokenized: list[list[str]] = [tokenize(chunk.text) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._rank_bm25 = BM25Okapi(self._tokenized)
        except ImportError:
            self._rank_bm25 = None

    def search(self, query: str, top_k: int) -> list[_RankedItem]:
        if not self.chunks:
            return []
        if self._rank_bm25 is not None:
            scores = self._rank_bm25.get_scores(tokenize(query))
        else:
            scores = [simple_bm25_score(query, tokens, self._tokenized) for tokens in self._tokenized]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[_RankedItem] = []
        for position, index in enumerate(ranked[:top_k], start=1):
            results.append(_RankedItem(self.chunks[index], position, float(scores[index])))
        return results


class InMemoryVectorIndex:
    def __init__(self, chunks: list[Chunk], embedding_provider: EmbeddingProvider | None = None):
        self.chunks = chunks
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.vectors = self.embedding_provider.embed([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int) -> list[_RankedItem]:
        query_vector = self.embedding_provider.embed([query])[0]
        scored = [
            (index, cosine_similarity(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [
            _RankedItem(self.chunks[index], position, float(score))
            for position, (index, score) in enumerate(ranked[:top_k], start=1)
        ]


class ChromaVectorIndex:
    """Persistent ChromaDB vector index using the supplied embedding provider."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        persist_path: str = "storage/chroma",
        collection_name: str = "transcript_chunks",
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use ChromaVectorIndex") from exc

        self.embedding_provider = embedding_provider
        self.chunks = chunks
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._index(chunks)

    def _index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        existing = set(self.collection.get(ids=ids).get("ids", []))
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing]
        if not new_chunks:
            return
        embeddings = self.embedding_provider.embed([chunk.text for chunk in new_chunks])
        self.collection.add(
            ids=[chunk.chunk_id for chunk in new_chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in new_chunks],
            metadatas=[
                {
                    "transcript_id": chunk.transcript_id,
                    "expert_name": chunk.expert_name,
                    "role": chunk.role,
                    "market": chunk.market,
                    "turn_id": chunk.turn_id,
                    "speaker": chunk.speaker,
                    "speaker_type": chunk.speaker_type,
                    "timestamp_start": chunk.timestamp_start,
                    "timestamp_end": chunk.timestamp_end,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
                for chunk in new_chunks
            ],
        )

    def search(self, query: str, top_k: int) -> list[_RankedItem]:
        query_embedding = self.embedding_provider.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        ranked: list[_RankedItem] = []
        for position, (chunk_id, metadata, document, distance) in enumerate(
            zip(ids, metadatas, documents, distances), start=1
        ):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                chunk = Chunk(
                    chunk_id=chunk_id,
                    transcript_id=str(metadata["transcript_id"]),
                    expert_name=str(metadata["expert_name"]),
                    role=str(metadata["role"]),
                    market=str(metadata["market"]),
                    turn_id=str(metadata["turn_id"]),
                    speaker=str(metadata["speaker"]),
                    speaker_type=str(metadata["speaker_type"]),
                    timestamp_start=str(metadata["timestamp_start"]),
                    timestamp_end=str(metadata["timestamp_end"]),
                    text=str(document),
                    start_char=int(metadata["start_char"]),
                    end_char=int(metadata["end_char"]),
                    start_line=int(metadata["start_line"]),
                    end_line=int(metadata["end_line"]),
                )
            ranked.append(_RankedItem(chunk, position, 1.0 - float(distance)))
        return ranked


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def simple_bm25_score(query: str, tokens: list[str], corpus_tokens: list[list[str]]) -> float:
    query_tokens = tokenize(query)
    if not query_tokens or not tokens:
        return 0.0
    n_docs = len(corpus_tokens)
    avgdl = sum(len(doc) for doc in corpus_tokens) / max(1, n_docs)
    frequencies = defaultdict(int)
    for term in tokens:
        frequencies[term] += 1
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term in query_tokens:
        df = sum(1 for doc in corpus_tokens if term in doc)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        tf = frequencies[term]
        denom = tf + k1 * (1 - b + b * len(tokens) / max(1.0, avgdl))
        score += idf * ((tf * (k1 + 1)) / max(1e-9, denom))
    return score


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


DEFAULT_QUERY_ALIASES = {
    "tco": "total cost of ownership",
    "roi": "return on investment payback economics",
    "nhs": "national health service",
    "barriers": "obstacles challenges bottlenecks",
    "growth": "outlook projection percent forecast volume",
}

ANCHOR_ALIASES = {
    "tco": "total cost of ownership",
    "roi": "return on investment payback economics",
    "nhs": "national health service",
    "french": "france",
    "german": "germany",
    "british": "uk united kingdom",
    "english": "uk united kingdom",
    "italian": "italy",
}


def expand_query(query: str, aliases: dict[str, str] | None = None) -> str:
    """Add configured lexical expansions without replacing the user's original query."""
    active = aliases if aliases is not None else DEFAULT_QUERY_ALIASES
    expanded = query
    lower = query.casefold()
    additions: list[str] = []
    for key, value in active.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            additions.append(value)
    if additions:
        expanded = f"{query} {' '.join(additions)}"
    return expanded


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        vector_index: InMemoryVectorIndex | ChromaVectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        top_k_each: int = 8,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.vector_index = vector_index or InMemoryVectorIndex(chunks, self.embedding_provider)
        self.top_k_each = top_k_each
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, top_k: int = 8, market: str | None = None, role: str | None = None, speaker_type: str = "expert") -> list[RetrievalResult]:
        expanded_query = expand_query(query)
        candidate_chunks = [
            chunk for chunk in self.chunks
            if speaker_type is None or chunk.speaker_type.casefold() == speaker_type.casefold()
        ]
        bm25_index = BM25Index(candidate_chunks)
        vector_index = self.vector_index
        if candidate_chunks is not self.chunks:
            vector_index = InMemoryVectorIndex(candidate_chunks, self.embedding_provider)
        bm25_results = bm25_index.search(expanded_query, self.top_k_each)
        vector_results = vector_index.search(expanded_query, self.top_k_each)
        bm25_by_id = {item.chunk.chunk_id: item for item in bm25_results}
        vector_by_id = {item.chunk.chunk_id: item for item in vector_results}
        all_ids = set(bm25_by_id) | set(vector_by_id)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        fused: list[RetrievalResult] = []
        max_possible = 2.0 / (self.rrf_k + 1.0)
        query_tokens = set(tokenize(expanded_query))
        for chunk_id in all_ids:
            chunk = chunks_by_id[chunk_id]
            if market and chunk.market.casefold() != market.casefold():
                continue
            if role and chunk.role.casefold() != role.casefold():
                continue
            bm25_item = bm25_by_id.get(chunk_id)
            vector_item = vector_by_id.get(chunk_id)
            bm25_rank = bm25_item.rank if bm25_item else None
            vector_rank = vector_item.rank if vector_item else None
            bm25_score = bm25_item.score if bm25_item else 0.0
            vector_score = vector_item.score if vector_item else 0.0
            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)
            if vector_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + vector_rank)
            normalized_rrf = min(1.0, rrf_score / max_possible)
            token_set = set(tokenize(chunk.text))
            overlap = len(query_tokens & token_set) / max(1, len(query_tokens))
            fused_relevance = 0.7 * normalized_rrf + 0.3 * overlap
            fused.append(
                RetrievalResult(
                    chunk=chunk,
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    rrf_score=rrf_score,
                    normalized_rrf_score=normalized_rrf,
                    token_overlap=overlap,
                    fused_relevance=fused_relevance,
                )
            )
        fused.sort(key=lambda result: result.fused_relevance, reverse=True)
        return deduplicate_results(fused[:top_k])


def deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    unique: list[RetrievalResult] = []
    for result in results:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        unique.append(result)
    return unique


def evidence_gate(
    retrieved: list[RetrievalResult],
    query: str,
    *,
    threshold: float = 0.50,
    max_evidence: int = 4,
    corpus_text: str | None = None,
) -> EvidenceGateResult:
    expert_results = [
        item for item in retrieved if item.chunk.speaker_type.casefold() == "expert"
    ]
    if not expert_results:
        return EvidenceGateResult(
            allowed=False,
            reason="No expert dialogue turns were retrieved.",
        )

    top_score = expert_results[0].fused_relevance
    if top_score < threshold:
        return EvidenceGateResult(
            allowed=False,
            reason=f"Evidence score {top_score:.3f} is below threshold {threshold:.3f}.",
        )

    # Guard against a semantically high-scoring but clearly out-of-corpus anchor.
    # This stays generic: it looks for acronyms/proper nouns in the query that are
    # not represented anywhere in the corpus, without hard-coding country names.
    corpus = corpus_text or " ".join(f"{item.chunk.text} {item.chunk.market} {item.chunk.expert_name} {item.chunk.role}" for item in expert_results)
    expanded_query = expand_query(query)
    anchor_terms = _unsupported_anchor_terms(query, expanded_query, corpus)
    if anchor_terms:
        return EvidenceGateResult(
            allowed=False,
            reason=f"Unsupported query terms were not found in source evidence: {', '.join(anchor_terms)}.",
        )

    return EvidenceGateResult(
        allowed=True,
        reason=f"Evidence score {top_score:.3f} meets threshold {threshold:.3f}.",
        evidence=expert_results[:max_evidence],
    )


def _unsupported_anchor_terms(original_query: str, expanded_query: str, corpus_text: str) -> list[str]:
    """Find strong query anchors that are absent from the available evidence.

    We only treat all-caps acronyms and non-leading capitalized terms as hard anchors.
    This prevents generic words such as 'What' from triggering abstention.
    """
    original_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", original_query)
    corpus = corpus_text.casefold()
    # Alias expansion is only used to recognize supported acronyms; do not add the
    # original query terms to the corpus evidence check.
    unsupported: list[str] = []
    for index, token in enumerate(original_tokens):
        is_acronym = token.isupper() and len(token) >= 2
        is_proper_noun = index > 0 and token[:1].isupper() and token[1:].islower()
        if not (is_acronym or is_proper_noun):
            continue
        alias_text = ANCHOR_ALIASES.get(token.casefold(), "")
        if token.casefold() not in corpus and not any(part in corpus for part in alias_text.casefold().split() if part):
            unsupported.append(token)
    return unsupported


__all__ = [
    "BM25Index",
    "ChromaVectorIndex",
    "HybridRetriever",
    "InMemoryVectorIndex",
    "cosine_similarity",
    "deduplicate_results",
    "evidence_gate",
    "simple_bm25_score",
    "tokenize",
    "expand_query",
]
