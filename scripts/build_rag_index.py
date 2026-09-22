from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.chunker import chunk_transcripts
from src.core.embedder import HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from src.core.parser import parse_transcript_file
from src.core.retriever import ChromaVectorIndex, HybridRetriever


def main() -> None:
    default_data = "data/raw" if (ROOT / "data" / "raw").exists() else "data"
    parser = argparse.ArgumentParser(description="Build the persistent Chroma vector index for transcript chunks.")
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument("--chroma-dir", default="storage/chroma")
    parser.add_argument("--collection", default="transcript_chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    paths = sorted(data_dir.glob("Transcript_*.txt"))
    if not paths:
        raise SystemExit(f"No transcript files found in {data_dir}")
    transcripts = [parse_transcript_file(path) for path in paths]
    chunks = chunk_transcripts(transcripts)
    try:
        embedder = SentenceTransformerEmbeddingProvider(args.model)
    except Exception:
        embedder = HashEmbeddingProvider()

    vector_index = ChromaVectorIndex(
        chunks,
        embedding_provider=embedder,
        persist_path=str(ROOT / args.chroma_dir),
        collection_name=args.collection,
    )
    retriever = HybridRetriever(
        chunks,
        vector_index=vector_index,
        embedding_provider=embedder,
    )
    print(f"Indexed {len(chunks)} transcript chunks.")
    print(f"Chroma path: {ROOT / args.chroma_dir}")
    print(f"Collection: {args.collection}")
    result = retriever.retrieve("What did procurement say about TCO?")
    for item in result[:5]:
        print(
            f"{item.chunk.market} | {item.chunk.timestamp_start} | "
            f"RRF={item.rrf_score:.4f} fused={item.fused_relevance:.3f} | {item.chunk.text[:100]}"
        )


if __name__ == "__main__":
    main()
