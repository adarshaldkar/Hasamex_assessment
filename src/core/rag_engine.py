from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from src.core.quote_verifier import resolve_evidence_turn_ids
from src.core.retriever import HybridRetriever, evidence_gate
from src.models.retrieval import EvidenceGateResult, RAGResponse, RetrievalResult
from src.models.transcript import TranscriptDocument
from src.prompts.rag import PROMPT_VERSION, SYSTEM_PROMPT, build_rag_prompt


class LLMRAGExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    evidence_turn_ids: list[str] = Field(default_factory=list)


class RAGProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def answer(self, *, query: str, evidence: list[RetrievalResult]) -> LLMRAGExtraction:
        raise NotImplementedError


class MockRAGProvider(RAGProvider):
    model_name = "mock-rag-v1"

    def answer(self, *, query: str, evidence: list[RetrievalResult]) -> LLMRAGExtraction:
        if not evidence:
            return LLMRAGExtraction(answer="", evidence_turn_ids=[])
        top = evidence[0]
        return LLMRAGExtraction(
            answer=top.chunk.text,
            evidence_turn_ids=[item.chunk.turn_id for item in evidence[:2]],
        )


class OpenAIRAGProvider(RAGProvider):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import os
        from dotenv import load_dotenv

        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        if not self.base_url and self.api_key.startswith("sk-or-v1-"):
            self.base_url = "https://openrouter.ai/api/v1"

        default_model = "openai/gpt-4o-mini" if "openrouter.ai" in self.base_url else "gpt-4o-mini"
        self.model_name = model_name or os.getenv("OPENAI_MODEL", default_model)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIRAGProvider")

    def answer(self, *, query: str, evidence: list[RetrievalResult]) -> LLMRAGExtraction:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the official openai package to use OpenAIRAGProvider") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_rag_prompt(query, evidence)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            return LLMRAGExtraction.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("LLM returned invalid RAG JSON") from exc


class GeminiRAGProvider(RAGProvider):
    def __init__(self, *, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiRAGProvider")

    def answer(self, *, query: str, evidence: list[RetrievalResult]) -> LLMRAGExtraction:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the google-genai package to use GeminiRAGProvider") from exc

        client = genai.Client(api_key=self.api_key)
        prompt = f"{SYSTEM_PROMPT}\n\n{build_rag_prompt(query, evidence)}"
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        content = response.text or "{}"
        try:
            return LLMRAGExtraction.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Gemini returned invalid RAG JSON") from exc


class RAGEngine:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        transcripts: list[TranscriptDocument],
        provider: RAGProvider,
        evidence_threshold: float = 0.50,
        top_k: int = 8,
        max_evidence: int = 4,
    ):
        self.retriever = retriever
        self.transcripts = transcripts
        self.provider = provider
        self.evidence_threshold = evidence_threshold
        self.top_k = top_k
        self.max_evidence = max_evidence
        self._turn_map = {
            turn.turn_id: (transcript, turn)
            for transcript in transcripts
            for turn in transcript.turns
        }

    def answer(
        self,
        query: str,
        *,
        market: str | None = None,
        role: str | None = None,
    ) -> RAGResponse:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k, market=market, role=role)
        corpus_text = " ".join(
            transcript.raw_text for transcript in self.transcripts
        )
        gate = evidence_gate(
            retrieved,
            query,
            threshold=self.evidence_threshold,
            max_evidence=self.max_evidence,
            corpus_text=corpus_text,
        )
        if not gate.allowed:
            return RAGResponse(
                answer="The provided transcripts do not contain sufficient evidence to answer this question.",
                evidence_turn_ids=[],
                retrieved_chunks=retrieved,
                abstained=True,
            )

        result = self.provider.answer(query=query, evidence=gate.evidence)
        ans_clean = result.answer.strip()
        is_refusal = (
            not ans_clean
            or "do not contain sufficient evidence" in ans_clean.lower()
            or "not enough evidence" in ans_clean.lower()
            or "cannot be answered" in ans_clean.lower()
            or "no mention" in ans_clean.lower()
            or "no information" in ans_clean.lower()
        )
        if is_refusal:
            return RAGResponse(
                answer="The provided transcripts do not contain sufficient evidence to answer this question.",
                evidence_turn_ids=[],
                retrieved_chunks=retrieved,
                abstained=True,
            )

        valid_ids = [turn_id for turn_id in result.evidence_turn_ids if turn_id in self._turn_map]
        if not valid_ids:
            valid_ids = [item.chunk.turn_id for item in gate.evidence]
        return RAGResponse(
            answer=ans_clean,
            evidence_turn_ids=valid_ids,
            retrieved_chunks=retrieved,
            abstained=False,
        )

    def citations_for_response(self, response: RAGResponse):
        citations = []
        for turn_id in response.evidence_turn_ids:
            transcript, _ = self._turn_map[turn_id]
            citations.extend(resolve_evidence_turn_ids(transcript, [turn_id]))
        return citations


__all__ = [
    "LLMRAGExtraction",
    "MockRAGProvider",
    "OpenAIRAGProvider",
    "PROMPT_VERSION",
    "RAGEngine",
    "RAGProvider",
]
