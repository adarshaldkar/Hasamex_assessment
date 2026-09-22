from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)

    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    google_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openrouter_api_key: str = ""
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_fallback: str = "hash"

    data_dir: Path = (ROOT_DIR / "data" / "raw") if (ROOT_DIR / "data" / "raw").exists() else (ROOT_DIR / "data")
    storage_dir: Path = ROOT_DIR / "storage"
    chroma_dir: Path = ROOT_DIR / "storage" / "chroma"
    processed_dir: Path = ROOT_DIR / "storage" / "processed"
    vector_store_path: Path = ROOT_DIR / "storage" / "chroma"
    processed_data_path: Path = ROOT_DIR / "storage" / "processed"

    rrf_k: int = Field(default=60, ge=1)
    evidence_gate_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=4, ge=1)
    retrieval_candidate_k: int = Field(default=8, ge=1)
    top_k_chunks: int = Field(default=8, ge=1)
    max_evidence: int = Field(default=4, ge=1)

    extraction_prompt_version: str = "extraction_v1"
    synthesis_prompt_version: str = "synthesis_v1"
    rag_prompt_version: str = "rag_v1"

    def ensure_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]
