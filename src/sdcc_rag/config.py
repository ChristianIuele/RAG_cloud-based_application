"""Configurazione centralizzata, caricata da variabili d'ambiente / file .env.

Usa pydantic-settings: ogni campo può essere sovrascritto da una env var omonima
(case-insensitive). È qui che vive lo switch OpenAI <-> Azure (`embedding_provider`).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Selezione provider di embedding -------------------------------------
    embedding_provider: str = "openai"  # "openai" | "azure" | "ollama"

    # --- OpenAI --------------------------------------------------------------
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Azure OpenAI Service ------------------------------------------------
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None  # nome del deployment dell'embedding
    azure_openai_api_version: str = "2024-02-01"

    # --- Ollama (locale, gratuito) -------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- LLM (arricchimento semantico) ---------------------------------------
    llm_provider: str = "ollama"  # oggi: "ollama"
    ollama_llm_model: str = "llama3"
    semantic_max_chars: int = 4000  # testo max passato all'LLM per summary/keywords

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- ChromaDB ------------------------------------------------------------
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "sdcc"

    # --- Loader JSON ---------------------------------------------------------
    json_text_field: str = "text"  # campo da cui estrarre il testo nei record JSON

    # --- Ingestion -----------------------------------------------------------
    data_path: str = "./data"
    batch_size: int = 64

    # --- Retrieval / query ---------------------------------------------------
    retrieval_top_k: int = 5  # numero di chunk recuperati per domanda
    # soglia di rilevanza (similarità coseno): 0.0 = nessun filtro finché non
    # viene calibrata contro il corpus reale.
    retrieval_min_score: float = 0.0
