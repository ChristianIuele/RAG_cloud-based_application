"""Configurazione centralizzata, caricata da variabili d'ambiente / file .env.

Usa pydantic-settings: ogni campo può essere sovrascritto da una env var omonima
(case-insensitive). È qui che vive lo switch OpenAI <-> Azure (`embedding_provider`).
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Selezione provider di embedding -------------------------------------
    embedding_provider: str = "azure"  # "openai" | "azure" | "ollama"

    # --- OpenAI --------------------------------------------------------------
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Azure OpenAI Service ------------------------------------------------
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None  # nome del deployment dell'embedding
    azure_openai_api_version: str = "2024-02-01"

    # --- Azure Blob Storage (sorgente documenti) -----------------------------
    azure_storage_connection_string: str = ""
    azure_storage_container_name: str = ""

    # --- Ollama (locale, gratuito) -------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- LLM (arricchimento semantico) ---------------------------------------
    llm_provider: str = "azure"  # "ollama" | "azure"
    ollama_llm_model: str = "llama3"
    azure_llm_deployment: str = "gpt-5.4-nano"  # deployment chat Azure (unico deployment reale)
    # deployment dedicato all'estrazione dei metadati automatici (separato dalla
    # generazione RAG): riusa endpoint/api-key/api-version di Azure OpenAI.
    azure_metadata_deployment: str = "gpt-5.4-nano"
    semantic_max_chars: int = 4000  # testo max passato all'LLM per summary/keywords

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Selezione vector store ----------------------------------------------
    vector_store: str = "azure_search"  # "chroma" | "azure_search"

    # --- ChromaDB ------------------------------------------------------------
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "sdcc"

    # --- Azure AI Search (vector store cloud) --------------------------------
    azure_search_endpoint: str | None = None
    azure_search_admin_key: str | None = None
    # L'override da env accetta SIA il nome canonico (AZURE_SEARCH_INDEX_NAME) SIA
    # quello storico usato come App Setting in produzione (AZURE_SEARCH_INDEX): senza
    # l'alias il secondo veniva ignorato e valeva solo il default per coincidenza.
    azure_search_index_name: str = Field(
        "rag-documents",
        validation_alias=AliasChoices("AZURE_SEARCH_INDEX_NAME", "AZURE_SEARCH_INDEX"),
    )
    # Soglia di rilevanza applicata dall'adapter Azure: scarta i match spuri a basso
    # @search.score. In hybrid search lo score è RRF (Reciprocal Rank Fusion, k=60),
    # su scala diversa dal coseno e da Chroma: i punteggi tipici stanno tra ~0.01 e
    # ~0.02, quindi il default 0.015 filtra i match palesemente irrilevanti senza
    # azzerare il recall. Setting separato dal globale `retrieval_min_score`.
    azure_search_min_score: float = 0.015

    # --- Loader JSON ---------------------------------------------------------
    json_text_field: str = "text"  # campo da cui estrarre il testo nei record JSON

    # --- Ingestion -----------------------------------------------------------
    document_source: str = "azure"  # "local" | "azure"
    data_path: str = "./data"
    batch_size: int = 64

    # --- Retrieval / query ---------------------------------------------------
    retrieval_top_k: int = 5  # numero di chunk recuperati per domanda
    # soglia di rilevanza (similarità coseno) applicata da RAGService sul percorso
    # Chroma: 0.3 scarta i chunk poco pertinenti (anti top-K pollution) mantenendo
    # il recall utile. Da ricalibrare sul corpus reale se necessario.
    retrieval_min_score: float = 0.3
