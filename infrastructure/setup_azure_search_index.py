"""Provisioning dell'indice Azure AI Search per il vector store RAG.

Script stand-alone (composition root del provisioning, sullo stile di
`scripts/ingest.py`): crea l'indice `rag-documents` sul servizio Azure AI Search
già provisionato. È il primo passo della migrazione del vector store da ChromaDB
ad Azure AI Search; non tocca il codice RAG esistente.

Lo schema riproduce ciò che oggi salviamo in Chroma (vedi `stores/chroma_store.py`):

    id              -> chunk_id deterministico (chiave primaria)
    doc_id          -> Document.source (filtrabile/facetable: purge dei chunk orfani)
    content         -> testo del chunk (searchable)
    metadata        -> dizionario dei metadati serializzato in JSON (non searchable)
    content_vector  -> embedding del chunk (HNSW vector search)

Le dimensioni di `content_vector` sono derivate dal provider di embedding
configurato (`EMBEDDING_PROVIDER`) tramite la factory esistente, così l'indice
resta sempre allineato al modello in uso senza valori cablati a mano.

Uso:
    python infrastructure/setup_azure_search_index.py [--force]

Senza `--force`, se l'indice esiste lo script avvisa ed esce senza modificarlo.
Con `--force`, elimina e ricrea l'indice.

Variabili d'ambiente richieste (lette da `.env`):
    AZURE_SEARCH_ENDPOINT     es. https://<service>.search.windows.net
    AZURE_SEARCH_ADMIN_KEY    chiave di amministrazione del servizio Search
    AZURE_SEARCH_INDEX_NAME   opzionale (default: rag-documents)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rende importabile il package `sdcc_rag` da src/ senza installazione.
# Da infrastructure/, parent.parent è la root del progetto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import ValidationError  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from sdcc_rag.config import Settings  # noqa: E402
from sdcc_rag.embeddings.factory import create_embedding_provider  # noqa: E402


class _SearchSettings(BaseSettings):
    """Credenziali del servizio Azure AI Search, lette dallo stesso `.env`.

    Modello locale (non tocca `sdcc_rag.config.Settings`) per non introdurre
    `python-dotenv` come dipendenza e riusare lo stesso meccanismo di caricamento
    già adottato dal resto del progetto.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_search_endpoint: str
    azure_search_admin_key: str
    azure_search_index_name: str = "rag-documents"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea l'indice Azure AI Search per il vector store RAG."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="se l'indice esiste già, eliminalo e ricrealo (default: avvisa ed esce)",
    )
    return parser.parse_args()


def _load_search_settings() -> _SearchSettings:
    try:
        return _SearchSettings()
    except ValidationError:
        raise SystemExit(
            "Configurazione Azure AI Search mancante: imposta AZURE_SEARCH_ENDPOINT "
            "e AZURE_SEARCH_ADMIN_KEY nel file .env (o come variabili d'ambiente)."
        )


def _build_index(name: str, dimension: int):
    """Costruisce la definizione dell'indice con vector search HNSW."""
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # doc_id (== Document.source): filtrabile per la cancellazione per documento
        # e facetable per elencare i doc_id distinti (Sync & Purge dei chunk orfani).
        SimpleField(
            name="doc_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="content", type=SearchFieldDataType.String),
        # SimpleField non è searchable: archivia i metadati come stringa JSON.
        SimpleField(name="metadata", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimension,
            vector_search_profile_name="rag-hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="rag-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="rag-hnsw-profile",
                algorithm_configuration_name="rag-hnsw",
            )
        ],
    )

    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


def main() -> None:
    args = _parse_args()
    search_settings = _load_search_settings()

    # Dimensione del vettore = single source of truth dal provider di embedding
    # configurato. La costruzione del provider non effettua chiamate API.
    settings = Settings()
    embedder = create_embedding_provider(settings)
    dimension = embedder.dimension
    print(
        f"Provider di embedding: {settings.embedding_provider!r} "
        f"→ dimensione vettore: {dimension}"
    )

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    client = SearchIndexClient(
        endpoint=search_settings.azure_search_endpoint,
        credential=AzureKeyCredential(search_settings.azure_search_admin_key),
    )

    index_name = search_settings.azure_search_index_name
    existing = set(client.list_index_names())

    if index_name in existing:
        if not args.force:
            raise SystemExit(
                f"L'indice {index_name!r} esiste già. "
                "Riesegui con --force per eliminarlo e ricrearlo."
            )
        print(f"Indice {index_name!r} esistente: eliminazione (--force)...")
        client.delete_index(index_name)

    index = _build_index(index_name, dimension)
    client.create_index(index)
    print(
        f"Indice {index_name!r} creato su {search_settings.azure_search_endpoint} "
        f"(content_vector: {dimension} dimensioni, HNSW)."
    )


if __name__ == "__main__":
    main()
