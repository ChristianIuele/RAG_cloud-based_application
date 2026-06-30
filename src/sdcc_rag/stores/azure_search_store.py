"""Vector store su Azure AI Search (cloud).

Adapter speculare a `ChromaVectorStore`: implementa la stessa porta `IVectorStore`
(`upsert`/`query`/`count`) e, come Chroma, è **passivo sugli embedding** — i vettori
sono calcolati a monte dall'`IEmbeddingProvider` (orchestratore in ingestion,
`RAGService` in query) e passati esplicitamente. Lo store non vettorizza nulla.

Schema dell'indice `rag-documents` (creato da `infrastructure/setup_azure_search_index.py`):

    id              chiave (chunk_id deterministico)
    content         testo del chunk (searchable)
    metadata        dizionario dei metadati serializzato in JSON (non searchable)
    content_vector  embedding del chunk (HNSW vector search)

`source` viene incluso nel JSON di `metadata` (specularmente a ChromaVectorStore, che
lo mette tra i metadata scalari) e ri-estratto in `query`. A differenza di Chroma, qui
`metadata` è una stringa JSON: può quindi conservare anche valori non scalari (es. liste).

Nota sullo score: `@search.score` di Azure (metrica cosine) è su scala diversa dal
`1 - distance` di Chroma. La soglia `retrieval_min_score` va perciò **ricalibrata**
quando si passa a questo store (default 0.0 = nessun filtro).
"""

from __future__ import annotations

import json

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IVectorStore
from sdcc_rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk

# Azure AI Search accetta fino a ~1000 documenti (o 16 MB) per richiesta di upload.
_BATCH_SIZE = 1000

# Nome del campo vettoriale nell'indice (vedi setup_azure_search_index.py).
_VECTOR_FIELD = "content_vector"


class AzureSearchVectorStore(IVectorStore):
    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_SEARCH_ENDPOINT": settings.azure_search_endpoint,
                "AZURE_SEARCH_ADMIN_KEY": settings.azure_search_admin_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Variabili Azure AI Search mancanti per il vector store "
                "'azure_search': " + ", ".join(missing)
            )

        # import locale: l'SDK serve solo quando questo store è effettivamente in uso
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self._client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_admin_key),
        )

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        documents = [self._to_document(ec) for ec in embedded_chunks]
        # upload_documents fa upsert keyed sull'`id`: re-ingestare aggiorna, non duplica.
        for start in range(0, len(documents), _BATCH_SIZE):
            self._client.upload_documents(documents=documents[start : start + _BATCH_SIZE])

    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields=_VECTOR_FIELD,
        )
        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
            select=["id", "content", "metadata"],
        )
        return [self._to_retrieved(result) for result in results]

    def count(self) -> int:
        return self._client.get_document_count()

    @staticmethod
    def _to_document(ec: EmbeddedChunk) -> dict[str, object]:
        # `source` viaggia dentro il JSON di metadata (speculare a ChromaVectorStore).
        metadata = {"source": ec.chunk.source, **ec.chunk.metadata}
        return {
            "id": ec.chunk.chunk_id,
            "content": ec.chunk.text,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            _VECTOR_FIELD: ec.embedding,
        }

    @staticmethod
    def _to_retrieved(result: dict) -> RetrievedChunk:
        raw = result.get("metadata")
        metadata = json.loads(raw) if raw else {}
        source = str(metadata.pop("source", ""))
        chunk = Chunk(
            text=result.get("content", ""),
            chunk_id=result.get("id", ""),
            source=source,
            metadata=metadata,
        )
        # `@search.score`: rilevanza Azure (più alto = più pertinente).
        return RetrievedChunk(chunk=chunk, score=float(result["@search.score"]))
