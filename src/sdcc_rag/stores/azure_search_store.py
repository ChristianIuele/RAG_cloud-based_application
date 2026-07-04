"""Vector store su Azure AI Search (cloud).

Adapter speculare a `ChromaVectorStore`: implementa la stessa porta `IVectorStore`
(`upsert`/`query`/`count`) e, come Chroma, è **passivo sugli embedding** — i vettori
sono calcolati a monte dall'`IEmbeddingProvider` (orchestratore in ingestion,
`RAGService` in query) e passati esplicitamente. Lo store non vettorizza nulla.

Schema dell'indice `rag-documents` (creato da `infrastructure/setup_azure_search_index.py`):

    id              chiave (chunk_id deterministico)
    doc_id          Document.source (filtrabile/facetable: purge dei chunk orfani)
    content         testo del chunk (searchable)
    metadata        dizionario dei metadati serializzato in JSON (non searchable)
    content_vector  embedding del chunk (HNSW vector search)

`source` viene incluso nel JSON di `metadata` (specularmente a ChromaVectorStore, che
lo mette tra i metadata scalari) e ri-estratto in `query`. A differenza di Chroma, qui
`metadata` è una stringa JSON: può quindi conservare anche valori non scalari (es. liste).

Nota sullo score: in **hybrid search** (testo + vettore) Azure fonde i risultati con
RRF (Reciprocal Rank Fusion), quindi `@search.score` è il punteggio RRF, su scala
diversa sia dalla cosine similarity sia dal `1 - distance` di Chroma. L'adapter applica
comunque un **taglio di rilevanza** proprio, `settings.azure_search_min_score`
(default 0.0 = nessun filtro, da ricalibrare sulla scala RRF del corpus reale): i match
sotto soglia sono rimossi in `query()` prima che il contesto raggiunga il modello
generativo. La soglia è separata dal globale `retrieval_min_score` (usato da Chroma)
proprio perché le scale non sono confrontabili.
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
        # Soglia coseno per scartare i match spuri (vedi docstring di modulo).
        self._min_score = settings.azure_search_min_score

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        documents = [self._to_document(ec) for ec in embedded_chunks]
        # upload_documents fa upsert keyed sull'`id`: re-ingestare aggiorna, non duplica.
        for start in range(0, len(documents), _BATCH_SIZE):
            self._client.upload_documents(documents=documents[start : start + _BATCH_SIZE])

    def query(
        self, embedding: list[float], top_k: int = 5, query_text: str | None = None
    ) -> list[RetrievedChunk]:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields=_VECTOR_FIELD,
        )
        # Hybrid search: passando SIA il vettore SIA il testo grezzo, Azure combina
        # ricerca lessicale (BM25 sul campo `content` searchable) e vettoriale, fondendo
        # i risultati con RRF. Con `query_text=None` degrada alla ricerca puramente
        # vettoriale (retro-compatibile).
        results = self._client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            top=top_k,
            select=["id", "content", "metadata"],
        )
        retrieved = [self._to_retrieved(result) for result in results]
        # Taglio di rilevanza: i match sotto la soglia coseno sono spuri e non
        # devono entrare nel contesto passato al modello generativo.
        return [rc for rc in retrieved if rc.score >= self._min_score]

    def count(self) -> int:
        return self._client.get_document_count()

    def delete_by_doc_id(self, doc_id: str) -> None:
        # `metadata` è una stringa JSON non filtrabile: la cancellazione per documento
        # si appoggia al campo top-level `doc_id` (filtrabile). Prima si recuperano le
        # chiavi (`id` = chunk_id) del documento, poi si cancellano a batch.
        escaped = doc_id.replace("'", "''")  # escape OData degli apici singoli
        results = self._client.search(
            search_text="*",
            filter=f"doc_id eq '{escaped}'",
            select=["id"],
        )
        ids = [result["id"] for result in results]
        if not ids:
            return
        for start in range(0, len(ids), _BATCH_SIZE):
            batch = ids[start : start + _BATCH_SIZE]
            self._client.delete_documents(documents=[{"id": i} for i in batch])

    def get_all_doc_ids(self) -> set[str]:
        # Faceting sul campo `doc_id`: una sola chiamata restituisce i valori distinti
        # (con `top=0` non scarica i documenti). `count:10000` alza il limite di default (10).
        results = self._client.search(
            search_text="*",
            facets=["doc_id,count:10000"],
            top=0,
        )
        facets = results.get_facets() or {}
        return {facet["value"] for facet in facets.get("doc_id", [])}

    @staticmethod
    def _to_document(ec: EmbeddedChunk) -> dict[str, object]:
        # `source` viaggia dentro il JSON di metadata (speculare a ChromaVectorStore) e,
        # in più, come campo top-level `doc_id` filtrabile per il purge dei chunk orfani.
        metadata = {"source": ec.chunk.source, **ec.chunk.metadata}
        return {
            "id": ec.chunk.chunk_id,
            "doc_id": ec.chunk.source,
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
