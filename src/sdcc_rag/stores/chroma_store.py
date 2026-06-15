"""Vector store su ChromaDB (persistente su disco).

Gli embedding sono calcolati a monte dall'IEmbeddingProvider e passati
esplicitamente a Chroma; non si usa quindi la embedding_function interna di
Chroma, così il provider resta l'unica sorgente di verità per la vettorizzazione.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IVectorStore
from sdcc_rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk


class ChromaVectorStore(IVectorStore):
    def __init__(self, settings: Settings) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        self._collection.upsert(
            ids=[ec.chunk.chunk_id for ec in embedded_chunks],
            embeddings=[ec.embedding for ec in embedded_chunks],
            documents=[ec.chunk.text for ec in embedded_chunks],
            metadatas=[self._metadata(ec.chunk) for ec in embedded_chunks],
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = dict(metadata or {})
            source = str(metadata.pop("source", ""))
            chunk = Chunk(text=text, chunk_id=chunk_id, source=source, metadata=metadata)
            # spazio "cosine": distance = 1 − similarità ⇒ score = 1 − distance.
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, object]:
        # Chroma accetta solo metadata scalari; `source` viene incluso per il retrieval.
        metadata: dict[str, object] = {"source": chunk.source}
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata
