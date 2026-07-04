"""Test dell'adapter ChromaVectorStore senza il chromadb reale (nessun I/O su disco).

La `Collection` di Chroma è sostituita da un fake in-memory iniettato via `__new__`:
si verifica la logica pura dell'adapter per il Sync & Purge dei chunk orfani
(`delete_by_doc_id` → filtro `where`, `get_all_doc_ids` → dedup dei `source`).
"""

from __future__ import annotations

from sdcc_rag.domain.models import Chunk, EmbeddedChunk
from sdcc_rag.stores.chroma_store import ChromaVectorStore


class FakeCollection:
    """Sostituto della Collection Chroma: registra delete e risponde a get."""

    def __init__(self, metadatas: list[dict] | None = None) -> None:
        self.metadatas = metadatas or []
        self.delete_calls: list[dict] = []

    def delete(self, where: dict) -> None:
        self.delete_calls.append(where)

    def get(self, include=None) -> dict:
        return {"metadatas": self.metadatas}


def _store_with_fake(metadatas: list[dict] | None = None) -> tuple[ChromaVectorStore, FakeCollection]:
    """Costruisce lo store bypassando l'__init__ reale (che apre un PersistentClient)."""
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    fake = FakeCollection(metadatas)
    store._collection = fake
    return store, fake


def _embedded(chunk_id: str, source: str) -> EmbeddedChunk:
    chunk = Chunk(text="testo", chunk_id=chunk_id, source=source)
    return EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2])


def test_delete_by_doc_id_usa_where_source():
    store, fake = _store_with_fake()

    store.delete_by_doc_id("doc.txt")

    assert fake.delete_calls == [{"source": "doc.txt"}]


def test_get_all_doc_ids_dai_metadata_deduplica():
    store, fake = _store_with_fake(
        metadatas=[
            {"source": "a.txt", "char_count": 10},
            {"source": "a.txt", "char_count": 8},  # secondo chunk dello stesso doc
            {"source": "b.txt", "char_count": 5},
        ]
    )

    assert store.get_all_doc_ids() == {"a.txt", "b.txt"}


def test_get_all_doc_ids_store_vuoto():
    store, fake = _store_with_fake(metadatas=[])
    assert store.get_all_doc_ids() == set()


def test_upsert_include_source_nei_metadata():
    # Regressione: `source` deve restare tra i metadata scalari così che
    # delete_by_doc_id/get_all_doc_ids possano filtrarci sopra.
    meta = ChromaVectorStore._metadata(_embedded("id-1", "doc.txt").chunk)
    assert meta["source"] == "doc.txt"
