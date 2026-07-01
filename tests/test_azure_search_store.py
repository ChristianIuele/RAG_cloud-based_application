"""Test dell'adapter AzureSearchVectorStore senza chiamate cloud.

Il `SearchClient` reale viene sostituito da un fake in-memory: si verifica la
logica pura dell'adapter (mapping EmbeddedChunk→documento Azure, batching,
de-serializzazione risultato→RetrievedChunk) e la guardia sulle credenziali.
"""

from __future__ import annotations

import json

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.domain.models import Chunk, EmbeddedChunk
from sdcc_rag.stores.azure_search_store import AzureSearchVectorStore


class FakeSearchClient:
    """Sostituto del SearchClient Azure: registra gli upload e risponde a search."""

    def __init__(self) -> None:
        self.uploaded_batches: list[list[dict]] = []
        self.search_results: list[dict] = []

    def upload_documents(self, documents):
        self.uploaded_batches.append(list(documents))

    def search(self, **kwargs):
        self.last_search_kwargs = kwargs
        return iter(self.search_results)

    def get_document_count(self) -> int:
        return sum(len(b) for b in self.uploaded_batches)


def _settings(**overrides) -> Settings:
    base = {
        "azure_search_endpoint": "https://svc.search.windows.net",
        "azure_search_admin_key": "key",
    }
    base.update(overrides)
    return Settings(**base)


def _store_with_fake(min_score: float = 0.0) -> tuple[AzureSearchVectorStore, FakeSearchClient]:
    """Costruisce lo store evitando l'__init__ reale (che istanzia l'SDK Azure).

    `min_score` replica `settings.azure_search_min_score`: default 0.0 (nessun
    filtro) così i test non-soglia restano invariati.
    """
    store = AzureSearchVectorStore.__new__(AzureSearchVectorStore)
    fake = FakeSearchClient()
    store._client = fake
    store._min_score = min_score
    return store, fake


def _embedded(chunk_id: str, text: str = "testo", source: str = "doc.txt", **metadata):
    chunk = Chunk(text=text, chunk_id=chunk_id, source=source, metadata=metadata)
    return EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3])


def test_costruttore_richiede_credenziali():
    with pytest.raises(ValueError, match="AZURE_SEARCH_ADMIN_KEY"):
        AzureSearchVectorStore(_settings(azure_search_admin_key=None))


def test_to_document_serializza_metadata_e_source():
    ec = _embedded("id-1", text="ciao", source="a/b.txt", title="T", keywords=["x", "y"])
    doc = AzureSearchVectorStore._to_document(ec)

    assert doc["id"] == "id-1"
    assert doc["content"] == "ciao"
    assert doc["content_vector"] == [0.1, 0.2, 0.3]
    meta = json.loads(doc["metadata"])
    assert meta["source"] == "a/b.txt"
    assert meta["title"] == "T"
    assert meta["keywords"] == ["x", "y"]  # liste preservate (JSON, non scalare)


def test_upsert_vuoto_non_chiama_il_client():
    store, fake = _store_with_fake()
    store.upsert([])
    assert fake.uploaded_batches == []


def test_upsert_carica_documenti_mappati():
    store, fake = _store_with_fake()
    store.upsert([_embedded("id-1"), _embedded("id-2")])

    assert len(fake.uploaded_batches) == 1
    ids = [d["id"] for d in fake.uploaded_batches[0]]
    assert ids == ["id-1", "id-2"]


def test_upsert_batching_oltre_il_limite():
    store, fake = _store_with_fake()
    store.upsert([_embedded(f"id-{i}") for i in range(2300)])

    # 2300 documenti → batch da 1000: 1000 + 1000 + 300
    assert [len(b) for b in fake.uploaded_batches] == [1000, 1000, 300]


def test_query_deserializza_risultati():
    store, fake = _store_with_fake()
    fake.search_results = [
        {
            "id": "id-1",
            "content": "passaggio rilevante",
            "metadata": json.dumps({"source": "doc.txt", "title": "T"}),
            "@search.score": 0.87,
        }
    ]

    retrieved = store.query([0.1, 0.2, 0.3], top_k=5)

    assert len(retrieved) == 1
    rc = retrieved[0]
    assert rc.chunk.chunk_id == "id-1"
    assert rc.chunk.text == "passaggio rilevante"
    assert rc.chunk.source == "doc.txt"           # estratto dal JSON
    assert rc.chunk.metadata == {"title": "T"}    # source rimosso dai metadata residui
    assert rc.score == pytest.approx(0.87)
    # il vettore di query è passato come VectorizedQuery sul campo content_vector
    vq = fake.last_search_kwargs["vector_queries"][0]
    assert vq.fields == "content_vector"
    assert vq.k_nearest_neighbors == 5


def test_query_filtra_chunk_sotto_soglia():
    # Mock di Azure con un match forte (0.85) e uno spurio (0.40).
    store, fake = _store_with_fake(min_score=0.70)
    fake.search_results = [
        {"id": "alto", "content": "pertinente", "@search.score": 0.85},
        {"id": "basso", "content": "spurio", "@search.score": 0.40},
    ]

    retrieved = store.query([0.1, 0.2, 0.3], top_k=5)

    # Solo il chunk sopra soglia sopravvive; quello a 0.40 è scartato dall'adapter.
    assert [rc.chunk.chunk_id for rc in retrieved] == ["alto"]
    assert retrieved[0].score == pytest.approx(0.85)


def test_query_senza_soglia_non_filtra():
    # Con min_score=0.0 (default globale) nessun match viene rimosso.
    store, fake = _store_with_fake(min_score=0.0)
    fake.search_results = [
        {"id": "alto", "content": "x", "@search.score": 0.85},
        {"id": "basso", "content": "y", "@search.score": 0.40},
    ]

    retrieved = store.query([0.1, 0.2, 0.3], top_k=5)

    assert [rc.chunk.chunk_id for rc in retrieved] == ["alto", "basso"]


def test_query_metadata_assente():
    store, fake = _store_with_fake()
    fake.search_results = [{"id": "id-1", "content": "x", "@search.score": 1.0}]

    rc = store.query([0.0], top_k=1)[0]
    assert rc.chunk.source == ""
    assert rc.chunk.metadata == {}


def test_count_delega_al_client():
    store, fake = _store_with_fake()
    store.upsert([_embedded("id-1"), _embedded("id-2")])
    assert store.count() == 2
