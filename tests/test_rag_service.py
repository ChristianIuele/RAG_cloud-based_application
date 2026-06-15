"""Test dell'orchestratore di query RAGService con fake (nessuna rete)."""

from __future__ import annotations

from conftest import EchoLLMProvider, FakeEmbeddingProvider, FakeVectorStore

from sdcc_rag.domain.models import Chunk, EmbeddedChunk
from sdcc_rag.retrieval.prompt import ABSTENTION_TEXT, SYSTEM_PROMPT
from sdcc_rag.retrieval.rag_service import RAGService


def _populate(store: FakeVectorStore, *chunks: Chunk) -> None:
    for chunk in chunks:
        store.upsert([EmbeddedChunk(chunk=chunk, embedding=[0.0, 0.0])])


def _chunk(text: str, source: str) -> Chunk:
    return Chunk(text=text, chunk_id=f"id-{text[:8]}-{source}", source=source)


def test_store_vuoto_astiene_senza_chiamare_llm():
    llm = EchoLLMProvider()
    service = RAGService(FakeEmbeddingProvider(), FakeVectorStore(), llm)

    answer = service.answer("Cos'è l'SDCC?")

    assert answer.text == ABSTENTION_TEXT
    assert answer.sources == [] and answer.chunks == []
    assert llm.calls == 0  # empty-guard: nessuna chiamata all'LLM senza contesto


def test_chunk_sotto_soglia_astiene_senza_chiamare_llm():
    store = FakeVectorStore(default_score=0.1)  # rilevanza bassa
    _populate(store, _chunk("contenuto poco pertinente", "doc1.txt"))
    llm = EchoLLMProvider()
    service = RAGService(FakeEmbeddingProvider(), store, llm, min_score=0.5)

    answer = service.answer("domanda fuori tema")

    assert answer.text == ABSTENTION_TEXT
    assert llm.calls == 0


def test_risposta_fondata_passa_contesto_e_system_all_llm():
    store = FakeVectorStore(default_score=0.9)
    _populate(
        store,
        _chunk("L'SDCC è il sistema di controllo distribuito.", "doc1.txt"),
        _chunk("Gestisce gli allarmi in tempo reale.", "doc2.md"),
    )
    llm = EchoLLMProvider(answer="L'SDCC è il sistema di controllo distribuito [1].")
    service = RAGService(FakeEmbeddingProvider(), store, llm, min_score=0.5)

    answer = service.answer("Cos'è l'SDCC?")

    # risposta generata e propagata
    assert answer.text == "L'SDCC è il sistema di controllo distribuito [1]."
    assert llm.calls == 1
    # fonti allineate ai passaggi tenuti (1:1, no dedup)
    assert answer.sources == ["doc1.txt", "doc2.md"]
    assert len(answer.chunks) == 2
    # il system prompt (policy) è arrivato separato dai dati
    assert llm.last_system == SYSTEM_PROMPT
    # i testi dei chunk sono nel messaggio utente (grounding)
    assert "sistema di controllo distribuito" in llm.last_prompt
    assert "allarmi in tempo reale" in llm.last_prompt
