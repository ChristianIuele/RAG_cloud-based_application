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
    # fonti = SOLO i passaggi citati: l'LLM ha citato [1], quindi solo doc1.txt
    assert answer.sources == ["doc1.txt"]
    assert len(answer.chunks) == 1
    # il system prompt (policy) è arrivato separato dai dati
    assert llm.last_system == SYSTEM_PROMPT
    # i testi dei chunk (entrambi) sono nel messaggio utente (grounding)
    assert "sistema di controllo distribuito" in llm.last_prompt
    assert "allarmi in tempo reale" in llm.last_prompt


def test_solo_le_fonti_citate_con_rinumerazione_sequenziale():
    store = FakeVectorStore(default_score=0.9)
    _populate(
        store,
        _chunk("Primo passaggio.", "doc1.txt"),
        _chunk("Secondo passaggio.", "doc2.md"),
        _chunk("Terzo passaggio.", "doc3.txt"),
    )
    # L'LLM cita [2] e [3] (non [1]): le fonti mostrate devono essere solo doc2/doc3
    # e il testo va rinumerato a [1]/[2] per restare coerente con la lista filtrata.
    llm = EchoLLMProvider(answer="Vedi [3] e anche [2].")
    service = RAGService(FakeEmbeddingProvider(), store, llm, min_score=0.5)

    answer = service.answer("domanda")

    # ordine di citazione: prima [3] poi [2] → nuova numerazione [1]=doc3, [2]=doc2
    assert answer.sources == ["doc3.txt", "doc2.md"]
    assert answer.text == "Vedi [1] e anche [2]."


def test_nessuna_citazione_valida_fallback_top1():
    store = FakeVectorStore(default_score=0.9)
    _populate(
        store,
        _chunk("Chunk piu rilevante.", "doc1.txt"),
        _chunk("Chunk secondario.", "doc2.md"),
    )
    # Nessuna [n] valida (l'unica citazione è fuori range): fallback al solo top-1.
    llm = EchoLLMProvider(answer="Risposta senza citazioni valide [9].")
    service = RAGService(FakeEmbeddingProvider(), store, llm, min_score=0.5)

    answer = service.answer("domanda")

    assert answer.sources == ["doc1.txt"]
    assert len(answer.chunks) == 1
