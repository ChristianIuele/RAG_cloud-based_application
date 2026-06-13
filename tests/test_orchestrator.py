"""Test dell'orchestratore con fake provider/store (nessuna rete, nessun Chroma)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeEmbeddingProvider, FakeVectorStore

from sdcc_rag.ingestion.orchestrator import IngestionOrchestrator
from sdcc_rag.loaders.json_loader import JsonLoader
from sdcc_rag.loaders.text_loader import TextLoader
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter


def _build(store: FakeVectorStore, embedder: FakeEmbeddingProvider) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        loaders=[TextLoader(), JsonLoader(text_field="text")],
        splitter=RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5),
        embedder=embedder,
        store=store,
        batch_size=8,
    )


def test_ingest_conta_documenti_e_chunk(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta gamma delta " * 10, encoding="utf-8")
    (tmp_path / "b.json").write_text(
        json.dumps([{"text": "primo record"}, {"text": "secondo record"}]),
        encoding="utf-8",
    )

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_path(tmp_path)

    assert report.documents_loaded == 3  # 1 txt + 2 record json
    assert report.chunks_indexed > 0
    assert store.count() == report.chunks_indexed


def test_file_non_supportato_viene_saltato(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("contenuto valido", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("ignorato", encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_path(tmp_path)

    assert any("skip.pdf" in s for s in report.files_skipped)
    assert report.documents_loaded == 1


def test_ingestion_idempotente(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta gamma delta " * 10, encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    orch = _build(store, embedder)

    first = orch.ingest_path(tmp_path)
    count_after_first = store.count()
    orch.ingest_path(tmp_path)  # seconda passata sullo stesso corpus

    assert store.count() == count_after_first  # nessun duplicato
    assert first.chunks_indexed == count_after_first


def test_json_corrotto_registra_errore_senza_fermarsi(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("contenuto", encoding="utf-8")
    (tmp_path / "rotto.json").write_text("{ non valido", encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_path(tmp_path)

    assert report.documents_loaded == 1  # il txt è stato comunque processato
    assert any("rotto.json" in e for e in report.errors)
