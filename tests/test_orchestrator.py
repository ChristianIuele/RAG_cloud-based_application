"""Test dell'orchestratore con fake provider/store (nessuna rete, nessun Chroma)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeEmbeddingProvider, FakeLLMProvider, FakeVectorStore

from sdcc_rag.enrichment.composite_enricher import CompositeMetadataEnricher
from sdcc_rag.enrichment.manual_enricher import ManualMetadataEnricher
from sdcc_rag.enrichment.semantic_enricher import SemanticMetadataEnricher
from sdcc_rag.enrichment.standard_enricher import StandardMetadataEnricher
from sdcc_rag.domain.models import Document
from sdcc_rag.ingestion.orchestrator import IngestionOrchestrator
from sdcc_rag.loaders.json_loader import JsonLoader
from sdcc_rag.loaders.text_loader import TextLoader
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter


def _build(store: FakeVectorStore, embedder: FakeEmbeddingProvider) -> IngestionOrchestrator:
    enricher = CompositeMetadataEnricher(
        [
            StandardMetadataEnricher(),
            SemanticMetadataEnricher(FakeLLMProvider()),
            ManualMetadataEnricher({"title": "Manuale SDCC", "author": "ACME"}),
        ]
    )
    return IngestionOrchestrator(
        loaders=[TextLoader(), JsonLoader(text_field="text")],
        splitter=RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5),
        embedder=embedder,
        store=store,
        enricher=enricher,
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


def test_metadati_arricchiti_arrivano_nei_chunk(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta gamma delta " * 10, encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    _build(store, embedder).ingest_path(tmp_path)

    meta = next(iter(store.items.values())).chunk.metadata
    # doc-level (tracciabilità + semantico + manuale) sceso nei chunk via splitter
    assert {"content_sha256", "ingested_at", "source_stem", "summary", "keywords",
            "title", "author"} <= meta.keys()
    # chunk-level
    assert "char_count" in meta and "word_count" in meta
    # tutti i valori sono scalari (sopravvivono al filtro di ChromaVectorStore)
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


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


def test_ingest_documents_riusa_la_pipeline():
    # Flusso "Azure": documenti già caricati in memoria, nessun filesystem.
    documents = [
        Document(content="alpha beta gamma delta " * 10, source="blob://a.txt",
                 metadata={"filename": "a.txt"}),
        Document(content="record singolo", source="blob://b.json#0",
                 metadata={"filename": "b.json"}),
    ]

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_documents(documents)

    assert report.documents_loaded == 2
    assert report.chunks_indexed > 0
    assert store.count() == report.chunks_indexed

    # stesso arricchimento del flusso locale (doc-level + chunk-level) nei chunk
    meta = next(iter(store.items.values())).chunk.metadata
    assert {"summary", "keywords", "title", "author", "char_count", "word_count"} <= meta.keys()


def test_json_corrotto_registra_errore_senza_fermarsi(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("contenuto", encoding="utf-8")
    (tmp_path / "rotto.json").write_text("{ non valido", encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_path(tmp_path)

    assert report.documents_loaded == 1  # il txt è stato comunque processato
    assert any("rotto.json" in e for e in report.errors)
