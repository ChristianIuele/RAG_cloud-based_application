"""Test dell'orchestratore con fake provider/store (nessuna rete, nessun Chroma)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeEmbeddingProvider, FakeMetadataExtractor, FakeVectorStore

from sdcc_rag.enrichment.composite_enricher import CompositeMetadataEnricher
from sdcc_rag.enrichment.extractor_enricher import ExtractorMetadataEnricher
from sdcc_rag.enrichment.manual_enricher import ManualMetadataEnricher
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
            ExtractorMetadataEnricher(FakeMetadataExtractor()),
            ManualMetadataEnricher(
                {
                    "title": "Manuale SDCC",
                    "author": "ACME",
                    "category": "Tecnico",
                    "description": "Guida operativa",
                    "tags": "sdcc, rag",
                }
            ),
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
    # doc-level (tracciabilità + automatico + manuale) sceso nei chunk via splitter
    assert {"content_sha256", "ingested_at", "source_stem",
            "summary", "keywords", "language", "suggested_categories", "entities",
            "title", "author", "category", "description", "tags"} <= meta.keys()
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
    assert {"summary", "keywords", "language", "title", "author",
            "char_count", "word_count"} <= meta.keys()


def test_json_corrotto_registra_errore_senza_fermarsi(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("contenuto", encoding="utf-8")
    (tmp_path / "rotto.json").write_text("{ non valido", encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    report = _build(store, embedder).ingest_path(tmp_path)

    assert report.documents_loaded == 1  # il txt è stato comunque processato
    assert any("rotto.json" in e for e in report.errors)


# --- Sync & Purge dei chunk orfani -------------------------------------------


def test_sync_and_ingest_rimuove_documenti_orfani():
    a = Document(content="alpha beta gamma delta " * 10, source="doc://a.txt")
    b = Document(content="uno due tre quattro cinque " * 10, source="doc://b.txt")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    orch = _build(store, embedder)

    orch.sync_and_ingest([a, b])  # corpus iniziale: A + B
    assert store.get_all_doc_ids() == {"doc://a.txt", "doc://b.txt"}

    # B sparisce dalla sorgente: i suoi chunk vanno purgati come orfani.
    report = orch.sync_and_ingest([a])

    assert report.documents_pruned == 1
    assert store.get_all_doc_ids() == {"doc://a.txt"}
    assert "doc://b.txt" in store.deleted_doc_ids


def test_sync_and_ingest_purge_versioni_precedenti():
    # Scenario centrale: un documento viene MODIFICATO. I chunk_id (hash del testo)
    # cambiano, quindi senza purge le versioni vecchie resterebbero orfane nel DB.
    v1 = Document(content="alpha beta gamma delta " * 10, source="doc://a.txt")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    orch = _build(store, embedder)

    orch.sync_and_ingest([v1])
    old_ids = set(store.items.keys())
    assert old_ids  # sono stati indicizzati dei chunk

    v2 = Document(content="testo completamente diverso ora " * 12, source="doc://a.txt")
    orch.sync_and_ingest([v2])

    # Nessun chunk_id della v1 sopravvive e il documento resta unico.
    assert old_ids.isdisjoint(store.items.keys())
    assert store.get_all_doc_ids() == {"doc://a.txt"}
    assert all(ec.chunk.source == "doc://a.txt" for ec in store.items.values())


def test_sync_and_ingest_idempotente():
    a = Document(content="alpha beta gamma delta " * 10, source="doc://a.txt")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    orch = _build(store, embedder)

    orch.sync_and_ingest([a])
    count_after_first = store.count()
    report = orch.sync_and_ingest([a])  # stessa sorgente, nessuna modifica

    assert store.count() == count_after_first  # nessun duplicato
    assert report.documents_pruned == 0  # nessun orfano


def test_sync_and_ingest_path_purga_file_rimosso(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta gamma delta " * 10, encoding="utf-8")
    (tmp_path / "b.txt").write_text("uno due tre quattro " * 10, encoding="utf-8")

    store, embedder = FakeVectorStore(), FakeEmbeddingProvider()
    orch = _build(store, embedder)

    orch.sync_and_ingest_path(tmp_path)
    assert store.get_all_doc_ids() == {str(tmp_path / "a.txt"), str(tmp_path / "b.txt")}

    (tmp_path / "b.txt").unlink()  # file eliminato dalla sorgente
    report = orch.sync_and_ingest_path(tmp_path)

    assert report.documents_pruned == 1
    assert store.get_all_doc_ids() == {str(tmp_path / "a.txt")}
