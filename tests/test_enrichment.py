"""Test degli enricher: tracciabilità, automatico (extractor), manuale, composite."""

from __future__ import annotations

import hashlib
from datetime import datetime

from conftest import FakeMetadataExtractor

from sdcc_rag.domain.models import Chunk, Document, make_chunk_id
from sdcc_rag.enrichment.composite_enricher import CompositeMetadataEnricher
from sdcc_rag.enrichment.extractor_enricher import ExtractorMetadataEnricher
from sdcc_rag.enrichment.manual_enricher import ManualMetadataEnricher
from sdcc_rag.enrichment.standard_enricher import StandardMetadataEnricher


def _doc(content: str = "contenuto di prova", source: str = r"C:\data\manuale.md") -> Document:
    return Document(content=content, source=source, metadata={"filename": "manuale.md"})


def _chunk(text: str = "alpha beta gamma") -> Chunk:
    return Chunk(text=text, chunk_id=make_chunk_id("s", 0, text), source="s", metadata={})


# --- Standard ---------------------------------------------------------------

def test_standard_document_aggiunge_tracciabilita():
    doc = _doc()
    out = StandardMetadataEnricher().enrich_document(doc)

    assert out.metadata["content_sha256"] == hashlib.sha256(doc.content.encode()).hexdigest()
    assert out.metadata["source_stem"] == "manuale"
    assert out.metadata["source_ext"] == ".md"
    datetime.fromisoformat(out.metadata["ingested_at"])  # ISO valido
    assert out.metadata["filename"] == "manuale.md"  # chiavi preesistenti preservate


def test_standard_document_gestisce_suffisso_json():
    out = StandardMetadataEnricher().enrich_document(_doc(source=r"C:\data\corpus.json#3"))
    assert out.metadata["source_stem"] == "corpus"
    assert out.metadata["source_ext"] == ".json"


def test_standard_chunk_aggiunge_conteggi_senza_toccare_id():
    chunk = _chunk("alpha beta gamma")
    out = StandardMetadataEnricher().enrich_chunk(chunk)

    assert out.metadata["char_count"] == len("alpha beta gamma")
    assert out.metadata["word_count"] == 3
    assert out.chunk_id == chunk.chunk_id  # idempotenza preservata
    assert out.text == chunk.text and out.source == chunk.source


def test_standard_immutabilita():
    doc = _doc()
    StandardMetadataEnricher().enrich_document(doc)
    assert "content_sha256" not in doc.metadata  # l'originale non viene mutato


# --- Manual -----------------------------------------------------------------

def test_manual_inietta_campi():
    out = ManualMetadataEnricher({"title": "T", "author": "A"}).enrich_document(_doc())
    assert out.metadata["title"] == "T"
    assert out.metadata["author"] == "A"


def test_manual_ignora_valori_vuoti():
    out = ManualMetadataEnricher({"title": "", "author": None}).enrich_document(_doc())
    assert "title" not in out.metadata and "author" not in out.metadata


# --- Composite --------------------------------------------------------------

def test_composite_applica_tutti_in_ordine():
    enricher = CompositeMetadataEnricher(
        [
            StandardMetadataEnricher(),
            ExtractorMetadataEnricher(FakeMetadataExtractor()),
            ManualMetadataEnricher({"title": "Manuale SDCC"}),
        ]
    )
    doc = enricher.enrich_document(_doc())
    chunk = enricher.enrich_chunk(_chunk())

    # campi dei tre enricher tutti presenti a livello documento
    assert {"content_sha256", "summary", "keywords", "title"} <= doc.metadata.keys()
    # il livello chunk riceve i conteggi
    assert "char_count" in chunk.metadata
