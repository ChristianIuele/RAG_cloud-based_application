"""Test dell'adapter ExtractorMetadataEnricher (extractor → pipeline)."""

from __future__ import annotations

from conftest import FakeMetadataExtractor

from sdcc_rag.domain.models import AutomaticMetadata, Chunk, Document, make_chunk_id
from sdcc_rag.enrichment.extractor_enricher import ExtractorMetadataEnricher


def _doc(content: str = "contenuto di prova") -> Document:
    return Document(content=content, source="s", metadata={"filename": "manuale.md"})


def _chunk(text: str = "alpha beta") -> Chunk:
    return Chunk(text=text, chunk_id=make_chunk_id("s", 0, text), source="s", metadata={})


def test_appiattisce_liste_a_stringa_scalare():
    extractor = FakeMetadataExtractor(
        AutomaticMetadata(
            summary="S",
            keywords=["a", "b", "c"],
            suggested_categories=["x", "y"],
            language="it",
            entities=["SDCC", "ACME"],
        )
    )
    out = ExtractorMetadataEnricher(extractor).enrich_document(_doc())

    assert out.metadata["summary"] == "S"
    assert out.metadata["language"] == "it"
    assert out.metadata["keywords"] == "a, b, c"
    assert out.metadata["suggested_categories"] == "x, y"
    assert out.metadata["entities"] == "SDCC, ACME"
    # tutti i valori automatici sono scalari
    assert all(isinstance(v, (str, int, float, bool)) for v in out.metadata.values())


def test_merge_non_distruttivo_e_passa_il_testo():
    extractor = FakeMetadataExtractor(AutomaticMetadata(summary="S"))
    out = ExtractorMetadataEnricher(extractor).enrich_document(_doc("testo doc"))

    assert out.metadata["filename"] == "manuale.md"  # chiavi preesistenti preservate
    assert out.metadata["summary"] == "S"
    assert extractor.last_text == "testo doc"  # il contenuto è passato all'extractor


def test_metadata_vuoto_lascia_documento_invariato():
    out = ExtractorMetadataEnricher(
        FakeMetadataExtractor(AutomaticMetadata())
    ).enrich_document(_doc())
    assert out.metadata == {"filename": "manuale.md"}


def test_enrich_chunk_e_noop():
    chunk = _chunk()
    out = ExtractorMetadataEnricher(FakeMetadataExtractor()).enrich_chunk(chunk)
    assert out is chunk


def test_immutabilita_originale():
    doc = _doc()
    ExtractorMetadataEnricher(FakeMetadataExtractor()).enrich_document(doc)
    assert "summary" not in doc.metadata  # l'originale non viene mutato
