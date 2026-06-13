"""Test dello splitter ricorsivo: chunking, overlap, idempotenza degli id."""

from __future__ import annotations

import pytest

from sdcc_rag.domain.models import Document
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter


def _doc(text: str) -> Document:
    return Document(content=text, source="mem://doc", metadata={"k": "v"})


def test_rispetta_chunk_size():
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    text = "parola " * 100  # ben oltre 50 caratteri
    chunks = splitter.split(_doc(text))

    assert len(chunks) > 1
    # tolleranza per l'overlap riportato a cavallo
    assert all(len(c.text) <= 50 + 10 for c in chunks)


def test_id_deterministici():
    splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5)
    text = "alpha beta gamma delta epsilon zeta eta theta iota"

    first = splitter.split(_doc(text))
    second = splitter.split(_doc(text))

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)  # nessuna collisione


def test_metadata_propagati_con_indice():
    splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split(_doc("uno due tre quattro cinque sei sette otto nove dieci"))

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["k"] == "v"
        assert chunk.metadata["chunk_index"] == index


def test_documento_corto_singolo_chunk():
    splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split(_doc("testo breve"))

    assert len(chunks) == 1
    assert chunks[0].text == "testo breve"


def test_overlap_invalido_solleva():
    with pytest.raises(ValueError):
        RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=100)
