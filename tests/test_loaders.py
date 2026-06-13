"""Test dei loader TXT/MD/JSON e del registry."""

from __future__ import annotations

import json
from pathlib import Path

from sdcc_rag.loaders.json_loader import JsonLoader
from sdcc_rag.loaders.registry import LoaderRegistry
from sdcc_rag.loaders.text_loader import TextLoader


def test_text_loader_txt_e_md(tmp_path: Path):
    loader = TextLoader()
    f = tmp_path / "nota.md"
    f.write_text("# Titolo\ncorpo", encoding="utf-8")

    docs = loader.load(f)

    assert len(docs) == 1
    assert docs[0].content == "# Titolo\ncorpo"
    assert docs[0].metadata["extension"] == ".md"


def test_json_loader_lista_di_record(tmp_path: Path):
    loader = JsonLoader(text_field="body")
    f = tmp_path / "corpus.json"
    f.write_text(
        json.dumps([{"body": "primo", "id": 1}, {"body": "secondo", "id": 2}]),
        encoding="utf-8",
    )

    docs = loader.load(f)

    assert [d.content for d in docs] == ["primo", "secondo"]
    assert docs[0].metadata["id"] == 1
    assert "body" not in docs[0].metadata  # il campo testo non finisce nei metadata


def test_json_loader_oggetto_singolo(tmp_path: Path):
    loader = JsonLoader(text_field="text")
    f = tmp_path / "uno.json"
    f.write_text(json.dumps({"text": "contenuto"}), encoding="utf-8")

    docs = loader.load(f)

    assert len(docs) == 1
    assert docs[0].content == "contenuto"


def test_registry_risolve_per_estensione(tmp_path: Path):
    registry = LoaderRegistry([TextLoader(), JsonLoader()])

    assert isinstance(registry.resolve(tmp_path / "a.txt"), TextLoader)
    assert isinstance(registry.resolve(tmp_path / "b.json"), JsonLoader)
    assert registry.resolve(tmp_path / "c.pdf") is None  # estensione non supportata
