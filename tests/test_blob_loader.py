"""Test del loader Azure Blob: solo unit test, nessuna rete né libreria Azure.

Il download è simulato con un Fake BlobServiceClient in-memory (NON unittest.mock),
iniettato tramite il seam di DI del costruttore. Si verifica la validazione delle
env var e la corretta delega del parsing ai loader locali via LoaderRegistry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.loaders.azure_blob_loader import AzureBlobDocumentLoader
from sdcc_rag.loaders.json_loader import JsonLoader
from sdcc_rag.loaders.registry import LoaderRegistry
from sdcc_rag.loaders.text_loader import TextLoader


class FakeBlob:
    """Risultato di download_blob: espone readall() come il client reale."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class FakeContainerClient:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs

    def download_blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._blobs[name])


class FakeBlobServiceClient:
    """Fake del BlobServiceClient: serve i bytes da un dict {blob_name: bytes}."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.requested_container: str | None = None

    def get_container_client(self, name: str) -> FakeContainerClient:
        self.requested_container = name
        return FakeContainerClient(self._blobs)


def _registry() -> LoaderRegistry:
    return LoaderRegistry([TextLoader(), JsonLoader()])


def _settings(**overrides) -> Settings:
    base = dict(
        azure_storage_connection_string="conn",
        azure_storage_container_name="docs",
    )
    base.update(overrides)
    return Settings(**base)


def _loader(blobs: dict[str, bytes], **settings_overrides) -> AzureBlobDocumentLoader:
    return AzureBlobDocumentLoader(
        _settings(**settings_overrides),
        _registry(),
        blob_service_client=FakeBlobServiceClient(blobs),
    )


def test_init_solleva_se_manca_connection_string():
    with pytest.raises(ValueError):
        AzureBlobDocumentLoader(
            _settings(azure_storage_connection_string=""),
            _registry(),
            blob_service_client=FakeBlobServiceClient({}),
        )


def test_init_solleva_se_manca_container_name():
    with pytest.raises(ValueError):
        AzureBlobDocumentLoader(
            _settings(azure_storage_container_name=""),
            _registry(),
            blob_service_client=FakeBlobServiceClient({}),
        )


def test_load_delega_al_text_loader():
    loader = _loader({"notes/a.txt": "ciao mondo".encode("utf-8")})

    docs = loader.load(Path("notes/a.txt"))

    assert len(docs) == 1
    assert docs[0].content == "ciao mondo"
    # provenance rimappata al blob name, non al path temporaneo effimero
    assert docs[0].source == "notes/a.txt"
    assert docs[0].metadata["filename"] == "a.txt"


def test_load_delega_al_json_loader():
    loader = _loader({"dati.json": b'[{"text": "primo"}, {"text": "secondo"}]'})

    docs = loader.load(Path("dati.json"))

    assert [d.content for d in docs] == ["primo", "secondo"]
    # JsonLoader usa source = f"{path}#{index}": il prefisso temp deve sparire
    assert docs[0].source == "dati.json#0"
    assert docs[1].source == "dati.json#1"


def test_load_estensione_non_supportata_solleva():
    loader = _loader({"report.pdf": b"%PDF-1.7"})

    with pytest.raises(ValueError):
        loader.load(Path("report.pdf"))


def test_supported_extensions_riflette_il_registry():
    loader = _loader({})

    assert loader.supported_extensions == {".txt", ".md", ".markdown", ".json"}
