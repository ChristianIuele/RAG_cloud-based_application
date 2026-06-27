"""Loader che legge documenti da Azure Blob Storage.

Adapter della porta `IDocumentLoader`: interpreta il `Path` ricevuto come *blob
name* nel container configurato, scarica il blob in un file temporaneo e **delega**
il parsing ai loader locali già testati (txt/md/json) tramite il `LoaderRegistry`.

Così il contratto `IDocumentLoader.load(path)` resta invariato e non si duplica la
logica di parsing. Coerente con l'architettura esagonale: il loader dipende solo
dall'astrazione `LoaderRegistry`/`IDocumentLoader` (i loader concreti vengono
iniettati), e l'import di `azure.storage.blob` è lazy nel costruttore.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IDocumentLoader
from sdcc_rag.domain.models import Document
from sdcc_rag.loaders.registry import LoaderRegistry


class AzureBlobDocumentLoader(IDocumentLoader):
    def __init__(
        self,
        settings: Settings,
        registry: LoaderRegistry,
        *,
        blob_service_client=None,
    ) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_STORAGE_CONNECTION_STRING": settings.azure_storage_connection_string,
                "AZURE_STORAGE_CONTAINER_NAME": settings.azure_storage_container_name,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Variabili Azure Storage mancanti per il blob loader: " + ", ".join(missing)
            )

        self._registry = registry

        # Seam di DI: nei test si inietta un Fake BlobServiceClient, così non serve
        # né la libreria né la rete. In produzione import lazy + connection string.
        service = blob_service_client
        if service is None:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            )
        self._container_client = service.get_container_client(
            settings.azure_storage_container_name
        )

    @property
    def supported_extensions(self) -> set[str]:
        # Riflette i loader effettivamente registrati: il blob loader supporta
        # esattamente ciò che i delegati sanno parsare.
        return self._registry.supported_extensions

    def load(self, path: Path) -> list[Document]:
        # I blob Azure usano '/' come separatore: as_posix() evita i backslash di
        # Windows e mantiene stabile il nome usato per il download e la provenance.
        blob_name = path.as_posix()

        delegate = self._registry.resolve(path)
        if delegate is None:
            raise ValueError(
                f"Nessun loader per l'estensione {path.suffix!r} (blob: {blob_name})"
            )

        data = self._container_client.download_blob(blob_name).readall()

        # NamedTemporaryFile(delete=False): su Windows il file non è riapribile da
        # un altro handle finché è aperto, quindi lo chiudiamo prima di delegare e
        # lo rimuoviamo nel finally. Stesso suffisso del blob → routing coerente.
        tmp = tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False)
        try:
            tmp.write(data)
            tmp.close()
            documents = delegate.load(Path(tmp.name))
        finally:
            os.unlink(tmp.name)

        # I delegati scrivono il path temporaneo (casuale, effimero) in source e
        # metadata["filename"]. Poiché make_chunk_id deriva da `source`, lasciarlo
        # romperebbe l'idempotenza tra run: rimappiamo al blob name.
        tmp_source = str(Path(tmp.name))
        return [self._rewrite_provenance(doc, tmp_source, blob_name) for doc in documents]

    @staticmethod
    def _rewrite_provenance(doc: Document, tmp_source: str, blob_name: str) -> Document:
        new_source = doc.source.replace(tmp_source, blob_name)
        new_metadata = dict(doc.metadata)
        new_metadata["filename"] = Path(blob_name).name
        return dataclasses.replace(doc, source=new_source, metadata=new_metadata)
