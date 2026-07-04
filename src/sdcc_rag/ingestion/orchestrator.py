"""Controller della pipeline di ingestion.

Riceve le dipendenze già costruite (DI puro): non importa né istanzia alcuna
classe concreta. Cammina la directory, instrada ogni file al loader giusto,
spezza i documenti, vettorizza a batch e persiste, accumulando un report.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from sdcc_rag.domain.interfaces import (
    IDocumentLoader,
    IDocumentSplitter,
    IEmbeddingProvider,
    IMetadataEnricher,
    IVectorStore,
)
from sdcc_rag.domain.models import Document, EmbeddedChunk, IngestionReport
from sdcc_rag.loaders.registry import LoaderRegistry

T = TypeVar("T")


def _batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class IngestionOrchestrator:
    def __init__(
        self,
        loaders: list[IDocumentLoader],
        splitter: IDocumentSplitter,
        embedder: IEmbeddingProvider,
        store: IVectorStore,
        enricher: IMetadataEnricher,
        batch_size: int = 64,
    ) -> None:
        self._registry = LoaderRegistry(loaders)
        self._splitter = splitter
        self._embedder = embedder
        self._store = store
        self._enricher = enricher
        self._batch_size = batch_size

    def ingest_path(self, root: Path) -> IngestionReport:
        report = IngestionReport()
        self._process_documents(self._load_documents(root, report), report)
        return report

    def ingest_documents(self, documents: Iterable[Document]) -> IngestionReport:
        """Processa documenti già caricati (es. scaricati da Azure Blob Storage).

        Punto d'ingresso alternativo a `ingest_path`: condivide lo stesso ciclo di
        split/enrich/embed/upsert, così la sorgente (filesystem o cloud) non cambia
        di una riga la logica di indicizzazione.
        """
        report = IngestionReport()
        self._process_documents(documents, report)
        return report

    def sync_and_ingest(self, source_documents: list[Document]) -> IngestionReport:
        """Ingestion con allineamento del DB alla sorgente ("Sync & Purge").

        Risolve l'anomalia dei *chunk orfani*: poiché il `chunk_id` è l'hash del
        contenuto (idempotenza), modificare o eliminare un documento sorgente
        lascerebbe nel vector store i chunk della versione precedente. Due fasi:

        1. **Purge**: elimina dallo store i documenti non più presenti nella sorgente.
        2. **Ingest con purge-first**: prima di re-indicizzare ogni documento fa
           "piazza pulita" delle sue versioni precedenti (`delete_by_doc_id`).

        `source_documents` deve rappresentare l'INTERO corpus: i documenti indicizzati
        ma assenti da questa lista vengono considerati orfani ed eliminati.
        """
        report = IngestionReport()
        source_ids = {document.source for document in source_documents}
        for orphan in self._store.get_all_doc_ids() - source_ids:
            self._store.delete_by_doc_id(orphan)
            report.documents_pruned += 1
        self._process_documents(source_documents, report, purge_first=True)
        return report

    def sync_and_ingest_path(self, root: Path) -> IngestionReport:
        """Variante di `sync_and_ingest` per la sorgente filesystem.

        Materializza l'intero corpus sotto `root` (necessario al purge degli orfani,
        che confronta la sorgente completa con il DB) riusando lo stesso caricamento
        e accounting `files_skipped`/`errors` di `ingest_path`.
        """
        report = IngestionReport()
        documents = self._load_documents(root, report)
        source_ids = {document.source for document in documents}
        for orphan in self._store.get_all_doc_ids() - source_ids:
            self._store.delete_by_doc_id(orphan)
            report.documents_pruned += 1
        self._process_documents(documents, report, purge_first=True)
        return report

    def _load_documents(self, root: Path, report: IngestionReport) -> list[Document]:
        """Cammina `root`, instrada ogni file al loader e restituisce i Document.

        Accumula in `report` i file saltati (estensione non gestita) e gli errori di
        caricamento, così un file rotto non ferma la pipeline.
        """
        documents: list[Document] = []
        for file in self._discover_files(root):
            loader = self._registry.resolve(file)
            if loader is None:
                report.files_skipped.append(str(file))
                continue
            try:
                documents.extend(loader.load(file))
            except Exception as exc:  # un file rotto non deve fermare la pipeline
                report.errors.append(f"{file}: {exc}")
        return documents

    def _process_documents(
        self,
        documents: Iterable[Document],
        report: IngestionReport,
        purge_first: bool = False,
    ) -> None:
        for document in documents:
            try:
                # piazza pulita delle versioni precedenti (Sync & Purge): rimuove i
                # chunk-hash obsoleti PRIMA del re-upsert. `source` è l'identità stabile
                # del documento (gli enricher non lo toccano).
                if purge_first:
                    self._store.delete_by_doc_id(document.source)
                # arricchimento doc-level (tracciabilità + semantico + manuale);
                # i campi scendono nei chunk via splitter
                document = self._enricher.enrich_document(document)
                chunks = self._splitter.split(document)
                # arricchimento chunk-level (es. conteggi): non altera chunk_id
                chunks = [self._enricher.enrich_chunk(chunk) for chunk in chunks]
                for batch in _batched(chunks, self._batch_size):
                    texts = [chunk.text for chunk in batch]
                    vectors = self._embedder.embed_documents(texts)
                    embedded = [
                        EmbeddedChunk(chunk=chunk, embedding=vector)
                        for chunk, vector in zip(batch, vectors)
                    ]
                    self._store.upsert(embedded)
                    report.chunks_indexed += len(embedded)
                report.documents_loaded += 1
            except Exception as exc:  # un documento problematico non ferma la pipeline
                report.errors.append(f"{document.source}: {exc}")
                continue

    @staticmethod
    def _discover_files(root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(p for p in root.rglob("*") if p.is_file())
