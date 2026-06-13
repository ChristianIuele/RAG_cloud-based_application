"""Controller della pipeline di ingestion.

Riceve le dipendenze già costruite (DI puro): non importa né istanzia alcuna
classe concreta. Cammina la directory, instrada ogni file al loader giusto,
spezza i documenti, vettorizza a batch e persiste, accumulando un report.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from sdcc_rag.domain.interfaces import (
    IDocumentLoader,
    IDocumentSplitter,
    IEmbeddingProvider,
    IVectorStore,
)
from sdcc_rag.domain.models import EmbeddedChunk, IngestionReport
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
        batch_size: int = 64,
    ) -> None:
        self._registry = LoaderRegistry(loaders)
        self._splitter = splitter
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def ingest_path(self, root: Path) -> IngestionReport:
        report = IngestionReport()

        for file in self._discover_files(root):
            loader = self._registry.resolve(file)
            if loader is None:
                report.files_skipped.append(str(file))
                continue

            try:
                documents = loader.load(file)
            except Exception as exc:  # un file rotto non deve fermare la pipeline
                report.errors.append(f"{file}: {exc}")
                continue

            for document in documents:
                chunks = self._splitter.split(document)
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

        return report

    @staticmethod
    def _discover_files(root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(p for p in root.rglob("*") if p.is_file())
