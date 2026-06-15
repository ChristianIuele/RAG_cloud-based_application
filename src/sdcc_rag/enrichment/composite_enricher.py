"""Composite di enricher: applica una catena di IMetadataEnricher in ordine.

Permette all'orchestratore di dipendere da un singolo IMetadataEnricher pur
combinando più strategie (tracciabilità + semantico + manuale). Gli enricher
successivi vedono i metadati prodotti dai precedenti (merge progressivo).
"""

from __future__ import annotations

from sdcc_rag.domain.interfaces import IMetadataEnricher
from sdcc_rag.domain.models import Chunk, Document


class CompositeMetadataEnricher(IMetadataEnricher):
    def __init__(self, enrichers: list[IMetadataEnricher]) -> None:
        self._enrichers = enrichers

    def enrich_document(self, document: Document) -> Document:
        for enricher in self._enrichers:
            document = enricher.enrich_document(document)
        return document

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        for enricher in self._enrichers:
            chunk = enricher.enrich_chunk(chunk)
        return chunk
