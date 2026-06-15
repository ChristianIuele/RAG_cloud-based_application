"""Enricher manuale: inietta metadati forniti dall'utente (titolo, autore, ...).

I campi sono costanti per l'intera esecuzione di ingestion (passati dalla CLI).
Applicato per ultimo nella catena, così l'intento esplicito dell'utente prevale
su eventuali chiavi omonime già presenti.
"""

from __future__ import annotations

import dataclasses

from sdcc_rag.domain.interfaces import IMetadataEnricher
from sdcc_rag.domain.models import Chunk, Document


class ManualMetadataEnricher(IMetadataEnricher):
    def __init__(self, fields: dict[str, str] | None = None) -> None:
        # tiene solo i campi valorizzati (evita di scrivere None/"")
        self._fields = {k: v for k, v in (fields or {}).items() if v}

    def enrich_document(self, document: Document) -> Document:
        if not self._fields:
            return document
        return dataclasses.replace(document, metadata={**document.metadata, **self._fields})

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        return chunk  # i metadati manuali sono a livello di documento
