"""Enricher di tracciabilità (meccanico, senza dipendenze esterne).

Aggiunge metadati derivabili localmente: timestamp di ingestion, hash del
contenuto, componenti del path sorgente (doc-level) e conteggi (chunk-level).
Tutti i valori sono scalari, così sopravvivono al filtro di ChromaVectorStore.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sdcc_rag.domain.interfaces import IMetadataEnricher
from sdcc_rag.domain.models import Chunk, Document


class StandardMetadataEnricher(IMetadataEnricher):
    def enrich_document(self, document: Document) -> Document:
        # `source` può avere il suffisso "#index" dei record JSON: lo rimuovo
        # prima di derivare i componenti del path.
        raw_source = document.source.split("#", 1)[0]
        path = Path(raw_source)
        enriched = {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "content_sha256": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
            "source_stem": path.stem,
            "source_ext": path.suffix.lower(),
            "source_parent": path.parent.name,
        }
        return dataclasses.replace(document, metadata={**document.metadata, **enriched})

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        enriched = {
            "char_count": len(chunk.text),
            "word_count": len(chunk.text.split()),
        }
        return dataclasses.replace(chunk, metadata={**chunk.metadata, **enriched})
