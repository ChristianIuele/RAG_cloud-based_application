"""Adapter enricher: collega un `IMetadataExtractor` alla catena di arricchimento.

Ponte fra la porta `IMetadataExtractor` (che restituisce un `AutomaticMetadata`
tipizzato) e la porta `IMetadataEnricher` su cui dipende l'orchestratore. Estrae i
metadati automatici a livello di documento e li appiattisce nel `dict` dei metadata.

Vincolo scalari (Chroma): le liste (`keywords`, `suggested_categories`, `entities`)
vengono serializzate in stringhe join-virgola, così sopravvivono al filtro di
`ChromaVectorStore._metadata` e restano coerenti anche su Azure AI Search.
"""

from __future__ import annotations

import dataclasses

from sdcc_rag.domain.interfaces import IMetadataEnricher, IMetadataExtractor
from sdcc_rag.domain.models import Chunk, Document


class ExtractorMetadataEnricher(IMetadataEnricher):
    def __init__(self, extractor: IMetadataExtractor) -> None:
        self._extractor = extractor

    def enrich_document(self, document: Document) -> Document:
        auto = self._extractor.extract(document.content)

        enriched: dict[str, object] = {}
        if auto.summary:
            enriched["summary"] = auto.summary
        if auto.language:
            enriched["language"] = auto.language
        # liste → stringa scalare (i valori metadata devono essere scalari)
        if auto.keywords:
            enriched["keywords"] = ", ".join(auto.keywords)
        if auto.suggested_categories:
            enriched["suggested_categories"] = ", ".join(auto.suggested_categories)
        if auto.entities:
            enriched["entities"] = ", ".join(auto.entities)

        if not enriched:
            return document
        return dataclasses.replace(document, metadata={**document.metadata, **enriched})

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        return chunk  # l'estrazione automatica è a livello di documento
