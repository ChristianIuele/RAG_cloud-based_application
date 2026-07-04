"""Modelli di dominio della pipeline di ingestion.

Sono dataclass immutabili (`frozen=True`) per i tipi di valore che attraversano
la pipeline, e un report mutabile che l'orchestratore accumula durante l'esecuzione.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Documento grezzo caricato da un loader, prima dello splitting."""

    content: str
    source: str  # path/origine, usato per provenance e per derivare i chunk_id
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Frammento di documento pronto per la vettorizzazione."""

    text: str
    chunk_id: str  # id DETERMINISTICO: vedi make_chunk_id
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutomaticMetadata:
    """Metadati estratti automaticamente da un LLM sul testo del documento.

    Struttura tipizzata restituita da `IMetadataExtractor`. Vive solo al confine
    dell'extractor: l'adapter (`ExtractorMetadataEnricher`) la appiattisce nel
    `dict` scalare che la pipeline sa già persistere (liste → stringa join-virgola).
    """

    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    suggested_categories: list[str] = field(default_factory=list)
    language: str = ""
    entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddedChunk:
    """Chunk con il relativo vettore di embedding."""

    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk recuperato dal vector store con il suo punteggio di rilevanza."""

    chunk: Chunk
    score: float  # similarità coseno: più alto = più pertinente alla query


@dataclass(frozen=True)
class Answer:
    """Risposta generata dal sistema RAG, con le fonti a supporto.

    `sources` è allineato 1:1 ai passaggi usati nel prompt (citazioni `[i]`);
    `chunks` conserva l'evidenza effettivamente passata all'LLM.
    """

    question: str
    text: str
    sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class IngestionReport:
    """Riepilogo accumulato durante un'esecuzione di ingestion."""

    documents_loaded: int = 0
    chunks_indexed: int = 0
    documents_pruned: int = 0  # documenti orfani rimossi dal DB durante il sync
    files_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # output human-friendly per la CLI
        lines = [
            "Ingestion report",
            f"  documenti caricati : {self.documents_loaded}",
            f"  chunk indicizzati  : {self.chunks_indexed}",
            f"  documenti rimossi  : {self.documents_pruned}",
            f"  file saltati       : {len(self.files_skipped)}",
            f"  errori             : {len(self.errors)}",
        ]
        for err in self.errors:
            lines.append(f"    - {err}")
        return "\n".join(lines)


def make_chunk_id(source: str, index: int, text: str) -> str:
    """ID deterministico di un chunk: hash di (source, indice, testo).

    Garantisce l'idempotenza dell'ingestion: re-ingestare lo stesso contenuto
    produce gli stessi id, quindi lo store aggiorna invece di duplicare.
    """

    digest = hashlib.sha256(f"{source}\x00{index}\x00{text}".encode("utf-8"))
    return digest.hexdigest()
