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
class EmbeddedChunk:
    """Chunk con il relativo vettore di embedding."""

    chunk: Chunk
    embedding: list[float]


@dataclass
class IngestionReport:
    """Riepilogo accumulato durante un'esecuzione di ingestion."""

    documents_loaded: int = 0
    chunks_indexed: int = 0
    files_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # output human-friendly per la CLI
        lines = [
            "Ingestion report",
            f"  documenti caricati : {self.documents_loaded}",
            f"  chunk indicizzati  : {self.chunks_indexed}",
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
