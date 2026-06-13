"""Contratti astratti (porte) della pipeline.

Ogni componente concreto implementa una di queste interfacce; l'orchestratore e
la composition root dipendono solo da questi tipi astratti (inversione delle dipendenze).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sdcc_rag.domain.models import Chunk, Document, EmbeddedChunk


class IDocumentLoader(ABC):
    """Carica un file di un formato specifico in uno o più Document."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Estensioni gestite, in minuscolo e col punto (es. {".txt", ".md"})."""

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Legge il file e restituisce i Document estratti."""


class IDocumentSplitter(ABC):
    """Suddivide un Document in Chunk. Disaccoppiato dalla strategia concreta."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Restituisce i chunk del documento con id deterministici."""


class IEmbeddingProvider(ABC):
    """Vettorizza testo. Default OpenAI, switch ad Azure via env (stessa interfaccia)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vettorizza un batch di testi (un vettore per testo, stesso ordine)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Vettorizza una singola query (lato retrieval)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensione dei vettori prodotti."""


class IVectorStore(ABC):
    """Persistenza/recupero vettoriale. Implementazione di default: ChromaDB."""

    @abstractmethod
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Inserisce/aggiorna i chunk usando chunk_id come chiave (idempotente)."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Restituisce i chunk più simili al vettore di query."""

    @abstractmethod
    def count(self) -> int:
        """Numero di chunk attualmente indicizzati."""
