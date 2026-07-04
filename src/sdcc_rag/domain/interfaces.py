"""Contratti astratti (porte) della pipeline.

Ogni componente concreto implementa una di queste interfacce; l'orchestratore e
la composition root dipendono solo da questi tipi astratti (inversione delle dipendenze).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sdcc_rag.domain.models import (
    AutomaticMetadata,
    Chunk,
    Document,
    EmbeddedChunk,
    RetrievedChunk,
)


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
    def query(
        self, embedding: list[float], top_k: int = 5, query_text: str | None = None
    ) -> list[RetrievedChunk]:
        """Restituisce i chunk più simili al vettore di query, con score di rilevanza.

        `query_text`, se fornito, è il testo grezzo della domanda: gli store che lo
        supportano (es. Azure AI Search) lo usano per una ricerca *ibrida*
        (lessicale + vettoriale). Gli store puramente vettoriali (es. Chroma) lo
        ignorano. Opzionale e retro-compatibile.
        """

    @abstractmethod
    def count(self) -> int:
        """Numero di chunk attualmente indicizzati."""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Elimina tutti i chunk appartenenti a un documento.

        `doc_id` coincide con `Document.source` (l'identità che deriva i `chunk_id`).
        Serve a fare "piazza pulita" delle versioni precedenti di un documento
        prima di re-indicizzarlo, così i chunk-hash obsoleti non restano orfani.
        """

    @abstractmethod
    def get_all_doc_ids(self) -> set[str]:
        """Insieme dei `doc_id` (== `Document.source`) distinti presenti nello store.

        Usato dalla fase di *purge* per individuare i documenti ancora indicizzati
        ma non più presenti nella sorgente, da eliminare via `delete_by_doc_id`.
        """


class ILLMProvider(ABC):
    """Modello di linguaggio per l'arricchimento semantico (summary, keywords)."""

    @abstractmethod
    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        """Esegue il prompt e restituisce il testo generato.

        `system`, se fornito, è il system message che definisce ruolo e regole del
        modello (separa la *policy* dai *dati* nel prompt utente). Se `json_output`
        è True il provider chiede al modello una risposta in formato JSON (il
        parsing resta a carico del chiamante).
        """


class IMetadataExtractor(ABC):
    """Estrae metadati automatici dal testo di un documento tramite LLM.

    Porta passiva: l'implementazione concreta (es. `AzureOpenAIMetadataExtractor`)
    invia il testo a un modello con output strutturato e restituisce un
    `AutomaticMetadata`. Degrado best-effort a carico dell'implementazione (un
    errore non deve fermare la pipeline).
    """

    @abstractmethod
    def extract(self, text: str) -> AutomaticMetadata:
        """Analizza il testo e restituisce i metadati automatici estratti."""


class IMetadataEnricher(ABC):
    """Arricchisce i metadati di Document/Chunk durante l'ingestion.

    Tocca esclusivamente il campo `metadata`: non altera mai testo/source/indice,
    così l'id deterministico dei chunk (e quindi l'idempotenza) resta invariato.
    """

    @abstractmethod
    def enrich_document(self, document: Document) -> Document:
        """Restituisce una nuova copia del documento con metadata arricchiti."""

    @abstractmethod
    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """Restituisce una nuova copia del chunk con metadata arricchiti."""
