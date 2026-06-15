"""Configurazione pytest: rende importabile il package da src/ e fornisce i fake.

I fake implementano le interfacce di dominio così che l'orchestratore possa
essere testato in isolamento, senza chiamate API reali né ChromaDB su disco.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json  # noqa: E402

from sdcc_rag.domain.interfaces import (  # noqa: E402
    IEmbeddingProvider,
    ILLMProvider,
    IVectorStore,
)
from sdcc_rag.domain.models import (  # noqa: E402
    Chunk,
    EmbeddedChunk,
    RetrievedChunk,
)


class FakeEmbeddingProvider(IEmbeddingProvider):
    """Embedding deterministici e privi di rete: [len(text), n_chiamata]."""

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = dimension
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(t)), float(self.calls)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class FakeVectorStore(IVectorStore):
    """Store in memoria, indicizzato per chunk_id (upsert idempotente).

    `query` restituisce `RetrievedChunk` con uno score uniforme `default_score`
    (1.0 = massima rilevanza), così i test della soglia possono iniettare valori
    sotto/sopra `min_score`.
    """

    def __init__(self, default_score: float = 1.0) -> None:
        self.items: dict[str, EmbeddedChunk] = {}
        self._default_score = default_score

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        for ec in embedded_chunks:
            self.items[ec.chunk.chunk_id] = ec

    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk=ec.chunk, score=self._default_score)
            for ec in list(self.items.values())[:top_k]
        ]

    def count(self) -> int:
        return len(self.items)


class FakeLLMProvider(ILLMProvider):
    """LLM finto: ritorna un JSON canned con summary e keywords."""

    def __init__(self, summary: str = "riassunto", keywords: list[str] | None = None) -> None:
        self._payload = {"summary": summary, "keywords": keywords or ["alpha", "beta"]}

    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        return json.dumps(self._payload)


class BrokenLLMProvider(ILLMProvider):
    """LLM che fallisce sempre: per testare il degrado controllato dell'enricher."""

    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        raise RuntimeError("LLM non raggiungibile")


class EchoLLMProvider(ILLMProvider):
    """LLM finto per il lato generazione: cattura ciò che riceve e ritorna una
    risposta canned. Permette di asserire che system prompt e contesto siano
    effettivamente passati al modello, e di contare le chiamate (empty-guard)."""

    def __init__(self, answer: str = "risposta") -> None:
        self._answer = answer
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system
        return self._answer
