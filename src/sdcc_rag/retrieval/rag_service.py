"""Orchestratore della fase di query (Retrieval-Augmented Generation).

Controller a DI pura, simmetrico a `IngestionOrchestrator`: dipende solo dai
port (`IEmbeddingProvider`, `IVectorStore`, `ILLMProvider`), mai da classi
concrete. Flusso:

    domanda → embed_query → store.query(top_k) → filtro per soglia di rilevanza
    → (se resta contesto) generazione fondata via LLM, altrimenti astensione.

Anti-allucinazione: l'empty-guard non chiama mai l'LLM senza contesto, e la
generazione separa policy (system) e dati (user) tramite `retrieval/prompt.py`.
"""

from __future__ import annotations

from sdcc_rag.domain.interfaces import IEmbeddingProvider, ILLMProvider, IVectorStore
from sdcc_rag.domain.models import Answer
from sdcc_rag.retrieval.prompt import ABSTENTION_TEXT, SYSTEM_PROMPT, build_user_prompt


class RAGService:
    def __init__(
        self,
        embedder: IEmbeddingProvider,
        store: IVectorStore,
        llm: ILLMProvider,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str) -> Answer:
        vector = self._embedder.embed_query(question)
        retrieved = self._store.query(vector, self._top_k)

        # Soglia di rilevanza: scarta i chunk troppo distanti dalla query.
        # La policy vive nel controller; lo store si limita a riportare lo score.
        kept = [r.chunk for r in retrieved if r.score >= self._min_score]

        # Empty-guard: senza contesto non si interroga l'LLM (niente allucinazioni).
        if not kept:
            return Answer(question=question, text=ABSTENTION_TEXT, sources=[], chunks=[])

        text = self._llm.complete(
            build_user_prompt(question, kept),
            system=SYSTEM_PROMPT,
            json_output=False,
        )

        # Fonti allineate ai passaggi [i]: nessuna deduplica → mappa 1:1 con kept.
        sources = [chunk.source for chunk in kept]
        return Answer(question=question, text=text, sources=sources, chunks=kept)
