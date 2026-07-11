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

import re

from sdcc_rag.domain.interfaces import IEmbeddingProvider, ILLMProvider, IVectorStore
from sdcc_rag.domain.models import Answer, Chunk
from sdcc_rag.retrieval.prompt import ABSTENTION_TEXT, SYSTEM_PROMPT, build_user_prompt

# Rete di sicurezza quando l'LLM NON produce alcuna citazione [n] valida: se True
# si mostra SOLO la fonte più rilevante (top-1), non tutte quelle recuperate — così
# la lista fonti non torna a esporre il rumore. Metti False per non mostrare nulla.
FALLBACK_SHOW_TOP1 = True

# Gruppo di citazione: parentesi quadre che contengono SOLO cifre/virgole/spazi.
# Matcha [1], [1,2], [1, 2] e — per adiacenza — [1][2] (due match separati). Esclude
# per costruzione i link markdown testuali tipo [nota] (niente cifre → nessun match).
_CITATION_GROUP = re.compile(r"\[([\d\s,]+)\]")


def _parse_citations(text: str, n: int) -> list[int]:
    """Indici [n] validi (1..n) nell'ordine di PRIMA citazione, deduplicati."""
    order: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for token in group.split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= n and idx not in order:
                    order.append(idx)
    return order


def _renumber_citations(text: str, mapping: dict[int, int]) -> str:
    """Riscrive gli [n] del testo secondo `mapping` (vecchio→nuovo).

    I riferimenti puntano agli stessi chunk di prima (solo il numero cambia), così
    testo e lista fonti restano coerenti con la numerazione sequenziale filtrata.
    Un gruppo che non contiene alcun indice mappato viene rimosso.
    """

    def _repl(match: re.Match) -> str:
        remapped: list[int] = []
        for token in match.group(1).split(","):
            token = token.strip()
            if token.isdigit() and int(token) in mapping:
                new = mapping[int(token)]
                if new not in remapped:
                    remapped.append(new)
        if not remapped:
            return ""
        return "[" + ", ".join(str(x) for x in remapped) + "]"

    return _CITATION_GROUP.sub(_repl, text)


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
        # `query_text=question` abilita l'hybrid search sugli store che la supportano
        # (Azure AI Search: lessicale + vettoriale); gli store vettoriali lo ignorano.
        retrieved = self._store.query(vector, self._top_k, query_text=question)

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

        # Fonti = SOLO i passaggi che l'LLM ha davvero citato con [n] (anti-rumore):
        # i chunk recuperati ma non citati sono similarità senza pertinenza e non
        # devono comparire in bibliografia.
        cited = _parse_citations(text, len(kept))
        if cited:
            # Rinumerazione sequenziale in ordine di citazione: la lista filtrata parte
            # da [1] e il testo viene rimappato di conseguenza (approccio self-contained,
            # nessun cambio alla numerazione posizionale della UI). Vedi delivery note.
            mapping = {old: new for new, old in enumerate(cited, start=1)}
            selected: list[Chunk] = [kept[old - 1] for old in cited]
            text = _renumber_citations(text, mapping)
        elif FALLBACK_SHOW_TOP1:
            # Nessuna citazione valida: mostra solo il chunk più rilevante. `kept`
            # conserva l'ordine dello store (score decrescente) → kept[0] = top-1.
            selected = [kept[0]]
        else:
            selected = []

        sources = [chunk.source for chunk in selected]
        return Answer(question=question, text=text, sources=sources, chunks=selected)
