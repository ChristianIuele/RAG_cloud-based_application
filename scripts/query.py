"""Composition root della fase di query (REPL interattiva).

Unico modulo che conosce le classi concrete del lato retrieval: costruisce le
dipendenze dalla configurazione, le inietta nel `RAGService` e avvia un loop di
domande/risposte sul corpus già ingerito in ChromaDB.

Uso:
    python scripts/query.py [--top-k N] [--min-score F]

L'embedding provider è lo STESSO dell'ingestion (via factory): la query deve
vivere nello stesso spazio vettoriale dei chunk indicizzati.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rende importabile il package `sdcc_rag` da src/ senza installazione.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdcc_rag.config import Settings  # noqa: E402
from sdcc_rag.embeddings.factory import create_embedding_provider  # noqa: E402
from sdcc_rag.llm.factory import create_llm_provider  # noqa: E402
from sdcc_rag.retrieval.rag_service import RAGService  # noqa: E402
from sdcc_rag.stores.factory import create_vector_store  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query RAG SDCC (REPL)")
    parser.add_argument("--top-k", type=int, help="numero di chunk recuperati (default: config)")
    parser.add_argument(
        "--min-score", type=float, help="soglia di rilevanza coseno (default: config)"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()

    embedder = create_embedding_provider(settings)  # STESSO provider dell'ingestion
    store = create_vector_store(settings)
    llm = create_llm_provider(settings)

    # Precedenza: override CLI esplicito > neutro su Azure Search > default coseno.
    # Su Azure AI Search lo score è RRF (scala ~0.01-0.03), incompatibile con la soglia
    # coseno 0.3 (tarata su Chroma): applicarla azzererebbe i risultati. Il taglio di
    # rilevanza di base resta allo store (`azure_search_min_score`).
    if args.min_score is not None:
        query_min_score = args.min_score
    elif settings.vector_store == "azure_search":
        query_min_score = 0.0
    else:
        query_min_score = settings.retrieval_min_score

    service = RAGService(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k if args.top_k is not None else settings.retrieval_top_k,
        min_score=query_min_score,
    )

    print("Query RAG SDCC — scrivi una domanda (vuoto, 'exit' o Ctrl-D per uscire).")
    while True:
        try:
            question = input("domanda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break

        answer = service.answer(question)
        print(f"\n{answer.text}")
        if answer.sources:
            print("\nFonti:")
            for i, source in enumerate(answer.sources, start=1):
                print(f"  [{i}] {source}")
        print()


if __name__ == "__main__":
    main()
