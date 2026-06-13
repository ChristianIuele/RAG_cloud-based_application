"""Composition root della pipeline di ingestion.

È l'unico modulo che conosce le classi concrete: costruisce le dipendenze a
partire dalla configurazione, le inietta nell'orchestratore e avvia l'ingestion
sulla cartella `data/`.

Uso:
    python scripts/ingest.py [PATH]

Se PATH è omesso, usa `Settings.data_path` (default ./data).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Rende importabile il package `sdcc_rag` da src/ senza installazione.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdcc_rag.config import Settings  # noqa: E402
from sdcc_rag.embeddings.factory import create_embedding_provider  # noqa: E402
from sdcc_rag.ingestion.orchestrator import IngestionOrchestrator  # noqa: E402
from sdcc_rag.loaders.json_loader import JsonLoader  # noqa: E402
from sdcc_rag.loaders.text_loader import TextLoader  # noqa: E402
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter  # noqa: E402
from sdcc_rag.stores.chroma_store import ChromaVectorStore  # noqa: E402


def main() -> None:
    settings = Settings()

    loaders = [TextLoader(), JsonLoader(text_field=settings.json_text_field)]
    splitter = RecursiveCharacterSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    embedder = create_embedding_provider(settings)
    store = ChromaVectorStore(settings)

    orchestrator = IngestionOrchestrator(
        loaders=loaders,
        splitter=splitter,
        embedder=embedder,
        store=store,
        batch_size=settings.batch_size,
    )

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(settings.data_path)
    report = orchestrator.ingest_path(root)

    print(report)
    print(f"  chunk totali nello store: {store.count()}")


if __name__ == "__main__":
    main()
