"""Composition root della pipeline di ingestion.

È l'unico modulo che conosce le classi concrete: costruisce le dipendenze a
partire dalla configurazione, le inietta nell'orchestratore e avvia l'ingestion.

Uso:
    python scripts/ingest.py [PATH] [--title T] [--author A] [--meta KEY=VALUE ...]

Se PATH è omesso, usa `Settings.data_path` (default ./data). I metadati manuali
(--title/--author/--meta) sono costanti per l'intero run e vengono iniettati in
ogni documento durante l'arricchimento.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rende importabile il package `sdcc_rag` da src/ senza installazione.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdcc_rag.config import Settings  # noqa: E402
from sdcc_rag.embeddings.factory import create_embedding_provider  # noqa: E402
from sdcc_rag.enrichment.composite_enricher import CompositeMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.manual_enricher import ManualMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.semantic_enricher import SemanticMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.standard_enricher import StandardMetadataEnricher  # noqa: E402
from sdcc_rag.ingestion.orchestrator import IngestionOrchestrator  # noqa: E402
from sdcc_rag.llm.factory import create_llm_provider  # noqa: E402
from sdcc_rag.loaders.json_loader import JsonLoader  # noqa: E402
from sdcc_rag.loaders.text_loader import TextLoader  # noqa: E402
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter  # noqa: E402
from sdcc_rag.stores.factory import create_vector_store  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion pipeline RAG SDCC")
    parser.add_argument("path", nargs="?", help="cartella da ingerire (default: DATA_PATH)")
    parser.add_argument("--title", help="metadato manuale: titolo")
    parser.add_argument("--author", help="metadato manuale: autore")
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metadato manuale arbitrario (ripetibile)",
    )
    return parser.parse_args()


def _manual_metadata(args: argparse.Namespace) -> dict[str, str]:
    manual: dict[str, str] = {}
    if args.title:
        manual["title"] = args.title
    if args.author:
        manual["author"] = args.author
    for item in args.meta:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--meta atteso nel formato KEY=VALUE, ricevuto: {item!r}")
        manual[key.strip()] = value.strip()
    return manual


def main() -> None:
    args = _parse_args()
    settings = Settings()

    loaders = [TextLoader(), JsonLoader(text_field=settings.json_text_field)]
    splitter = RecursiveCharacterSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    embedder = create_embedding_provider(settings)
    store = create_vector_store(settings)

    # Catena di arricchimento: tracciabilità → semantico (LLM) → manuale (utente).
    llm = create_llm_provider(settings)
    enricher = CompositeMetadataEnricher(
        [
            StandardMetadataEnricher(),
            SemanticMetadataEnricher(llm, max_chars=settings.semantic_max_chars),
            ManualMetadataEnricher(_manual_metadata(args)),
        ]
    )

    orchestrator = IngestionOrchestrator(
        loaders=loaders,
        splitter=splitter,
        embedder=embedder,
        store=store,
        enricher=enricher,
        batch_size=settings.batch_size,
    )

    source = settings.document_source.lower()
    if source == "azure":
        documents = _load_azure_documents(settings, loaders)
        report = orchestrator.ingest_documents(documents)
    elif source == "local":
        root = Path(args.path) if args.path else Path(settings.data_path)
        report = orchestrator.ingest_path(root)
    else:
        raise SystemExit(
            f"DOCUMENT_SOURCE sconosciuto: {settings.document_source!r} "
            "(valori ammessi: 'local', 'azure')"
        )

    print(report)
    print(f"  chunk totali nello store: {store.count()}")


def _load_azure_documents(settings: Settings, loaders: list) -> list:
    """Scarica e parsa tutti i blob del container configurato.

    Import lazy di Azure: il flusso locale non richiede la libreria. La fase di
    split/enrich/embed/upsert resta condivisa con l'ingestion locale tramite
    `IngestionOrchestrator.ingest_documents`.
    """
    from azure.storage.blob import BlobServiceClient

    from sdcc_rag.loaders.azure_blob_loader import AzureBlobDocumentLoader
    from sdcc_rag.loaders.registry import LoaderRegistry

    # Valida le credenziali Azure (ValueError se mancanti) e riusa i loader locali.
    azure_loader = AzureBlobDocumentLoader(settings, LoaderRegistry(loaders))
    service = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )
    container = service.get_container_client(settings.azure_storage_container_name)

    documents = []
    for blob in container.list_blobs():
        try:
            documents.extend(azure_loader.load(Path(blob.name)))
        except Exception as exc:  # un blob rotto non deve fermare la pipeline
            print(f"  blob saltato {blob.name}: {exc}")
    return documents


if __name__ == "__main__":
    main()
