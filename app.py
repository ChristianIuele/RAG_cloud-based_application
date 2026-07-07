"""Frontend Streamlit del sistema RAG SDCC.

Seconda *composition root* accanto a `scripts/ingest.py` e `scripts/query.py`:
è l'unico posto (con quelle) che conosce le classi concrete del backend e le
cabla insieme. Espone due funzioni all'utente:

- **Ingestione** (sidebar): upload di un file txt/md/json + metadati manuali. Il
  file crudo viene prima salvato su Azure Blob Storage (RF-004) e poi riletto dal
  blob e indicizzato (parsing → chunking → embedding → upsert nel vector store).
- **Chat RAG** (area principale): domanda → recupero dei chunk → risposta fondata
  con citazioni tracciabili alle fonti (RF-054).

Il backend è config-driven: embedding provider, vector store e LLM sono scelti da
`.env` tramite le factory. L'upload su Blob è sempre attivo (RF-004): se le
credenziali Azure Storage mancano, l'ingestione si ferma con un errore bloccante.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Rende importabile il package `sdcc_rag` da src/ senza installazione (come scripts/*.py).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sdcc_rag.config import Settings  # noqa: E402
from sdcc_rag.embeddings.factory import create_embedding_provider  # noqa: E402
from sdcc_rag.enrichment.azure_metadata_extractor import AzureOpenAIMetadataExtractor  # noqa: E402
from sdcc_rag.enrichment.composite_enricher import CompositeMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.extractor_enricher import ExtractorMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.manual_enricher import ManualMetadataEnricher  # noqa: E402
from sdcc_rag.enrichment.standard_enricher import StandardMetadataEnricher  # noqa: E402
from sdcc_rag.ingestion.orchestrator import IngestionOrchestrator  # noqa: E402
from sdcc_rag.llm.factory import create_llm_provider  # noqa: E402
from sdcc_rag.loaders.azure_blob_loader import AzureBlobDocumentLoader  # noqa: E402
from sdcc_rag.loaders.json_loader import JsonLoader  # noqa: E402
from sdcc_rag.loaders.registry import LoaderRegistry  # noqa: E402
from sdcc_rag.loaders.text_loader import TextLoader  # noqa: E402
from sdcc_rag.retrieval.rag_service import RAGService  # noqa: E402
from sdcc_rag.splitters.recursive_splitter import RecursiveCharacterSplitter  # noqa: E402
from sdcc_rag.stores.factory import create_vector_store  # noqa: E402

# --- Direttive di sicurezza -------------------------------------------------
# AB-01: solo queste estensioni sono accettate dall'uploader (whitelist a UI).
ALLOWED_EXTENSIONS = ["txt", "md", "json"]
# AB-02: dimensione massima del file caricato (5 MB).
MAX_FILE_MB = 5
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024


# --- Risorse condivise (costruite una sola volta) ---------------------------
@st.cache_resource
def get_settings() -> Settings:
    """Configurazione centralizzata, letta da `.env` (cache per sessione app)."""
    return Settings()


@st.cache_resource
def get_shared() -> dict:
    """Costruisce le dipendenze costose/singleton e le mette in cache.

    Sono riusate sia dall'ingestione sia dalla chat: embedder e store devono
    essere gli STESSI (stesso spazio vettoriale) tra le due fasi. `loaders` e
    `splitter` sono stateless e condivisi; l'orchestrator NON è qui perché va
    ricostruito a ogni ingestione con i metadati manuali del singolo file.
    """
    settings = get_settings()
    embedder = create_embedding_provider(settings)  # stesso provider per ingest e query
    store = create_vector_store(settings)
    llm = create_llm_provider(settings)
    loaders = [TextLoader(), JsonLoader(text_field=settings.json_text_field)]
    splitter = RecursiveCharacterSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    service = RAGService(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
    )
    return {
        "settings": settings,
        "embedder": embedder,
        "store": store,
        "llm": llm,
        "loaders": loaders,
        "splitter": splitter,
        "service": service,
    }


def _build_orchestrator(shared: dict, manual: dict[str, str]) -> IngestionOrchestrator:
    """Ricostruisce l'orchestrator con i metadati manuali del file corrente.

    Riusa embedder/store (dalla cache) ma inietta un `ManualMetadataEnricher`
    fresco: la catena replica quella di `scripts/ingest.py` (tracciabilità →
    automatico via LLM → manuale, applicato per ultimo così l'intento dell'utente
    vince sulle chiavi omonime).
    """
    settings = shared["settings"]
    enricher = CompositeMetadataEnricher(
        [
            StandardMetadataEnricher(),
            ExtractorMetadataEnricher(AzureOpenAIMetadataExtractor(settings)),
            ManualMetadataEnricher(manual),
        ]
    )
    return IngestionOrchestrator(
        loaders=shared["loaders"],
        splitter=shared["splitter"],
        embedder=shared["embedder"],
        store=shared["store"],
        enricher=enricher,
        batch_size=settings.batch_size,
    )


def _manual_metadata(
    title: str, author: str, category: str, description: str, tags: str
) -> dict[str, str]:
    """Costruisce il dict di metadati manuali (replica `ingest.py:_manual_metadata`).

    Scarta i campi vuoti; i tag (separati da virgola) sono normalizzati a una
    singola stringa scalare join-virgola, coerente con il vincolo di metadati
    scalari dei vector store.
    """
    manual: dict[str, str] = {}
    if title.strip():
        manual["title"] = title.strip()
    if author.strip():
        manual["author"] = author.strip()
    if category.strip():
        manual["category"] = category.strip()
    if description.strip():
        manual["description"] = description.strip()
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
    if parsed_tags:
        manual["tags"] = ", ".join(parsed_tags)
    return manual


def _upload_raw_to_blob(settings: Settings, blob_name: str, data: bytes) -> None:
    """Salva il file crudo su Azure Blob Storage (RF-004).

    Nessun codice di upload esiste nel backend (solo lettura): riusa lo stesso
    pattern di connessione dell'`AzureBlobDocumentLoader` (connection string +
    container name da `Settings`). Import lazy della libreria Azure.
    """
    missing = [
        name
        for name, value in {
            "AZURE_STORAGE_CONNECTION_STRING": settings.azure_storage_connection_string,
            "AZURE_STORAGE_CONTAINER_NAME": settings.azure_storage_container_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Credenziali Azure Storage mancanti per l'upload su Blob (RF-004): "
            + ", ".join(missing)
        )

    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )
    container = service.get_container_client(settings.azure_storage_container_name)
    # overwrite=True: re-caricare lo stesso nome aggiorna il blob (idempotenza col
    # chunk_id derivato dal nome-blob stabile a valle).
    container.upload_blob(name=blob_name, data=data, overwrite=True)


def _ingest_from_blob(shared: dict, blob_name: str, manual: dict[str, str]):
    """Rilegge il blob appena caricato e lo indicizza.

    Usa `AzureBlobDocumentLoader`, che riscrive la provenance sul nome-blob
    stabile: la stessa idempotenza del percorso `DOCUMENT_SOURCE=azure`. I
    documenti risultanti passano per il ciclo split/enrich/embed/upsert condiviso.
    """
    settings = shared["settings"]
    registry = LoaderRegistry(shared["loaders"])
    # Il costruttore valida le credenziali Azure (ValueError se assenti).
    blob_loader = AzureBlobDocumentLoader(settings, registry)
    documents = blob_loader.load(Path(blob_name))

    orchestrator = _build_orchestrator(shared, manual)
    return orchestrator.ingest_documents(documents)


# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="SDCC RAG", page_icon="📚", layout="wide")

shared = get_shared()
settings = shared["settings"]
service = shared["service"]
store = shared["store"]


# --- Sidebar: ingestione + scudo di sicurezza -------------------------------
with st.sidebar:
    st.header("📥 Ingestione documenti")

    uploaded = st.file_uploader(
        "Carica un documento",
        type=ALLOWED_EXTENSIONS,  # AB-01: whitelist estensioni
        help=f"Estensioni consentite: {', '.join(ALLOWED_EXTENSIONS)}. Max {MAX_FILE_MB} MB.",
    )

    # AB-02: controllo dimensione. Errore bloccante se il file supera il limite.
    if uploaded is not None and uploaded.size > MAX_FILE_BYTES:
        st.error(
            f"File troppo grande ({uploaded.size / 1024 / 1024:.1f} MB). "
            f"Limite massimo: {MAX_FILE_MB} MB."
        )
        st.stop()

    st.subheader("Metadati")
    title = st.text_input("Titolo")
    author = st.text_input("Autore")
    category = st.text_input("Categoria")
    description = st.text_area("Descrizione")
    tags = st.text_input("Tag (separati da virgola)", placeholder="sdcc, rag, azure")

    ingest_clicked = st.button("Ingerisci", type="primary", disabled=uploaded is None)

    if ingest_clicked and uploaded is not None:
        manual = _manual_metadata(title, author, category, description, tags)
        try:
            with st.spinner("Salvataggio del file crudo su Blob Storage (RF-004)…"):
                _upload_raw_to_blob(settings, uploaded.name, uploaded.getvalue())
            with st.spinner("Parsing, chunking, embedding e indicizzazione…"):
                report = _ingest_from_blob(shared, uploaded.name, manual)
        except ValueError as exc:
            # Credenziali Azure mancanti o estensione non risolvibile: errore chiaro.
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - superficie UI: mostra l'errore
            st.error(f"Ingestione fallita: {exc}")
        else:
            st.success(
                f"'{uploaded.name}' indicizzato: "
                f"{report.documents_loaded} documento/i, {report.chunks_indexed} chunk. "
                f"Totale chunk nello store: {store.count()}."
            )
            if report.errors:
                st.warning("Errori durante l'ingestione:\n" + "\n".join(report.errors))


# --- Area principale: chat RAG ----------------------------------------------
st.title("📚 SDCC RAG")
st.caption("Domande sul corpus SDCC — risposte fondate sui documenti indicizzati.")

# Storico conversazione in session_state (persiste tra i rerun della sessione).
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def _render_sources(sources: list[str], chunks: list) -> None:
    """Blocco espandibile con le citazioni ai chunk/documenti sorgente (RF-054)."""
    if not sources:
        return
    with st.expander(f"📎 Fonti e citazioni ({len(sources)})"):
        for i, source in enumerate(sources):
            chunk = chunks[i] if i < len(chunks) else None
            filename = chunk.metadata.get("filename", source) if chunk else source
            st.markdown(f"**[{i + 1}]** `{filename}` — {source}")
            if chunk is not None:
                # Anteprima del passaggio effettivamente passato all'LLM.
                snippet = chunk.text.strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "…"
                st.caption(snippet)


# Re-render dello storico a ogni ciclo.
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []), message.get("chunks", []))


# Input dell'utente.
if question := st.chat_input("Fai una domanda sul corpus…"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recupero dei chunk e generazione…"):
            answer = service.answer(question)
        st.markdown(answer.text)
        _render_sources(answer.sources, answer.chunks)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "chunks": answer.chunks,
        }
    )
