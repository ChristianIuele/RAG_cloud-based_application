# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python RAG (Retrieval Augmented Generation) system over the SDCC corpus. The codebase
implements two phases: the **ingestion pipeline** (load TXT/Markdown/JSON documents, split
into chunks, embed, persist in ChromaDB) and the **query phase** (embed a question,
retrieve the most relevant chunks, generate a grounded answer with the LLM).

## Commands

```bash
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
copy .env.example .env                           # then fill in keys

python scripts/ingest.py [PATH] [--title T] [--author A] [--category C] \
    [--description D] [--tags "a,b,c"] [--meta KEY=VALUE ...] [--sync]   # ingestion (default ./data)
python scripts/query.py [--top-k N] [--min-score F]   # interactive query REPL
python infrastructure/setup_azure_search_index.py [--force]   # provision Azure AI Search index (VECTOR_STORE=azure_search)
pytest                                           # all tests
pytest tests/test_orchestrator.py::test_ingest_conta_documenti_e_chunk   # single test
```

## Architecture

Interface-driven (SOLID + dependency injection). Every pipeline stage is an abstract
interface in `src/sdcc_rag/domain/interfaces.py`, with interchangeable concrete impls:

- `IDocumentLoader` → `TextLoader` (.txt/.md), `JsonLoader` (.json) — routed by extension via `loaders/registry.py:LoaderRegistry`; `AzureBlobDocumentLoader` reads a container and **delegates parsing** to the local loaders via the registry
- `IDocumentSplitter` → `RecursiveCharacterSplitter` (self-contained, no LangChain)
- `IEmbeddingProvider` → `OpenAIEmbeddingProvider` / `AzureOpenAIEmbeddingProvider` / `OllamaEmbeddingProvider` (local, free) — selected via `embeddings/factory.py:create_embedding_provider`
- `IVectorStore` → `ChromaVectorStore` (local, default) / `AzureSearchVectorStore` (Azure AI Search) — selected via `stores/factory.py:create_vector_store`. Beyond `upsert`/`query`/`count`, the port exposes `delete_by_doc_id(doc_id)` and `get_all_doc_ids()` for the Sync & Purge flow (see design rules); `doc_id == Document.source`
- `IMetadataEnricher` → `CompositeMetadataEnricher` of `StandardMetadataEnricher` (traceability + counts) / `ExtractorMetadataEnricher` (adapter over `IMetadataExtractor`, flattens automatic fields) / `ManualMetadataEnricher` (user title/author/category/description/tags)
- `IMetadataExtractor` → `AzureOpenAIMetadataExtractor` — extracts `AutomaticMetadata` (summary, keywords, suggested_categories, language, entities) from the doc text via the `azure_metadata_deployment` (e.g. `gpt-5.4-nano`) in JSON mode; degrades to empty on error
- `ILLMProvider` → `OllamaLLMProvider` (local, free) / `AzureOpenAILLMProvider` (Azure OpenAI chat) — selected via `llm/factory.py:create_llm_provider`

Query phase (`retrieval/`): `rag_service.py:RAGService` is the query controller —
pure DI over the existing ports (`IEmbeddingProvider`, `IVectorStore`, `ILLMProvider`),
no concrete imports. It embeds the question, retrieves top-k chunks, drops those below
`retrieval_min_score`, and either generates a grounded answer or abstains. Prompt
construction lives in `retrieval/prompt.py` (pure, no LLM).

Data flows as immutable dataclasses (`domain/models.py`): `Document` → `Chunk` →
`EmbeddedChunk`; query side: `IVectorStore.query` returns `RetrievedChunk` (chunk + score)
and `RAGService` produces an `Answer` (text + per-passage `sources` + supporting chunks).

`ingestion/orchestrator.py:IngestionOrchestrator` is the controller — it receives all
dependencies pre-built (pure DI, no concrete imports inside). `scripts/ingest.py` is the
**composition root**: the only module that knows concrete classes and wires them together.
It branches on `DOCUMENT_SOURCE` (`local`|`azure`): `local` walks `data_path`;
`azure` lists the configured blob container, loads each blob through
`AzureBlobDocumentLoader`, and feeds the resulting documents to
`orchestrator.ingest_documents` (the split/enrich/embed/upsert stages are shared).
The `--sync` flag switches to the Sync & Purge variant (`sync_and_ingest_path` for
`local`, `sync_and_ingest` for `azure`); without it, the plain append/upsert flow runs.

### Key design rules

- **Provider/backend switches are config-only, one factory each**: `EMBEDDING_PROVIDER` (`openai`|`azure`|`ollama`) → `embeddings/factory.py`, `LLM_PROVIDER` (`ollama`|`azure`) → `llm/factory.py`, `VECTOR_STORE` (`chroma`|`azure_search`) → `stores/factory.py`, `DOCUMENT_SOURCE` (`local`|`azure`) → `scripts/ingest.py`. Each factory is the *only* place that imports the concrete impls (lazily, so an unused backend needs neither its SDK nor its credentials). Never hardcode a backend outside its factory. OpenAI/Azure embedding both use the `openai` SDK (`OpenAI` vs `AzureOpenAI`); on Azure the API "model" is the deployment name.
- **Embeddings are computed by the provider, not Chroma**: vectors are passed explicitly to `ChromaVectorStore`; do not enable Chroma's internal embedding_function.
- **Idempotent ingestion**: `domain/models.py:make_chunk_id` derives a deterministic SHA-256 id from `(source, index, text)`; the store uses `upsert` keyed on it, so re-ingesting updates instead of duplicating. Preserve this when changing chunking or storage. Because the id depends on `source`, `AzureBlobDocumentLoader` downloads each blob to a *random* temp file but then **rewrites provenance** (`source` + `metadata["filename"]`) back to the stable blob name — otherwise the temp path would break idempotency across runs.
- **Sync & Purge (orphan-chunk pruning)**: because `chunk_id` hashes the content, *editing* a source document changes its chunk ids and *deleting* it removes the source entirely — either way the old chunks would linger as **orphans**. `sync_and_ingest(source_documents)` (and `sync_and_ingest_path(root)` for local) fixes this in two phases: (1) **purge** — delete any `doc_id` in `get_all_doc_ids()` but not in the source set; (2) **ingest with purge-first** — `_process_documents(..., purge_first=True)` calls `delete_by_doc_id(document.source)` before re-upserting each doc, so stale versions never survive. `doc_id == Document.source` (unique per `Document`; JSON records are `path#index`). It is **opt-in via `--sync`** because purge deletes everything not in the current run — only safe on full-corpus runs, never a subfolder. Store mapping: Chroma filters the existing scalar `source` metadata (`where={"source": …}`); Azure needs a top-level **filterable+facetable `doc_id` field** (source lives buried in the non-filterable `metadata` JSON), so adding it to the index means re-provisioning (`infrastructure/setup_azure_search_index.py --force`) + re-ingest.
- **Two retrieval thresholds, different scales**: `RETRIEVAL_MIN_SCORE` is the store-agnostic filter applied by `RAGService`; `AZURE_SEARCH_MIN_SCORE` is applied *inside* `AzureSearchVectorStore` because Azure `@search.score` lives on a different scale than Chroma cosine distance. Keep them separate; don't collapse into one setting.
- **One file must not stop the pipeline**: loader errors and per-document errors are caught and recorded in `IngestionReport.errors`; unsupported extensions go to `files_skipped`.
- **Metadata enrichment is metadata-only**: enrichers return new dataclass copies (`dataclasses.replace`) and touch only `metadata`, never `text`/`source`/`chunk_index` — so `chunk_id` and idempotency hold even with volatile/non-deterministic values (timestamp, LLM output). Enriched values must be **scalar** (`str`/`int`/`float`/`bool`) to survive `ChromaVectorStore._metadata`; lists like `keywords` are joined into a string. Doc-level enrichment fans out to chunks via the splitter. `AzureOpenAIMetadataExtractor` degrades gracefully (LLM/JSON error → empty `AutomaticMetadata`, document unchanged).
- New concrete impls must subclass the matching ABC and be injected at the composition root — keep the orchestrator free of concrete imports.
- **Grounded generation (anti-hallucination)**: `RAGService` separates *policy* (a fixed `SYSTEM_PROMPT`) from *data* (the numbered retrieved passages in the user message). The system prompt forces answers to use only the context, mandates an exact abstention sentence (`ABSTENTION_TEXT`) when the answer isn't present, requires `[n]` citations, and treats context as data not instructions (prompt-injection hardening). The **empty-guard** returns the abstention answer *without calling the LLM* when no chunk survives retrieval/threshold. The query embedder must be the same provider used at ingestion (same vector space) — both go through `create_embedding_provider`.

## Conventions

Docstrings and inline comments are written in **Italian** (identifiers/APIs stay
English). Match this when adding code — keep new comments in Italian to read like the
surrounding source.

## Config

All settings live in `src/sdcc_rag/config.py:Settings` (pydantic-settings, reads `.env`,
env vars override). `data/`, `chroma_db/`, and `.env` are gitignored.

## Tests

Unit tests use fakes (`tests/conftest.py`: `FakeEmbeddingProvider`, `FakeVectorStore`) — no
real API calls, no on-disk Chroma. Both phases are covered in isolation: ingestion
(orchestrator, splitter, loaders, enrichment, factory) and query (`RAGService`, prompt).
Azure adapters (blob loader, search store, LLM, metadata extractor) are tested with injected
fakes / a fake `BlobServiceClient` — never the real SDK or network. `pythonpath = ["src"]`
is set in `pyproject.toml`.
