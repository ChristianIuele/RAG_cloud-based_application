# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python RAG (Retrieval Augmented Generation) system over the SDCC corpus. The current
codebase implements the **ingestion pipeline**: load TXT/Markdown/JSON documents, split
them into chunks, embed them, and persist them in ChromaDB.

## Commands

```bash
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
copy .env.example .env                           # then fill in keys

python scripts/ingest.py [PATH]                  # run ingestion (default ./data)
pytest                                           # all tests
pytest tests/test_orchestrator.py::test_ingest_conta_documenti_e_chunk   # single test
```

## Architecture

Interface-driven (SOLID + dependency injection). Every pipeline stage is an abstract
interface in `src/sdcc_rag/domain/interfaces.py`, with interchangeable concrete impls:

- `IDocumentLoader` → `TextLoader` (.txt/.md), `JsonLoader` (.json) — routed by extension via `loaders/registry.py:LoaderRegistry`
- `IDocumentSplitter` → `RecursiveCharacterSplitter` (self-contained, no LangChain)
- `IEmbeddingProvider` → `OpenAIEmbeddingProvider` / `AzureOpenAIEmbeddingProvider` / `OllamaEmbeddingProvider` (local, free)
- `IVectorStore` → `ChromaVectorStore`

Data flows as immutable dataclasses (`domain/models.py`): `Document` → `Chunk` → `EmbeddedChunk`.

`ingestion/orchestrator.py:IngestionOrchestrator` is the controller — it receives all
dependencies pre-built (pure DI, no concrete imports inside). `scripts/ingest.py` is the
**composition root**: the only module that knows concrete classes and wires them together.

### Key design rules

- **Embedding provider switch is config-only**: `EMBEDDING_PROVIDER` env var (`openai`|`azure`) drives `embeddings/factory.py:create_embedding_provider`. Never hardcode a provider outside the factory. Both providers use the `openai` SDK (`OpenAI` vs `AzureOpenAI` client); on Azure the API "model" is the deployment name.
- **Embeddings are computed by the provider, not Chroma**: vectors are passed explicitly to `ChromaVectorStore`; do not enable Chroma's internal embedding_function.
- **Idempotent ingestion**: `domain/models.py:make_chunk_id` derives a deterministic SHA-256 id from `(source, index, text)`; the store uses `upsert` keyed on it, so re-ingesting updates instead of duplicating. Preserve this when changing chunking or storage.
- **One file must not stop the pipeline**: loader errors are caught and recorded in `IngestionReport.errors`; unsupported extensions go to `files_skipped`.
- New concrete impls must subclass the matching ABC and be injected at the composition root — keep the orchestrator free of concrete imports.

## Config

All settings live in `src/sdcc_rag/config.py:Settings` (pydantic-settings, reads `.env`,
env vars override). `data/`, `chroma_db/`, and `.env` are gitignored.

## Tests

Unit tests use fakes (`tests/conftest.py`: `FakeEmbeddingProvider`, `FakeVectorStore`) — no
real API calls, no on-disk Chroma. The orchestrator, splitter, loaders, and factory are each
tested in isolation. `pythonpath = ["src"]` is set in `pyproject.toml`.
