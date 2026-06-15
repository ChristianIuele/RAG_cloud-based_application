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

python scripts/ingest.py [PATH]                  # run ingestion (default ./data)
python scripts/query.py [--top-k N] [--min-score F]   # interactive query REPL
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
- `IMetadataEnricher` → `CompositeMetadataEnricher` of `StandardMetadataEnricher` (traceability + counts) / `SemanticMetadataEnricher` (LLM summary+keywords) / `ManualMetadataEnricher` (user title/author)
- `ILLMProvider` → `OllamaLLMProvider` (local, free) — selected via `llm/factory.py:create_llm_provider`

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

### Key design rules

- **Embedding provider switch is config-only**: `EMBEDDING_PROVIDER` env var (`openai`|`azure`) drives `embeddings/factory.py:create_embedding_provider`. Never hardcode a provider outside the factory. Both providers use the `openai` SDK (`OpenAI` vs `AzureOpenAI` client); on Azure the API "model" is the deployment name.
- **Embeddings are computed by the provider, not Chroma**: vectors are passed explicitly to `ChromaVectorStore`; do not enable Chroma's internal embedding_function.
- **Idempotent ingestion**: `domain/models.py:make_chunk_id` derives a deterministic SHA-256 id from `(source, index, text)`; the store uses `upsert` keyed on it, so re-ingesting updates instead of duplicating. Preserve this when changing chunking or storage.
- **One file must not stop the pipeline**: loader errors and per-document errors are caught and recorded in `IngestionReport.errors`; unsupported extensions go to `files_skipped`.
- **Metadata enrichment is metadata-only**: enrichers return new dataclass copies (`dataclasses.replace`) and touch only `metadata`, never `text`/`source`/`chunk_index` — so `chunk_id` and idempotency hold even with volatile/non-deterministic values (timestamp, LLM output). Enriched values must be **scalar** (`str`/`int`/`float`/`bool`) to survive `ChromaVectorStore._metadata`; lists like `keywords` are joined into a string. Doc-level enrichment fans out to chunks via the splitter. `SemanticMetadataEnricher` degrades gracefully (LLM/JSON error → document unchanged).
- New concrete impls must subclass the matching ABC and be injected at the composition root — keep the orchestrator free of concrete imports.
- **Grounded generation (anti-hallucination)**: `RAGService` separates *policy* (a fixed `SYSTEM_PROMPT`) from *data* (the numbered retrieved passages in the user message). The system prompt forces answers to use only the context, mandates an exact abstention sentence (`ABSTENTION_TEXT`) when the answer isn't present, requires `[n]` citations, and treats context as data not instructions (prompt-injection hardening). The **empty-guard** returns the abstention answer *without calling the LLM* when no chunk survives retrieval/threshold. The query embedder must be the same provider used at ingestion (same vector space) — both go through `create_embedding_provider`.

## Config

All settings live in `src/sdcc_rag/config.py:Settings` (pydantic-settings, reads `.env`,
env vars override). `data/`, `chroma_db/`, and `.env` are gitignored.

## Tests

Unit tests use fakes (`tests/conftest.py`: `FakeEmbeddingProvider`, `FakeVectorStore`) — no
real API calls, no on-disk Chroma. The orchestrator, splitter, loaders, and factory are each
tested in isolation. `pythonpath = ["src"]` is set in `pyproject.toml`.
