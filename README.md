# SDCC RAG

Sistema di Retrieval Augmented Generation sul corpus SDCC. Questo repository
contiene la **pipeline di ingestion**: carica documenti TXT / Markdown / JSON,
li suddivide in chunk, li vettorizza e li persiste in ChromaDB.

L'architettura è interface-driven (SOLID + dependency injection): ogni stadio è
un'interfaccia astratta con implementazioni concrete intercambiabili.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # poi compila le chiavi
```

## Esecuzione

Metti i documenti del corpus in `data/`, poi:

```bash
python scripts/ingest.py            # usa DATA_PATH (default ./data)
python scripts/ingest.py path/to    # oppure una cartella esplicita
```

Lo store persistente viene creato in `chroma_db/`.

## Switch OpenAI ↔ Azure OpenAI

Il provider di embedding si seleziona da `.env`, senza toccare il codice:

```dotenv
EMBEDDING_PROVIDER=openai   # default
# EMBEDDING_PROVIDER=azure  # + AZURE_OPENAI_* (endpoint, key, deployment)
```

## Test

```bash
pytest
```

I test di unità usano fake provider/store: nessuna chiamata API né ChromaDB su
disco.

## Architettura

| Stadio        | Interfaccia          | Default               |
|---------------|----------------------|-----------------------|
| Caricamento   | `IDocumentLoader`    | `TextLoader`, `JsonLoader` |
| Splitting     | `IDocumentSplitter`  | `RecursiveCharacterSplitter` |
| Embedding     | `IEmbeddingProvider` | `OpenAIEmbeddingProvider` (o Azure) |
| Persistenza   | `IVectorStore`       | `ChromaVectorStore`   |

`IngestionOrchestrator` riceve le dipendenze già costruite; `scripts/ingest.py`
è la composition root, l'unico punto che conosce le classi concrete.
