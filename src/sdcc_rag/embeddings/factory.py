"""Factory che seleziona il provider di embedding in base alla configurazione.

È l'unico punto che conosce entrambe le implementazioni concrete: il resto del
codice dipende solo da IEmbeddingProvider.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IEmbeddingProvider


def create_embedding_provider(settings: Settings) -> IEmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "azure":
        from sdcc_rag.embeddings.azure_provider import AzureOpenAIEmbeddingProvider

        return AzureOpenAIEmbeddingProvider(settings)
    if provider == "openai":
        from sdcc_rag.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(settings)
    if provider == "ollama":
        from sdcc_rag.embeddings.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(settings)
    raise ValueError(
        f"embedding_provider sconosciuto: {settings.embedding_provider!r} "
        "(valori ammessi: 'openai', 'azure', 'ollama')"
    )
