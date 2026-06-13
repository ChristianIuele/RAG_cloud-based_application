"""Test della factory di embedding: selezione provider senza chiamate reali."""

from __future__ import annotations

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.embeddings.factory import create_embedding_provider


def test_factory_seleziona_openai():
    from sdcc_rag.embeddings.openai_provider import OpenAIEmbeddingProvider

    settings = Settings(embedding_provider="openai", openai_api_key="sk-test")
    provider = create_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_factory_seleziona_azure():
    from sdcc_rag.embeddings.azure_provider import AzureOpenAIEmbeddingProvider

    settings = Settings(
        embedding_provider="azure",
        azure_openai_endpoint="https://x.openai.azure.com",
        azure_openai_api_key="key",
        azure_openai_deployment="embed",
    )
    provider = create_embedding_provider(settings)

    assert isinstance(provider, AzureOpenAIEmbeddingProvider)


def test_factory_seleziona_ollama():
    from sdcc_rag.embeddings.ollama_provider import OllamaEmbeddingProvider

    # Nessuna credenziale né server: il client non contatta Ollama in __init__.
    settings = Settings(embedding_provider="ollama")
    provider = create_embedding_provider(settings)

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.dimension == 768  # nomic-embed-text


def test_factory_provider_sconosciuto_solleva():
    settings = Settings(embedding_provider="cohere")
    with pytest.raises(ValueError):
        create_embedding_provider(settings)


def test_openai_senza_chiave_solleva():
    settings = Settings(embedding_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError):
        create_embedding_provider(settings)
