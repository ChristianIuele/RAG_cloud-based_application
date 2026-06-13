"""Provider di embedding basato su Azure OpenAI Service.

Stessa interfaccia di OpenAIEmbeddingProvider: cambia solo la costruzione del
client (AzureOpenAI con endpoint/deployment/api-version). Lo switch è guidato
da `Settings.embedding_provider` tramite la factory, quindi è zero-codice lato
chiamante.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IEmbeddingProvider
from sdcc_rag.embeddings.openai_provider import _MODEL_DIMENSIONS


class AzureOpenAIEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                "AZURE_OPENAI_DEPLOYMENT": settings.azure_openai_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Variabili Azure mancanti per il provider 'azure': " + ", ".join(missing)
            )

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        # su Azure il "model" della chiamata è il nome del deployment
        self._deployment = settings.azure_openai_deployment
        self._dimension = _MODEL_DIMENSIONS.get(settings.openai_embedding_model, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._deployment, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
