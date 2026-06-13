"""Provider di embedding basato su OpenAI API (default ambiente locale)."""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IEmbeddingProvider

# Dimensioni note dei modelli OpenAI, per esporre `dimension` senza una chiamata.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY mancante per il provider 'openai'")
        # import locale: l'SDK è richiesto solo quando il provider è in uso
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._dimension = _MODEL_DIMENSIONS.get(self._model, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
