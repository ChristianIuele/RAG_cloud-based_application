"""Provider di embedding basato su Ollama (locale, gratuito).

Adapter dell'architettura esagonale: implementa la porta `IEmbeddingProvider`
usando la libreria ufficiale `ollama`, coerente con gli altri adapter che usano
gli SDK ufficiali direttamente (vedi openai_provider.py / azure_provider.py).

Pensato per testare la pipeline a costo zero contro un server Ollama in esecuzione
in locale: nessuna API key né chiamata cloud. L'import dell'SDK è locale al
costruttore, così il pacchetto `ollama` è richiesto solo quando questo provider
è effettivamente selezionato.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IEmbeddingProvider

# Dimensioni note dei modelli di embedding più comuni serviti da Ollama,
# per esporre `dimension` senza dover interrogare il server.
_OLLAMA_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class OllamaEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        # import locale: la dipendenza `ollama` serve solo se il provider è in uso
        from ollama import Client

        # Nessuna credenziale: ci si connette al daemon Ollama locale (o all'host
        # configurato). Il client non contatta il server finché non si chiama embed().
        self._client = Client(host=settings.ollama_host)
        self._model = settings.ollama_embedding_model
        self._dimension = _OLLAMA_MODEL_DIMENSIONS.get(self._model, 768)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # `embed` accetta nativamente un batch di testi e restituisce un vettore per testo.
        response = self._client.embed(model=self._model, input=texts)
        return response["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
