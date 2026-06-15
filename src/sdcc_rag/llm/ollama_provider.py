"""Provider LLM basato su Ollama (locale, gratuito).

Adapter della porta `ILLMProvider`: usa la libreria ufficiale `ollama` per
invocare un modello di generazione locale (default `llama3`), coerente con gli
adapter di embedding (import lazy nel costruttore, riuso di `ollama_host`).

Serve all'arricchimento semantico (summary, keywords) a costo zero, senza
dipendere da server cloud.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import ILLMProvider


class OllamaLLMProvider(ILLMProvider):
    def __init__(self, settings: Settings) -> None:
        # import locale: la dipendenza `ollama` serve solo se il provider è in uso
        from ollama import Client

        # Nessuna credenziale: server locale. Il client non contatta Ollama
        # finché non si chiama complete().
        self._client = Client(host=settings.ollama_host)
        self._model = settings.ollama_llm_model

    def complete(self, prompt: str, *, json_output: bool = False) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            # `format="json"` vincola il modello a produrre JSON valido.
            format="json" if json_output else "",
            # temperatura 0 → output il più stabile/riproducibile possibile.
            options={"temperature": 0},
        )
        return response["message"]["content"]
