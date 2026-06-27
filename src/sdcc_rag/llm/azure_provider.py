"""Provider LLM basato su Azure OpenAI Service.

Adapter della porta `ILLMProvider`: speculare a `OllamaLLMProvider` lato contratto,
ma costruisce un client `AzureOpenAI` (endpoint/deployment/api-version) come fa
`AzureOpenAIEmbeddingProvider`. Lo switch è guidato da `Settings.llm_provider`
tramite la factory, quindi è zero-codice lato chiamante. Import lazy di
`AzureOpenAI`: la dipendenza serve solo se il provider è effettivamente in uso.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import ILLMProvider


class AzureOpenAILLMProvider(ILLMProvider):
    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
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
        self._deployment = settings.azure_llm_deployment

    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        # Il system message (policy/ruolo) precede il messaggio utente (dati):
        # separarli rende il modello più aderente alle regole e resistente al
        # prompt-injection contenuto nei dati.
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        # `response_format` json_object vincola il modello a produrre JSON valido;
        # lo si imposta solo quando richiesto per non alterare gli altri usi.
        kwargs: dict = {}
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            # temperatura 0 → output il più stabile/riproducibile possibile.
            temperature=0,
            **kwargs,
        )
        return response.choices[0].message.content
