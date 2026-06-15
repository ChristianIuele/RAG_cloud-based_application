"""Factory che seleziona il provider LLM in base alla configurazione.

Speculare a `embeddings/factory.py`: unico punto che conosce l'implementazione
concreta; il resto del codice dipende solo da `ILLMProvider`.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import ILLMProvider


def create_llm_provider(settings: Settings) -> ILLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        from sdcc_rag.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(settings)
    raise ValueError(
        f"llm_provider sconosciuto: {settings.llm_provider!r} (valori ammessi: 'ollama')"
    )
