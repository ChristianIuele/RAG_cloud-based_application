"""Test della factory LLM: selezione provider senza contattare il server."""

from __future__ import annotations

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.llm.factory import create_llm_provider


def test_factory_seleziona_ollama():
    from sdcc_rag.llm.ollama_provider import OllamaLLMProvider

    # Il costruttore istanzia solo il Client(host=...): nessuna chiamata di rete.
    provider = create_llm_provider(Settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaLLMProvider)


def test_factory_provider_sconosciuto_solleva():
    with pytest.raises(ValueError):
        create_llm_provider(Settings(llm_provider="openai"))
