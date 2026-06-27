"""Test del provider LLM Azure: solo unit test, nessuna chiamata reale ad Azure.

La validazione delle env var nel costruttore è verificabile offline; il contratto
`complete()` è documentato tramite un Fake in-memory che implementa `ILLMProvider`
(NON unittest.mock), così non serve mai contattare il servizio.
"""

from __future__ import annotations

import json

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import ILLMProvider
from sdcc_rag.llm.azure_provider import AzureOpenAILLMProvider
from sdcc_rag.llm.factory import create_llm_provider


class FakeAzureLLM(ILLMProvider):
    """Fake concreto in-memory: cattura ciò che riceve e ritorna un JSON canned.

    Permette di asserire che `system` e `json_output` siano accettati e propagati,
    senza alcuna dipendenza di rete o da `openai`.
    """

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {"summary": "ok", "keywords": ["a", "b"]}
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_json_output: bool | None = None

    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system
        self.last_json_output = json_output
        return json.dumps(self._payload)


def _settings_azure(**overrides) -> Settings:
    base = dict(
        llm_provider="azure",
        azure_openai_endpoint="https://x.openai.azure.com",
        azure_openai_api_key="key",
    )
    base.update(overrides)
    return Settings(**base)


def test_fake_rispetta_il_contratto_complete():
    """Il Fake documenta il contratto ILLMProvider.complete senza toccare la rete."""
    fake = FakeAzureLLM()
    out = fake.complete("domanda", system="sei un assistente", json_output=True)

    assert isinstance(out, str)
    assert json.loads(out) == {"summary": "ok", "keywords": ["a", "b"]}
    assert fake.last_system == "sei un assistente"
    assert fake.last_json_output is True


def test_init_solleva_se_manca_endpoint(monkeypatch):
    settings = _settings_azure()
    monkeypatch.setattr(settings, "azure_openai_endpoint", None)
    with pytest.raises(ValueError):
        AzureOpenAILLMProvider(settings)


def test_init_solleva_se_manca_api_key(monkeypatch):
    settings = _settings_azure()
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    with pytest.raises(ValueError):
        AzureOpenAILLMProvider(settings)


def test_init_solleva_se_mancano_tutte_le_var(monkeypatch):
    settings = _settings_azure()
    monkeypatch.setattr(settings, "azure_openai_endpoint", None)
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    with pytest.raises(ValueError):
        AzureOpenAILLMProvider(settings)


def test_factory_seleziona_azure():
    # Con env var valide il costruttore crea solo il client AzureOpenAI:
    # nessuna chiamata di rete finché non si invoca complete().
    provider = create_llm_provider(_settings_azure())
    assert isinstance(provider, AzureOpenAILLMProvider)
