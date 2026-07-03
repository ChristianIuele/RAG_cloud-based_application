"""Test di AzureOpenAIMetadataExtractor senza chiamate cloud.

Il client `AzureOpenAI` reale viene sostituito da un fake: si verifica la logica
pura dell'extractor (prompt/troncamento, parsing JSON → AutomaticMetadata, degrado
best-effort su errore) e la guardia sulle credenziali.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sdcc_rag.config import Settings
from sdcc_rag.domain.models import AutomaticMetadata
from sdcc_rag.enrichment.azure_metadata_extractor import AzureOpenAIMetadataExtractor


class FakeChatCompletions:
    """Sostituto di client.chat.completions: registra la chiamata e risponde."""

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _extractor(content=None, error=None, max_chars=4000):
    """Costruisce l'extractor bypassando l'__init__ reale (che istanzia l'SDK)."""
    ext = AzureOpenAIMetadataExtractor.__new__(AzureOpenAIMetadataExtractor)
    completions = FakeChatCompletions(content=content, error=error)
    ext._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    ext._deployment = "gpt-5.4-nano"
    ext._max_chars = max_chars
    return ext, completions


def test_costruttore_richiede_credenziali():
    settings = Settings(azure_openai_endpoint=None, azure_openai_api_key=None)
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAIMetadataExtractor(settings)


def test_parsing_completo():
    payload = {
        "summary": "un riassunto",
        "keywords": ["a", "b"],
        "suggested_categories": ["tecnico"],
        "language": "it",
        "entities": ["SDCC", "ACME"],
    }
    ext, _ = _extractor(content=json.dumps(payload))

    result = ext.extract("qualche testo")

    assert result == AutomaticMetadata(
        summary="un riassunto",
        keywords=["a", "b"],
        suggested_categories=["tecnico"],
        language="it",
        entities=["SDCC", "ACME"],
    )


def test_tollera_lista_come_stringa_csv():
    # il modello può restituire una stringa CSV invece di un array
    payload = {"summary": "s", "keywords": "a, b, c", "language": "en"}
    ext, _ = _extractor(content=json.dumps(payload))

    result = ext.extract("t")

    assert result.keywords == ["a", "b", "c"]
    assert result.suggested_categories == []  # campo assente → lista vuota


def test_json_malformato_ritorna_vuoto():
    ext, _ = _extractor(content="non un json")
    assert ext.extract("t") == AutomaticMetadata()


def test_eccezione_client_ritorna_vuoto():
    ext, _ = _extractor(error=RuntimeError("Azure non raggiungibile"))
    assert ext.extract("t") == AutomaticMetadata()


def test_troncamento_a_max_chars():
    ext, completions = _extractor(content=json.dumps({"summary": "s"}), max_chars=10)
    ext.extract("x" * 100)

    user_message = completions.last_kwargs["messages"][1]["content"]
    assert user_message.count("x") == 10  # testo troncato a max_chars
    assert completions.last_kwargs["response_format"] == {"type": "json_object"}
    assert completions.last_kwargs["model"] == "gpt-5.4-nano"
