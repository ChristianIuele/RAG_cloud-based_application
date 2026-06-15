"""Test delle primitive pure di prompting (nessun LLM coinvolto)."""

from __future__ import annotations

from sdcc_rag.domain.models import Chunk
from sdcc_rag.retrieval.prompt import (
    ABSTENTION_TEXT,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def _chunk(text: str, source: str) -> Chunk:
    return Chunk(text=text, chunk_id=f"id-{source}", source=source)


def test_system_prompt_contiene_regole_anti_allucinazione():
    # grounding: solo il contesto
    assert "ESCLUSIVAMENTE" in SYSTEM_PROMPT
    # frase di astensione esatta presente nel system
    assert ABSTENTION_TEXT in SYSTEM_PROMPT
    # regola di citazione numerica
    assert "[1]" in SYSTEM_PROMPT
    # hardening contro prompt-injection nei documenti
    assert "non istruzioni" in SYSTEM_PROMPT


def test_build_user_prompt_numera_passaggi_e_include_testi_e_fonti():
    chunks = [
        _chunk("Il cielo è blu.", "doc1.txt"),
        _chunk("L'erba è verde.", "doc2.md"),
    ]
    prompt = build_user_prompt("Di che colore è il cielo?", chunks)

    # passaggi numerati con la fonte
    assert "[1] (fonte: doc1.txt)" in prompt
    assert "[2] (fonte: doc2.md)" in prompt
    # i testi dei chunk sono presenti
    assert "Il cielo è blu." in prompt
    assert "L'erba è verde." in prompt
    # la domanda è inclusa
    assert "Di che colore è il cielo?" in prompt
    # istruzione finale di grounding
    assert "solo i passaggi del CONTESTO" in prompt
