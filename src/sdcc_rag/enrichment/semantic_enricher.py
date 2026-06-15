"""Enricher semantico: estrae summary e keywords dal testo via LLM.

Dipende dalla porta `ILLMProvider` (iniettata): chiede al modello un JSON
`{"summary": "...", "keywords": [...]}` sul testo del documento.

Robustezza ("un file non ferma la pipeline"): qualsiasi errore di rete, del
modello o di parsing JSON viene assorbito → il documento viene restituito
invariato, senza propagare eccezioni.

Vincolo scalari di Chroma: `keywords` è una lista, quindi viene serializzata in
una stringa (join con virgola) per poter essere persistita nei metadata.
"""

from __future__ import annotations

import dataclasses
import json

from sdcc_rag.domain.interfaces import ILLMProvider, IMetadataEnricher
from sdcc_rag.domain.models import Chunk, Document

_PROMPT_TEMPLATE = (
    "Analizza il seguente testo ed estrai un breve riassunto e le parole chiave.\n"
    'Rispondi SOLO con un JSON nel formato: {{"summary": "<riassunto>", '
    '"keywords": ["parola1", "parola2"]}}.\n\n'
    "TESTO:\n{text}"
)


class SemanticMetadataEnricher(IMetadataEnricher):
    def __init__(self, llm: ILLMProvider, max_chars: int = 4000) -> None:
        self._llm = llm
        self._max_chars = max_chars

    def enrich_document(self, document: Document) -> Document:
        try:
            prompt = _PROMPT_TEMPLATE.format(text=document.content[: self._max_chars])
            raw = self._llm.complete(prompt, json_output=True)
            data = json.loads(raw)

            enriched: dict[str, object] = {}
            summary = data.get("summary")
            if isinstance(summary, str) and summary:
                enriched["summary"] = summary

            keywords = data.get("keywords")
            if isinstance(keywords, list) and keywords:
                # serializzazione a stringa: i valori metadata devono essere scalari
                enriched["keywords"] = ", ".join(str(k) for k in keywords)

            if not enriched:
                return document
            return dataclasses.replace(document, metadata={**document.metadata, **enriched})
        except Exception:
            # degrado controllato: l'arricchimento semantico è best-effort
            return document

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        return chunk  # nessun arricchimento semantico a livello di chunk
