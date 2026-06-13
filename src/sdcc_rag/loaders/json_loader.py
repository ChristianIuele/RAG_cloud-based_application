"""Loader per file JSON.

Supporta due forme comuni del corpus:
- un oggetto singolo  -> un Document
- una lista di oggetti -> un Document per elemento

Il testo viene estratto dal campo configurabile `json_text_field`; gli altri
campi scalari del record diventano metadata. Se il campo manca (o il JSON non è
fatto di oggetti) si serializza l'intero valore come fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sdcc_rag.domain.interfaces import IDocumentLoader
from sdcc_rag.domain.models import Document


class JsonLoader(IDocumentLoader):
    def __init__(self, text_field: str = "text") -> None:
        self._text_field = text_field

    @property
    def supported_extensions(self) -> set[str]:
        return {".json"}

    def load(self, path: Path) -> list[Document]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else [raw]

        documents: list[Document] = []
        for index, record in enumerate(records):
            content = self._extract_text(record)
            metadata: dict[str, Any] = {
                "filename": path.name,
                "extension": ".json",
                "record_index": index,
            }
            if isinstance(record, dict):
                metadata.update(
                    {
                        k: v
                        for k, v in record.items()
                        if k != self._text_field and isinstance(v, (str, int, float, bool))
                    }
                )
            documents.append(
                Document(content=content, source=f"{path}#{index}", metadata=metadata)
            )
        return documents

    def _extract_text(self, record: Any) -> str:
        if isinstance(record, dict) and self._text_field in record:
            return str(record[self._text_field])
        if isinstance(record, str):
            return record
        # fallback: serializza l'intero record
        return json.dumps(record, ensure_ascii=False)
