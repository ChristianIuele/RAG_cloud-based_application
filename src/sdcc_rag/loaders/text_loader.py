"""Loader per file di testo semplice e Markdown."""

from __future__ import annotations

from pathlib import Path

from sdcc_rag.domain.interfaces import IDocumentLoader
from sdcc_rag.domain.models import Document


class TextLoader(IDocumentLoader):
    """Carica .txt e .md come un singolo Document col contenuto integrale."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".md", ".markdown"}

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8")
        metadata = {"filename": path.name, "extension": path.suffix.lower()}
        return [Document(content=content, source=str(path), metadata=metadata)]
