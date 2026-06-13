"""Registro che instrada un file al loader giusto in base all'estensione."""

from __future__ import annotations

from pathlib import Path

from sdcc_rag.domain.interfaces import IDocumentLoader


class LoaderRegistry:
    def __init__(self, loaders: list[IDocumentLoader]) -> None:
        self._by_extension: dict[str, IDocumentLoader] = {}
        for loader in loaders:
            for ext in loader.supported_extensions:
                self._by_extension[ext.lower()] = loader

    def resolve(self, path: Path) -> IDocumentLoader | None:
        """Loader per il file, o None se l'estensione non è supportata."""
        return self._by_extension.get(path.suffix.lower())

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._by_extension)
