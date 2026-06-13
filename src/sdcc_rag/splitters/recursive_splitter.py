"""Splitter ricorsivo per caratteri (default per l'ambiente locale).

Implementazione autonoma (nessuna dipendenza esterna) ispirata al
RecursiveCharacterTextSplitter: prova a tagliare sui separatori più "grossi"
(paragrafo, riga, frase, parola) e ricorre su quelli più fini finché i pezzi
non rientrano in `chunk_size`, poi li ricompone applicando l'overlap.
"""

from __future__ import annotations

from sdcc_rag.domain.interfaces import IDocumentSplitter
from sdcc_rag.domain.models import Chunk, Document, make_chunk_id

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


class RecursiveCharacterSplitter(IDocumentSplitter):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap deve essere minore di chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        pieces = self._split_text(document.content, self._separators)
        merged = self._merge(pieces)
        chunks: list[Chunk] = []
        for index, text in enumerate(merged):
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=make_chunk_id(document.source, index, text),
                    source=document.source,
                    metadata={**document.metadata, "chunk_index": index},
                )
            )
        return chunks

    # -- internals -----------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Spezza ricorsivamente finché i frammenti stanno in chunk_size."""
        if len(text) <= self._chunk_size:
            return [text] if text else []

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            # ultimo livello: taglio "duro" per caratteri
            return [text[i : i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

        parts = text.split(separator)
        out: list[str] = []
        for part in parts:
            piece = part + separator if separator else part
            if len(piece) <= self._chunk_size:
                if piece:
                    out.append(piece)
            elif rest:
                out.extend(self._split_text(piece, rest))
            else:
                out.append(piece)
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        """Ricompone i frammenti in chunk ~chunk_size con overlap a cavallo."""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self._chunk_size:
                chunks.append(current.strip())
                current = self._tail(current) + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _tail(self, text: str) -> str:
        """Coda di `chunk_overlap` caratteri, da riportare sul chunk successivo."""
        if self._chunk_overlap <= 0:
            return ""
        return text[-self._chunk_overlap :]
