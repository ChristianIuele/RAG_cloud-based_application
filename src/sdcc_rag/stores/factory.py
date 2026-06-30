"""Factory che seleziona il vector store in base alla configurazione.

È l'unico punto che conosce entrambe le implementazioni concrete: il resto del
codice dipende solo da IVectorStore. Lo switch è guidato da `Settings.vector_store`
(env `VECTOR_STORE`), specularmente a `embeddings/factory.py`.
"""

from __future__ import annotations

from sdcc_rag.config import Settings
from sdcc_rag.domain.interfaces import IVectorStore


def create_vector_store(settings: Settings) -> IVectorStore:
    store = settings.vector_store.lower()
    if store in {"azure_search", "azure"}:
        from sdcc_rag.stores.azure_search_store import AzureSearchVectorStore

        return AzureSearchVectorStore(settings)
    if store == "chroma":
        from sdcc_rag.stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(settings)
    raise ValueError(
        f"vector_store sconosciuto: {settings.vector_store!r} "
        "(valori ammessi: 'chroma', 'azure_search')"
    )
