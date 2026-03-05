"""
app/services/ingestors/text_ingestor.py — Ingest plain abstract or text
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.knowledge.vector_store import get_vector_store


async def ingest_text(source: str, title: str, text: str, authors: str = "") -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source, "title": title, "authors": authors, "type": "text"}],
    )

    store = get_vector_store()
    ids = store.add_documents(chunks)
    return {"source": source, "title": title, "chunks_stored": len(ids)}
