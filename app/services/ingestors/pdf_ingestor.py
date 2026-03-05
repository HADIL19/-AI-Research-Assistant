"""
app/services/ingestors/pdf_ingestor.py — Parse and ingest PDF papers
"""
from __future__ import annotations
import io
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.knowledge.vector_store import get_vector_store


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Extract rough title from first 300 chars
    title = filename.replace(".pdf", "").replace("_", " ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.create_documents(
        texts=[full_text],
        metadatas=[{"source": filename, "title": title, "type": "pdf"}],
    )

    store = get_vector_store()
    ids = store.add_documents(chunks)
    return {"source": filename, "title": title, "chunks_stored": len(ids)}
