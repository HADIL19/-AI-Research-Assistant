"""
app/services/ingestors/arxiv_ingestor.py — Fetch and ingest papers from ArXiv
"""
from __future__ import annotations
import re
import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.knowledge.vector_store import get_vector_store


def _extract_arxiv_id(url_or_id: str) -> str:
    """Accept full URL or bare ID like 2401.12345."""
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract ArXiv ID from: {url_or_id}")


async def ingest_arxiv(url_or_id: str) -> dict:
    arxiv_id = _extract_arxiv_id(url_or_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        raise ValueError(f"No paper found for ArXiv ID: {arxiv_id}")

    paper = results[0]
    text = f"Title: {paper.title}\n\nAuthors: {', '.join(a.name for a in paper.authors)}\n\nAbstract: {paper.summary}"
    source = f"arxiv:{arxiv_id}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source": source,
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors),
            "type": "arxiv",
            "arxiv_id": arxiv_id,
            "url": paper.entry_id,
        }],
    )

    store = get_vector_store()
    ids = store.add_documents(chunks)
    return {
        "source": source,
        "title": paper.title,
        "authors": ", ".join(a.name for a in paper.authors),
        "chunks_stored": len(ids),
        "url": paper.entry_id,
    }
