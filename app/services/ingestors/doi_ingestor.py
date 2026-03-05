"""
app/services/ingestors/doi_ingestor.py — Fetch metadata via CrossRef API and ingest
"""
from __future__ import annotations
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.knowledge.vector_store import get_vector_store


async def ingest_doi(doi: str) -> dict:
    doi = doi.strip().lstrip("https://doi.org/").lstrip("doi:")
    url = f"https://api.crossref.org/works/{doi}"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "ResearchAI/0.1"})
        resp.raise_for_status()
        data = resp.json()["message"]

    title = " ".join(data.get("title", ["Unknown Title"]))
    authors = ", ".join(
        f"{a.get('given','')} {a.get('family','')}".strip()
        for a in data.get("author", [])
    )
    abstract = data.get("abstract", "No abstract available.")
    journal = data.get("container-title", [""])[0]
    year = data.get("published", {}).get("date-parts", [[""]])[0][0]

    text = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Journal: {journal} ({year})\n"
        f"DOI: {doi}\n\n"
        f"Abstract: {abstract}"
    )
    source = f"doi:{doi}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source": source,
            "title": title,
            "authors": authors,
            "type": "doi",
            "doi": doi,
            "journal": journal,
        }],
    )

    store = get_vector_store()
    ids = store.add_documents(chunks)
    return {
        "source": source,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": str(year),
        "chunks_stored": len(ids),
    }
