"""
app/api/papers.py — Ingest endpoints for all paper sources
"""
from __future__ import annotations
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.services.ingestors.pdf_ingestor import ingest_pdf
from app.services.ingestors.arxiv_ingestor import ingest_arxiv
from app.services.ingestors.doi_ingestor import ingest_doi
from app.services.ingestors.text_ingestor import ingest_text
from app.services.knowledge.vector_store import get_vector_store

router = APIRouter(prefix="/api/papers", tags=["papers"])


class ArxivRequest(BaseModel):
    url_or_id: str


class DoiRequest(BaseModel):
    doi: str


class TextRequest(BaseModel):
    source: str
    title: str
    text: str
    authors: str = ""


@router.post("/upload", summary="Upload a PDF paper")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    try:
        contents = await file.read()
        return ingest_pdf(contents, file.filename)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/arxiv", summary="Ingest from ArXiv URL or ID")
async def from_arxiv(body: ArxivRequest) -> dict:
    try:
        return await ingest_arxiv(body.url_or_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/doi", summary="Ingest from DOI")
async def from_doi(body: DoiRequest) -> dict:
    try:
        return await ingest_doi(body.doi)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/text", summary="Ingest plain text or abstract")
async def from_text(body: TextRequest) -> dict:
    try:
        return await ingest_text(body.source, body.title, body.text, body.authors)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/", summary="List all ingested papers")
async def list_papers() -> dict:
    store = get_vector_store()
    return {"papers": store.list_papers(), "total_chunks": store.count()}


@router.delete("/", summary="Remove a paper by source")
async def delete_paper(source: str) -> dict:
    store = get_vector_store()
    store.delete_by_source(source)
    return {"deleted": True, "source": source}
