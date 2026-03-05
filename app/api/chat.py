"""
app/api/chat.py — AI feature endpoints: summarize, contributions, related, Q&A
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.ai.rag_service import RAGService

router = APIRouter(prefix="/api/chat", tags=["chat"])


class SourceRequest(BaseModel):
    source: str = Field(..., description="The paper source ID (filename, arxiv:id, doi:xxx)")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    source: str | None = Field(None, description="Filter to a specific paper, or None for all papers")
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/summarize", summary="Summarize a paper")
async def summarize(body: SourceRequest) -> dict:
    try:
        svc = RAGService()
        return await svc.summarize(body.source)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contributions", summary="Extract key contributions & limitations")
async def contributions(body: SourceRequest) -> dict:
    try:
        svc = RAGService()
        return await svc.extract_contributions(body.source)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/related", summary="Find related papers and topics")
async def related(body: SourceRequest) -> dict:
    try:
        svc = RAGService()
        return await svc.find_related(body.source)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ask", summary="Q&A chat with a paper or entire library")
async def ask(body: AskRequest) -> dict:
    try:
        svc = RAGService()
        return await svc.ask(body.question, source=body.source, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ask/stream", summary="Streaming Q&A via SSE")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    svc = RAGService()

    async def generator():
        try:
            async for token in svc.stream_ask(body.question, source=body.source, top_k=body.top_k):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
