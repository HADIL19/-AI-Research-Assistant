"""
app/main.py — ResearchAI FastAPI entrypoint
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.api.papers import router as papers_router

app = FastAPI(
    title="ResearchAI",
    description="AI Research Assistant — summarize papers, Q&A, find sources",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(papers_router)
app.include_router(chat_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "ResearchAI"}
