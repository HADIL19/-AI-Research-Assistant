"""
app/services/knowledge/vector_store.py — ChromaDB + Gemini embeddings
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self._db = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._db.add_documents(documents)

    def similarity_search(self, query: str, k: Optional[int] = None, filter: Optional[dict] = None) -> list[RetrievedChunk]:
        k = k or settings.top_k_results
        results = self._db.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        return [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=round(score, 4),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]

    def delete_by_source(self, source: str) -> None:
        self._db.delete(where={"source": source})

    def list_papers(self) -> list[dict]:
        """Return unique papers stored in the DB."""
        results = self._db.get(include=["metadatas"])
        seen, papers = set(), []
        for meta in results["metadatas"]:
            src = meta.get("source", "")
            if src not in seen:
                seen.add(src)
                papers.append({
                    "source": src,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", ""),
                    "type": meta.get("type", ""),
                })
        return papers

    def count(self) -> int:
        return self._db._collection.count()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
