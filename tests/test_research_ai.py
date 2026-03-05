"""
tests/test_research_ai.py — Unit tests (fully mocked, no API key needed)
Run: python -m pytest tests/ -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestLLMProvider:
    def test_factory_returns_gemini(self):
        with patch("app.services.ai.llm_provider.ChatGoogleGenerativeAI"):
            from app.services.ai.llm_provider import get_llm_provider, GeminiProvider
            assert isinstance(get_llm_provider(), GeminiProvider)

    @pytest.mark.asyncio
    async def test_generate(self):
        with patch("app.services.ai.llm_provider.ChatGoogleGenerativeAI") as m:
            m.return_value.ainvoke = AsyncMock(return_value=MagicMock(content="Answer"))
            from app.services.ai.llm_provider import GeminiProvider
            p = GeminiProvider()
            assert await p.generate("sys", "user") == "Answer"


class TestVectorStore:
    def test_add_and_search(self):
        with patch("app.services.knowledge.vector_store.Chroma") as mc, \
             patch("app.services.knowledge.vector_store.GoogleGenerativeAIEmbeddings"):
            from langchain_core.documents import Document
            mc.return_value.add_documents.return_value = ["id1"]
            mc.return_value.similarity_search_with_relevance_scores.return_value = [
                (Document(page_content="test", metadata={"source": "arxiv:1234"}), 0.91)
            ]
            from app.services.knowledge.vector_store import VectorStore
            store = VectorStore()
            ids = store.add_documents([Document(page_content="test", metadata={"source": "arxiv:1234"})])
            assert ids == ["id1"]
            results = store.similarity_search("test query")
            assert results[0].score == 0.91


class TestIngestors:
    def test_pdf_ingestor(self):
        with patch("app.services.ingestors.pdf_ingestor.fitz") as mf, \
             patch("app.services.ingestors.pdf_ingestor.get_vector_store") as mv:
            mock_doc = MagicMock()
            mock_doc.__iter__ = MagicMock(return_value=iter([MagicMock(get_text=lambda: "text")]))
            mf.open.return_value = mock_doc
            mv.return_value.add_documents.return_value = ["id1", "id2"]
            from app.services.ingestors.pdf_ingestor import ingest_pdf
            result = ingest_pdf(b"fake", "paper.pdf")
            assert result["chunks_stored"] == 2

    @pytest.mark.asyncio
    async def test_text_ingestor(self):
        with patch("app.services.ingestors.text_ingestor.get_vector_store") as mv:
            mv.return_value.add_documents.return_value = ["id1"]
            from app.services.ingestors.text_ingestor import ingest_text
            result = await ingest_text("src/test", "My Paper", "This is a test abstract.", "Author A")
            assert result["chunks_stored"] == 1
            assert result["title"] == "My Paper"


class TestChatAPI:
    @pytest.mark.asyncio
    async def test_ask_endpoint(self):
        with patch("app.api.chat.RAGService") as ms:
            ms.return_value.ask = AsyncMock(return_value={
                "answer": "The model uses attention.",
                "sources": [{"source": "arxiv:1706.03762", "title": "Attention is All You Need", "score": 0.97, "snippet": "..."}]
            })
            from fastapi.testclient import TestClient
            from app.main import app
            client = TestClient(app)
            resp = client.post("/api/chat/ask", json={"question": "How does attention work?"})
            assert resp.status_code == 200
            assert "answer" in resp.json()

    @pytest.mark.asyncio
    async def test_summarize_endpoint(self):
        with patch("app.api.chat.RAGService") as ms:
            ms.return_value.summarize = AsyncMock(return_value={
                "summary": "This paper proposes...",
                "sources": []
            })
            from fastapi.testclient import TestClient
            from app.main import app
            client = TestClient(app)
            resp = client.post("/api/chat/summarize", json={"source": "arxiv:1706.03762"})
            assert resp.status_code == 200
            assert "summary" in resp.json()
