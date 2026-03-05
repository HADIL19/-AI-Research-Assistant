"""
app/services/ai/rag_service.py — Core AI logic for all research features
"""
from __future__ import annotations
from app.services.ai.llm_provider import LLMProvider, get_llm_provider
from app.services.knowledge.vector_store import RetrievedChunk, get_vector_store

# ── Prompt templates ───────────────────────────────────────────────────────────

SUMMARIZE_PROMPT = """You are a scientific research assistant.
Summarize the following research paper content clearly and concisely.
Structure your summary with:
- **Objective**: What problem does the paper address?
- **Methods**: What approach or methodology is used?
- **Results**: What are the key findings?
- **Conclusion**: What is the overall takeaway?

Context:
{context}"""

CONTRIBUTIONS_PROMPT = """You are a critical scientific research assistant.
Based on the paper content below, extract and structure:
- **Key Contributions**: The main novel contributions of this work (bullet points)
- **Limitations**: Weaknesses, constraints, or gaps acknowledged or apparent
- **Future Work**: What directions do the authors suggest or what is missing?

Context:
{context}"""

RELATED_PROMPT = """You are a research assistant helping find related work.
Based on the paper content below, identify and describe:
- **Related Topics**: Key research areas and topics this paper connects to
- **Related Papers Mentioned**: Any papers, authors, or works cited or referenced
- **Research Gap**: What gap in literature does this paper fill?

Context:
{context}"""

QA_PROMPT = """You are a precise scientific research assistant.
Answer the user's question using ONLY the context from the research papers provided.
If the context doesn't contain enough information, say so clearly.
Always cite which paper your answer comes from.

Context:
{context}"""


class RAGService:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_llm_provider()
        self._store = get_vector_store()

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("title", chunk.source)
            parts.append(f"[{i}] From: '{title}' (score: {chunk.score})\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "source": c.source,
                "title": c.metadata.get("title", c.source),
                "score": c.score,
                "snippet": c.content[:200],
            }
            for c in chunks
        ]

    async def summarize(self, source: str) -> dict:
        chunks = self._store.similarity_search("summary abstract introduction conclusion", filter={"source": source})
        context = self._build_context(chunks)
        result = await self._llm.generate(SUMMARIZE_PROMPT.format(context=context), "Summarize this paper.")
        return {"summary": result, "sources": self._sources(chunks)}

    async def extract_contributions(self, source: str) -> dict:
        chunks = self._store.similarity_search("contributions limitations future work novelty", filter={"source": source})
        context = self._build_context(chunks)
        result = await self._llm.generate(CONTRIBUTIONS_PROMPT.format(context=context), "Extract contributions and limitations.")
        return {"analysis": result, "sources": self._sources(chunks)}

    async def find_related(self, source: str) -> dict:
        chunks = self._store.similarity_search("related work references background prior art", filter={"source": source})
        context = self._build_context(chunks)
        result = await self._llm.generate(RELATED_PROMPT.format(context=context), "Find related papers and topics.")
        return {"related": result, "sources": self._sources(chunks)}

    async def ask(self, question: str, source: str | None = None, top_k: int = 5) -> dict:
        filter_dict = {"source": source} if source else None
        chunks = self._store.similarity_search(question, k=top_k, filter=filter_dict)
        context = self._build_context(chunks)
        answer = await self._llm.generate(QA_PROMPT.format(context=context), question)
        return {"answer": answer, "sources": self._sources(chunks)}

    async def stream_ask(self, question: str, source: str | None = None, top_k: int = 5):
        filter_dict = {"source": source} if source else None
        chunks = self._store.similarity_search(question, k=top_k, filter=filter_dict)
        context = self._build_context(chunks)
        async for token in self._llm.stream(QA_PROMPT.format(context=context), question):
            yield token
