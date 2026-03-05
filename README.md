# 🔬 ResearchAI — AI Research Assistant

> Upload papers, ask questions, get summaries, extract contributions, and find related work — all powered by RAG.

---

## ✨ Features

- 📄 **PDF Upload** — drag & drop any research paper
- 🌐 **ArXiv & DOI** — fetch papers directly by ID or DOI
- 📝 **Plain Text** — paste any abstract or text
- 📊 **Summarize** — structured objective / methods / results / conclusion
- 🔬 **Contributions & Limitations** — extract what's novel and what's missing
- 🔗 **Related Work** — find connected topics and cited works
- 💬 **Q&A Chat** — ask anything across your entire library or a single paper

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI |
| AI / LLM | Gemini 1.5 Flash via LangChain |
| Embeddings | Google `embedding-001` |
| Vector DB | ChromaDB |
| PDF Parsing | PyMuPDF |
| ArXiv | `arxiv` Python library |
| DOI | CrossRef API (free) |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📁 Project Structure

```
research-ai/
├── app/
│   ├── api/
│   │   ├── papers.py        # Ingest endpoints
│   │   └── chat.py          # AI feature endpoints
│   ├── core/
│   │   └── config.py
│   ├── services/
│   │   ├── ai/
│   │   │   ├── llm_provider.py
│   │   │   └── rag_service.py
│   │   ├── ingestors/
│   │   │   ├── pdf_ingestor.py
│   │   │   ├── arxiv_ingestor.py
│   │   │   ├── doi_ingestor.py
│   │   │   └── text_ingestor.py
│   │   └── knowledge/
│   │       └── vector_store.py
│   └── main.py
├── static/
│   └── index.html           # Web UI
├── tests/
│   └── test_research_ai.py
├── .env.example
└── pyproject.toml
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/YounesBensafia/research-ai.git
cd research-ai

pip install fastapi uvicorn python-multipart langchain langchain-google-genai \
    langchain-chroma langchain-text-splitters chromadb google-generativeai \
    pydantic-settings python-dotenv pymupdf arxiv httpx pytest pytest-asyncio
```

### 2. Configure

```bash
cp .env.example .env
# Add your Gemini API key inside .env
```

Get a free key at → https://aistudio.google.com/app/apikey

### 3. Run

```bash
python -m uvicorn app.main:app --reload
```

Open → **http://localhost:8000**

---

## 📡 API Reference

### Papers (Ingest)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/papers/upload` | Upload PDF |
| `POST` | `/api/papers/arxiv` | Ingest from ArXiv |
| `POST` | `/api/papers/doi` | Ingest from DOI |
| `POST` | `/api/papers/text` | Ingest plain text |
| `GET` | `/api/papers/` | List all papers |
| `DELETE` | `/api/papers/` | Remove a paper |

### Chat (AI Features)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/summarize` | Summarize a paper |
| `POST` | `/api/chat/contributions` | Contributions & limitations |
| `POST` | `/api/chat/related` | Related work |
| `POST` | `/api/chat/ask` | Free Q&A |
| `POST` | `/api/chat/ask/stream` | Streaming Q&A (SSE) |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

No API key needed — all tests are fully mocked.

---

## 📄 License

MIT — free to use and adapt.
