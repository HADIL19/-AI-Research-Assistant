# 🔬 ResearchAI — AI Research Assistant

> Upload papers, ask questions, get summaries, extract contributions, and find related work — all powered by RAG (Retrieval-Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview
ResearchAI is a Retrieval-Augmented Generation (RAG) system specialized for scientific research. It allows users to build a personal library of academic papers from multiple sources (PDF, ArXiv, DOI, plain text), and then interact with that library using natural language — asking questions, getting summaries, extracting contributions, and finding related work.

🎯 Core Problem Solved
Traditional LLMs hallucinate facts and have outdated knowledge. ResearchAI grounds every answer in your actual uploaded papers, providing citations and relevance scores with every response.


## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Upload** | Drag & drop any research paper |
| 🌐 **ArXiv & DOI** | Fetch papers directly by ID, URL, or DOI |
| 📝 **Plain Text** | Paste any abstract or raw text |
| 📊 **Summarize** | Structured summary: objective / methods / results / conclusion |
| 🔬 **Contributions** | Extract novel contributions, limitations, and future work |
| 🔗 **Related Work** | Find connected topics, cited papers, and research gaps |
| 💬 **Q&A Chat** | Ask anything across your entire library or a single paper |
| ⚡ **Streaming** | Real-time token-by-token responses via SSE |

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Backend | FastAPI | HTTP framework, async, auto-docs |
| AI / LLM | Gemini 1.5 Flash via LangChain | Answer generation |
| Embeddings | Google `embedding-001` | Text → vector conversion |
| Vector DB | ChromaDB | Local persistent similarity search |
| PDF Parsing | PyMuPDF (`fitz`) | Fast PDF text extraction |
| ArXiv | `arxiv` Python library | Fetch papers by ID or URL |
| DOI | CrossRef REST API (free) | Fetch metadata by DOI |
| Frontend | Vanilla HTML / CSS / JS | Single-page UI served by FastAPI |

---

## 📁 Project Structure

```
research-ai/
├── app/
│   ├── api/
│   │   ├── papers.py             # Ingest endpoints (PDF, ArXiv, DOI, text)
│   │   └── chat.py               # AI endpoints (summarize, Q&A, contributions)
│   ├── core/
│   │   └── config.py             # Settings loaded from .env
│   ├── services/
│   │   ├── ai/
│   │   │   ├── llm_provider.py   # LLM interface + Gemini implementation
│   │   │   └── rag_service.py    # RAG orchestration (retrieve → generate)
│   │   ├── ingestors/
│   │   │   ├── pdf_ingestor.py   # PyMuPDF parser
│   │   │   ├── arxiv_ingestor.py # ArXiv API client
│   │   │   ├── doi_ingestor.py   # CrossRef API client
│   │   │   └── text_ingestor.py  # Plain text handler
│   │   └── knowledge/
│   │       └── vector_store.py   # ChromaDB wrapper
│   └── main.py                   # FastAPI app + static file serving
├── static/
│   └── index.html                # Web UI (no separate server needed)
├── tests/
│   └── test_research_ai.py       # Unit tests (fully mocked)
├── .env.example                  # Environment variable template
├── .gitignore
└── pyproject.toml
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YounesBensafia/research-ai.git
cd research-ai
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn python-multipart langchain langchain-google-genai \
    langchain-chroma langchain-text-splitters chromadb google-generativeai \
    pydantic-settings python-dotenv pymupdf arxiv httpx pytest pytest-asyncio
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
```

> 🔑 Get a free key at → https://aistudio.google.com/app/apikey

### 4. Run the server

```bash
python -m uvicorn app.main:app --reload
```

Open → **http://localhost:8000**  
Swagger docs → **http://localhost:8000/docs**

---

## 🧠 How RAG Works

```
User Question
     │
     ▼
Embed question → vector
     │
     ▼
Similarity search in ChromaDB (Top-K chunks)
     │
     ▼
Build context from retrieved chunks
     │
     ▼
Inject context into prompt template
     │
     ▼
Gemini generates grounded answer
     │
     ▼
Response + source citations returned
```

Every answer is grounded in your actual uploaded papers — no hallucinations, full citations.

---

## 📡 API Reference

### Papers — Ingest

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/papers/upload` | Upload a PDF file |
| `POST` | `/api/papers/arxiv` | Ingest from ArXiv URL or ID |
| `POST` | `/api/papers/doi` | Ingest from DOI |
| `POST` | `/api/papers/text` | Ingest plain text or abstract |
| `GET` | `/api/papers/` | List all papers + stats |
| `DELETE` | `/api/papers/` | Remove a paper by source |

### Chat — AI Features

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/summarize` | Structured paper summary |
| `POST` | `/api/chat/contributions` | Contributions & limitations |
| `POST` | `/api/chat/related` | Related work & topics |
| `POST` | `/api/chat/ask` | Free Q&A (full response) |
| `POST` | `/api/chat/ask/stream` | Streaming Q&A via SSE |

### Example

```bash
# Ingest the Attention paper from ArXiv
curl -X POST http://localhost:8000/api/papers/arxiv \
  -H "Content-Type: application/json" \
  -d '{ "url_or_id": "1706.03762" }'

# Ask a question
curl -X POST http://localhost:8000/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{ "question": "What is the main contribution of this paper?", "top_k": 5 }'
```

---


## 📄 License

MIT — free to use and adapt.

---

<p align="center">Built with ❤️ by <a href="https://github.com/YounesBensafia">YounesBensafia</a></p>
