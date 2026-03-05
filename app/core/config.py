"""
app/core/config.py — Central settings loaded from .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini
    gemini_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "research_papers"

    # RAG
    top_k_results: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200


settings = Settings()
