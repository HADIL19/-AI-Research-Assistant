"""
app/services/ai/llm_provider.py — LLM interface + Gemini implementation
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str) -> str: ...

    @abstractmethod
    async def stream(self, system_prompt: str, user_message: str) -> AsyncIterator[str]: ...


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )

    async def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = await self._llm.ainvoke(messages)
        return response.content

    async def stream(self, system_prompt: str, user_message: str) -> AsyncIterator[str]:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        async for chunk in self._llm.astream(messages):
            if chunk.content:
                yield chunk.content


def get_llm_provider() -> LLMProvider:
    return GeminiProvider()
