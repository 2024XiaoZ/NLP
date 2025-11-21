"""LangChain ChatOpenAI 客户端，复用 demo 中的配置方式。"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from backend.core.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """返回全局 ChatOpenAI 实例，沿用 teammate demo 的参数。"""

    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，无法调用对话模型。")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        openai_api_key=settings.llm_api_key,
        base_url=str(settings.llm_base_url),
    )



