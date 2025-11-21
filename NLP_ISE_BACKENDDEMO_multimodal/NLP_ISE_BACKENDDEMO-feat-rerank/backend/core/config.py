"""全局配置模块。

通过 `pydantic_settings.BaseSettings` 读取 `.env` 或系统环境变量，
避免在代码中硬编码任何密钥。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一的服务端配置对象。"""

    model_config = SettingsConfigDict(
        env_file=[".env", "backend/.env"],  # 支持项目根目录和 backend 目录的 .env
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: HttpUrl = Field(
        "https://api.zhizengzeng.com/v1",
        description="OpenAI 兼容推理服务地址，对齐 demo 配置。",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="OpenAI 兼容 API Key，必须提供才能运行。",
        repr=False,
    )
    llm_model: str = Field(
        "gpt-4o-mini",
        description="对话模型 ID，默认与 demo 一致。",
    )
    llm_temperature: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="生成温度，控制回答确定性。",
    )
    llm_max_tokens: int = Field(
        800,
        ge=64,
        description="回答阶段的最大片段数。",
    )

    llm_vision_model: str = Field(
        "gpt-4o-mini",
        description="支持图像输入的模型 ID。",
    )

    embedding_model: str = Field(
        "text-embedding-3-small",
        description="LangChain OpenAI Embeddings 模型。",
    )

    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily 搜索密钥，为空时禁用实时检索。",
        repr=False,
    )

    local_top_k: int = Field(
        6,
        ge=1,
        description="本地向量库召回数量（等同 demo 的 similarity_search k）。",
    )

    cache_ttl_seconds: int = Field(
        900,
        ge=60,
        description="内存级缓存的默认 TTL（秒）。",
    )

    enable_rerank: bool = Field(
        True,
        description="是否启用重排序功能。",
    )

    rerank_vector_weight: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="本地重排序中向量相似度权重。",
    )

    rerank_bm25_weight: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="本地重排序中 BM25 权重。",
    )

    rerank_recency_weight: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="网络重排序中时效性权重。",
    )

    rerank_authority_weight: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="网络重排序中权威性权重。",
    )

    rerank_relevance_weight: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="网络重排序中相关性权重。",
    )

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="代码根目录，自动推导。",
    )
    storage_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "storage",
        description="存储目录，承载索引与原始数据。",
    )

    def path_for(self, *relative_parts: str) -> Path:
        """根据相对路径拼接出绝对路径，便于统一管理。"""

        return self.base_dir.joinpath(*relative_parts).resolve()

    @property
    def indexes_dir(self) -> Path:
        """索引文件目录。"""

        return self.storage_dir / "indexes"

    @property
    def data_dir(self) -> Path:
        """原始知识库目录。"""

        return self.storage_dir / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一的 Settings 实例。"""

    settings = Settings()  # 触发一次读取
    _ensure_directories(settings)
    return settings


def _ensure_directories(settings: Settings) -> None:
    """在应用启动时保证关键目录存在。"""

    for folder in (
        settings.storage_dir,
        settings.indexes_dir,
        settings.data_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)


