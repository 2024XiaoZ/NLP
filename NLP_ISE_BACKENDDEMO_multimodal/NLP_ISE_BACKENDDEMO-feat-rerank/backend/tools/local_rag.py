"""本地 RAG 工具门面（LangChain + FAISS 版本）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.core.config import get_settings
from backend.rag import rerank as rerank_module, vectorstore
from backend.utils.timing import Timer

logger = logging.getLogger(__name__)


async def search_local(query: str, topn: int) -> dict[str, Any]:
    """执行 LangChain 向量检索，返回证据与耗时。"""

    store = vectorstore.get_vectorstore()

    def _search() -> list[tuple[Any, float]]:
        return store.similarity_search_with_score(query, k=topn)

    with Timer() as retrieve_timer:
        docs_with_scores = await asyncio.to_thread(_search)

    items: list[dict[str, Any]] = []
    for doc, score in docs_with_scores:
        chunk_id = doc.metadata.get("chunk_id") or f"chunk-{len(items):04d}"
        section = doc.metadata.get("section") or "未命名章节"
        excerpt = _build_excerpt(doc.page_content)
        items.append(
            {
                "chunk_id": chunk_id,
                "section": section,
                "text": doc.page_content,
                "excerpt": excerpt,
                "score_init": float(score),
            }
        )

    # 重排序
    rerank_ms = 0
    settings = get_settings()
    if settings.enable_rerank:
        items, rerank_ms = rerank_module.rerank_local(
            query,
            items,
            vector_weight=settings.rerank_vector_weight,
            bm25_weight=settings.rerank_bm25_weight,
        )

    result = {
        "items": items,
        "latency": {
            "retrieve": retrieve_timer.elapsed_ms,
            "rerank": rerank_ms,
        },
    }
    logger.info(
        f"tool.local_rag: evidences={len(items)}, retrieve_ms={result['latency']['retrieve']}, rerank_ms={result['latency']['rerank']}",
        extra={
            "evidences": len(items),
            "retrieve_ms": result["latency"]["retrieve"],
            "rerank_ms": result["latency"]["rerank"],
        },
    )
    return result


def _build_excerpt(text: str, max_chars: int = 400) -> str:
    snippet = text.strip().replace("\n", " ")
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3] + "..."


