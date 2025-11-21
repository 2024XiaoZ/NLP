"""Web 搜索工具封装（Tavily 版本）。"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from typing import Any

from langchain_tavily import TavilySearch

from backend.core.config import get_settings
from backend.rag import rerank as rerank_module
from backend.utils.cache import TTLMemoryCache, cache_key
from backend.utils.timing import Timer

logger = logging.getLogger(__name__)
_CACHE = TTLMemoryCache()


async def search_web(query: str, k: int) -> dict[str, Any]:
    """调用 Tavily Search 并加上 TTL 缓存。"""

    ttl = get_settings().cache_ttl_seconds
    key = cache_key("web_search", query, k)
    cached = _CACHE.get(key)
    if cached:
        return cached

    with Timer() as timer:
        try:
            raw_results = await asyncio.to_thread(_invoke_tavily, query, k)
        except Exception as exc:  # pragma: no cover
            logger.warning("tool.web.error", exc_info=exc)
            raw_results = {"results": []}

    normalized = _normalize_results(raw_results, k)
    
    # 重排序
    rerank_ms = 0
    settings = get_settings()
    if settings.enable_rerank:
        normalized, rerank_ms = rerank_module.rerank_web(
            query,
            normalized,
            recency_weight=settings.rerank_recency_weight,
            authority_weight=settings.rerank_authority_weight,
            relevance_weight=settings.rerank_relevance_weight,
        )
    
    payload = {
        "items": normalized,
        "latency": {"retrieve": timer.elapsed_ms, "rerank": rerank_ms},
    }
    _CACHE.set(key, payload, ttl)
    logger.info(
        f"tool.web: evidences={len(normalized)}, retrieve_ms={timer.elapsed_ms}, rerank_ms={rerank_ms}",
        extra={
            "evidences": len(normalized),
            "retrieve_ms": timer.elapsed_ms,
            "rerank_ms": rerank_ms,
        },
    )
    return payload


def _invoke_tavily(query: str, max_results: int) -> Any:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("缺少 TAVILY_API_KEY，无法执行实时搜索。")
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    tool = TavilySearch(max_result=max_results)
    return tool.invoke({"query": query})


def _normalize_results(payload: Any, limit: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            candidates = results
        else:
            candidates = []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []

    normalized: list[dict[str, Any]] = []
    for item in candidates[:limit]:
        # 获取原始内容
        raw_content = item.get("content") or item.get("summary") or ""
        snippet = raw_content.strip()
        
        # 尝试清理 JSON 格式的 snippet，使其更易读
        if snippet.startswith("{") or (snippet.startswith("'") and "{" in snippet):
            # 尝试解析 JSON 或类似格式，转换为更友好的文本
            try:
                import json
                import ast
                # 处理单引号 Python dict 字符串（如 "{'key': 'value'}"）
                if snippet.startswith("'") or (snippet.startswith("{") and "'" in snippet[:50]):
                    try:
                        # 尝试使用 ast.literal_eval 解析 Python dict
                        parsed = ast.literal_eval(snippet)
                    except (ValueError, SyntaxError):
                        # 如果失败，尝试转换为标准 JSON
                        # 替换单引号为双引号（但要注意处理字符串中的引号）
                        cleaned = snippet.replace("'", '"')
                        parsed = json.loads(cleaned)
                else:
                    # 标准 JSON 格式
                    parsed = json.loads(snippet)
                
                # 转换为友好的文本格式
                snippet = _format_structured_data(parsed)
            except (json.JSONDecodeError, ValueError, SyntaxError, AttributeError):
                # 如果解析失败，保持原样，但尝试简单清理
                # 移除多余的转义字符
                snippet = snippet.replace("\\", "").strip()
        
        normalized.append(
            {
                "type": "web",
                "title": item.get("title") or item.get("url") or "未命名网页",
                "url": item.get("url") or "",
                "snippet": snippet[:400],  # 限制长度
                "time": item.get("published_date") or item.get("date") or _now_iso(),
                "score_init": float(item.get("score") or 0.0),
            }
        )
    return normalized


def _format_structured_data(data: Any, max_depth: int = 2, current_depth: int = 0) -> str:
    """将结构化数据转换为易读的文本格式"""
    if current_depth >= max_depth:
        return str(data)[:200]
    
    if isinstance(data, dict):
        parts = []
        for key, value in list(data.items())[:10]:  # 限制条目数
            if isinstance(value, (dict, list)):
                value_str = _format_structured_data(value, max_depth, current_depth + 1)
            else:
                value_str = str(value)
            parts.append(f"{key}: {value_str}")
        return "; ".join(parts)
    elif isinstance(data, list):
        parts = []
        for item in data[:5]:  # 限制列表长度
            parts.append(_format_structured_data(item, max_depth, current_depth + 1))
        return ", ".join(parts)
    else:
        return str(data)


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


