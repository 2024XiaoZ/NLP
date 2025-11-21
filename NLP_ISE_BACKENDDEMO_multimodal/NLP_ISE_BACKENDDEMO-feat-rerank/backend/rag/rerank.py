"""重排序模块：BM25 + 向量相似度混合排序，来源可信度评估，时效性评分。"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

from backend.utils.timing import Timer

logger = logging.getLogger(__name__)

# 权威域名列表（可根据需要扩展）
AUTHORITATIVE_DOMAINS: set[str] = {
    "wikipedia.org",
    "edu",
    "gov",
    "nature.com",
    "science.org",
    "arxiv.org",
    "ieee.org",
    "acm.org",
}


def rerank_local(
    query: str,
    items: list[dict[str, Any]],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> tuple[list[dict[str, Any]], int]:
    """对本地检索结果进行重排序：BM25 + 向量相似度混合。
    
    Args:
        query: 查询文本
        items: 检索结果列表，每个包含 text, score_init 等字段
        vector_weight: 向量相似度权重（默认 0.6）
        bm25_weight: BM25 分数权重（默认 0.4）
    
    Returns:
        (重排序后的 items, 耗时毫秒)
    """
    if not items:
        return items, 0
    
    with Timer() as timer:
        # 计算 BM25 分数
        bm25_scores = _compute_bm25_scores(query, items)
        
        # 归一化向量相似度分数（FAISS 返回的是距离，越小越好，需要转换）
        vector_scores = _normalize_vector_scores([item["score_init"] for item in items])
        
        # 混合排序
        for i, item in enumerate(items):
            combined_score = (
                vector_weight * vector_scores[i] + 
                bm25_weight * bm25_scores[i]
            )
            item["score_rerank"] = combined_score
            item["score_bm25"] = bm25_scores[i]
            item["score_vector"] = vector_scores[i]
        
        # 按重排序分数降序排列
        items.sort(key=lambda x: x["score_rerank"], reverse=True)
    
    logger.info(f"rerank.local: items={len(items)}, ms={timer.elapsed_ms}")
    return items, timer.elapsed_ms


def rerank_web(
    query: str,
    items: list[dict[str, Any]],
    recency_weight: float = 0.3,
    authority_weight: float = 0.3,
    relevance_weight: float = 0.4,
) -> tuple[list[dict[str, Any]], int]:
    """对网络检索结果进行重排序：时效性 + 来源可信度 + 相关性。
    
    Args:
        query: 查询文本
        items: 检索结果列表，每个包含 url, snippet, time, score_init 等字段
        recency_weight: 时效性权重（默认 0.3）
        authority_weight: 权威性权重（默认 0.3）
        relevance_weight: 相关性权重（默认 0.4）
    
    Returns:
        (重排序后的 items, 耗时毫秒)
    """
    if not items:
        return items, 0
    
    with Timer() as timer:
        # 计算各项分数
        recency_scores = _compute_recency_scores(items)
        authority_scores = _compute_authority_scores(items)
        relevance_scores = _normalize_vector_scores([item.get("score_init", 0.0) for item in items])
        
        # 混合排序
        for i, item in enumerate(items):
            combined_score = (
                recency_weight * recency_scores[i] +
                authority_weight * authority_scores[i] +
                relevance_weight * relevance_scores[i]
            )
            item["score_rerank"] = combined_score
            item["score_recency"] = recency_scores[i]
            item["score_authority"] = authority_scores[i]
            item["score_relevance"] = relevance_scores[i]
        
        # 按重排序分数降序排列
        items.sort(key=lambda x: x["score_rerank"], reverse=True)
    
    logger.info(f"rerank.web: items={len(items)}, ms={timer.elapsed_ms}")
    return items, timer.elapsed_ms


def _compute_bm25_scores(query: str, items: list[dict[str, Any]]) -> list[float]:
    """计算 BM25 分数（简化版本）。"""
    query_terms = _tokenize(query.lower())
    if not query_terms:
        return [0.0] * len(items)
    
    # 计算文档频率
    doc_freqs: dict[str, int] = {}
    for item in items:
        text = item.get("text", "").lower()
        terms = set(_tokenize(text))
        for term in query_terms:
            if term in terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
    
    # BM25 参数
    k1, b = 1.5, 0.75
    avg_doc_len = sum(len(item.get("text", "")) for item in items) / max(len(items), 1)
    
    scores: list[float] = []
    for item in items:
        text = item.get("text", "")
        doc_terms = _tokenize(text.lower())
        doc_len = len(text)
        score = 0.0
        
        for term in query_terms:
            if term not in doc_terms:
                continue
            tf = doc_terms.count(term)
            df = doc_freqs.get(term, 1)
            idf = max(0, (len(items) - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1))))
        
        scores.append(score)
    
    # 归一化到 [0, 1]
    max_score = max(scores) if scores else 1.0
    return [s / max_score if max_score > 0 else 0.0 for s in scores]


def _normalize_vector_scores(scores: list[float]) -> list[float]:
    """归一化向量相似度分数（距离转相似度，并归一化到 [0, 1]）。"""
    if not scores:
        return []
    
    # FAISS 返回的是距离（L2 或内积），越小越好
    # 转换为相似度：sim = 1 / (1 + distance) 或使用负距离
    # 这里假设是 L2 距离，使用倒数转换
    similarities = [1.0 / (1.0 + abs(s)) for s in scores]
    
    # 归一化到 [0, 1]
    min_sim = min(similarities) if similarities else 0.0
    max_sim = max(similarities) if similarities else 1.0
    if max_sim == min_sim:
        return [1.0] * len(similarities)
    
    return [(s - min_sim) / (max_sim - min_sim) for s in similarities]


def _compute_recency_scores(items: list[dict[str, Any]]) -> list[float]:
    """计算时效性分数（基于发布时间）。"""
    now = dt.datetime.utcnow()
    scores: list[float] = []
    
    for item in items:
        time_str = item.get("time", "")
        if not time_str:
            scores.append(0.5)  # 无时间信息，给中等分数
            continue
        
        try:
            # 解析 ISO 格式时间
            # 处理 "Z" 后缀（UTC 时间）
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            elif "Z" in time_str:
                time_str = time_str.replace("Z", "+00:00")
            
            # 尝试解析 ISO 格式
            pub_time = dt.datetime.fromisoformat(time_str)
            
            # 如果有时区信息，转换为 UTC 并移除时区
            if pub_time.tzinfo:
                pub_time = pub_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
            
            # 计算天数差
            days_diff = (now - pub_time).days
            
            # 时效性分数：越新分数越高（30天内为1.0，超过1年为0.1）
            if days_diff <= 30:
                score = 1.0
            elif days_diff <= 365:
                score = 1.0 - (days_diff - 30) / 365 * 0.9
            else:
                score = 0.1
            
            scores.append(max(0.0, min(1.0, score)))
        except (ValueError, TypeError):
            scores.append(0.5)  # 解析失败，给中等分数
    
    return scores


def _compute_authority_scores(items: list[dict[str, Any]]) -> list[float]:
    """计算来源权威性分数。"""
    scores: list[float] = []
    
    for item in items:
        url = item.get("url", "")
        if not url:
            scores.append(0.5)
            continue
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # 检查是否为权威域名
            is_authoritative = any(
                domain.endswith(f".{auth}") or domain == auth 
                for auth in AUTHORITATIVE_DOMAINS
            )
            
            # 权威域名得高分，其他域名根据 TLD 判断
            if is_authoritative:
                score = 1.0
            elif domain.endswith(".edu") or domain.endswith(".gov"):
                score = 0.9
            elif domain.endswith(".org"):
                score = 0.7
            elif domain.endswith(".com") or domain.endswith(".net"):
                score = 0.6
            else:
                score = 0.5
            
            scores.append(score)
        except Exception:
            scores.append(0.5)
    
    return scores


def _tokenize(text: str) -> list[str]:
    """简单的分词（英文）。"""
    # 移除标点，转小写，分词
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 1]

