"""证据聚合与上下文构建。"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from backend.schemas.common import LocalEvidence, WebEvidence

logger = logging.getLogger(__name__)


def aggregate_evidence(
    local_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
    local_budget: int = 2000,
    web_budget: int = 2000,
) -> dict[str, Any]:
    """整理证据并生成上下文文本块。"""

    normalized_local = _normalize_local(local_hits, max_chars=local_budget)
    normalized_web = _normalize_web(web_hits, max_chars=web_budget)

    local_block = _render_local_block(normalized_local, local_budget)
    web_block = _render_web_block(normalized_web, web_budget)

    return {
        "local_sources": normalized_local,
        "web_sources": normalized_web,
        "local_block": local_block,
        "web_block": web_block,
    }


def _normalize_local(hits: list[dict[str, Any]], max_chars: int) -> list[LocalEvidence]:
    seen: set[str] = set()
    sources: list[LocalEvidence] = []
    budget = max_chars
    for hit in hits:
        if hit["chunk_id"] in seen:
            continue
        excerpt = hit.get("excerpt") or hit.get("text", "")
        excerpt = excerpt.strip().replace("\n", " ")
        excerpt = excerpt[: min(len(excerpt), 400)]
        if budget <= 0:
            break
        sources.append(
            LocalEvidence(
                chunk_id=hit["chunk_id"],
                section=hit.get("section", "未知章节"),
                excerpt=excerpt,
            )
        )
        seen.add(hit["chunk_id"])
        budget -= len(excerpt)
    return sources


def _normalize_web(hits: list[dict[str, Any]], max_chars: int) -> list[WebEvidence]:
    seen: set[str] = set()
    sources: list[WebEvidence] = []
    budget = max_chars
    for hit in hits:
        url = hit.get("url")
        if not url or url in seen:
            continue
        snippet = hit.get("snippet", "").strip().replace("\n", " ")[:400]
        title = hit.get("title") or "未命名网页"
        time_str = hit.get("time") or _now_iso()
        sources.append(
            WebEvidence(
                title=title,
                url=url,
                snippet=snippet,
                time=time_str,
            )
        )
        seen.add(url)
        budget -= len(snippet)
        if budget <= 0:
            break
    return sources


def _render_local_block(sources: list[LocalEvidence], budget: int) -> str:
    if not sources:
        return "无本地证据。"
    lines: list[str] = []
    remaining = budget
    for src in sources:
        line = f"[{src.chunk_id}] {src.section}: {src.excerpt}"
        if remaining - len(line) <= 0:
            break
        lines.append(line)
        remaining -= len(line)
    return "\n".join(lines)


def _render_web_block(sources: list[WebEvidence], budget: int) -> str:
    if not sources:
        return "无网页证据。"
    lines: list[str] = []
    remaining = budget
    for src in sources:
        line = f"[{src.time}] {src.title} ({src.url}): {src.snippet}"
        if remaining - len(line) <= 0:
            break
        lines.append(line)
        remaining -= len(line)
    return "\n".join(lines)


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


