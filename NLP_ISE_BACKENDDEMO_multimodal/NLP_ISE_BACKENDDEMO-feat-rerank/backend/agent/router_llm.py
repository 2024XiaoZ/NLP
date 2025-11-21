"""LLM 驱动的路由模块：规则优先，未命中再调用大模型。"""

from __future__ import annotations

import json
import logging
import re
from typing import Final

import httpx

from backend.core.config import get_settings
from backend.schemas.common import RoutingDecision
from backend.utils.cache import TTLMemoryCache, cache_key

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: Final[str] = (
    "You are an intent classifier that routes user questions to the correct knowledge source.\n"
    "You MUST output a strict JSON object with fields:\n"
    '- "policy": one of ["local", "web", "hybrid"]\n'
    '- "rationale": a short explanation\n\n'
    "Definitions:\n"
    '1. "local": The question refers to fictional entities or domain-specific concepts stored '
    "in the local knowledge base.\n"
    "   Examples: Sereleia, Xylos, Elara Vance, Vance Protocol, etc.\n"
    '2. "web": The question requires real-world, time-sensitive, factual, or up-to-date information.\n'
    "   Examples: news, AI updates, weather, stock prices, traffic, today's events.\n"
    '3. "hybrid": The question mixes fictional/local knowledge with real-world/timely information.\n'
    '   Examples: "Explain the Vance Protocol and give the latest real-world impact."\n'
    '   Also use "hybrid" when the question could benefit from both sources.\n\n'
    "Your job: Infer the correct policy from the semantics of the question.\n"
    "Respond with JSON only. No commentary."
)

LOCAL_KEYWORDS: Final[set[str]] = {
    "sereleia",
    "xylos",
    "elara vance",
    "vance protocol",
    "aether core",
    "lys harbor",
    "dr. elara",
    "sereleian",
}

WEB_KEYWORDS: Final[set[str]] = {
    "today",
    "latest",
    "price",
    "prices",
    "weather",
    "traffic",
    "now",
    "breaking",
    "2024",
    "2025",
    "trend",
    "news",
    "stock",
}

_VALID_POLICIES: Final[set[str]] = {"local", "web", "hybrid"}
_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_CACHE = TTLMemoryCache()


async def llm_route(query: str) -> RoutingDecision:
    """根据规则+LLM 判断策略，返回 RoutingDecision。"""

    normalized = query.lower()
    local_hit = _match_keyword(normalized, LOCAL_KEYWORDS)
    web_hit = _match_keyword(normalized, WEB_KEYWORDS)
    
    # 如果同时命中本地和实时关键词，需要调用 LLM 判断是否为混合问题
    if local_hit and web_hit:
        # 跳过规则匹配，直接调用 LLM 判断
        pass
    elif local_hit:
        # 仅命中本地关键词，直接返回 local
        rationale = f"命中本地关键词 `{local_hit}`，无需调用 LLM。"
        return RoutingDecision(policy="local", rationale=rationale)
    elif web_hit:
        # 仅命中实时关键词，直接返回 web
        rationale = f"命中实时关键词 `{web_hit}`，直接走 Web 检索。"
        return RoutingDecision(policy="web", rationale=rationale)

    cache_key_str = cache_key("router_llm.route", normalized)
    cached = _CACHE.get(cache_key_str)
    if cached:
        return cached

    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，无法执行路由判别。")

    url = str(settings.llm_base_url).rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Question: "{query}"'},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": min(settings.llm_max_tokens, 200),
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(10.0, read=20.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # pragma: no cover - 网络层异常
        logger.exception("router.llm_request_failed")
        return _fallback("LLM 路由请求失败，回退 hybrid。")

    content = _extract_content(data)
    decision = _safe_parse_decision(content)
    _CACHE.set(cache_key_str, decision, settings.cache_ttl_seconds)
    logger.info(f"router.llm_decision: policy={decision.policy}")
    return decision


def _match_keyword(text: str, keywords: set[str]) -> str | None:
    for word in keywords:
        if word in text:
            return word
    return None


def _extract_content(response: dict[str, object]) -> str:
    choices = response.get("choices") or []
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return json.dumps(response, ensure_ascii=False)


def _safe_parse_decision(content: str) -> RoutingDecision:
    match = _JSON_PATTERN.search(content)
    raw_json = match.group(0) if match else content
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning(f"router.llm_json_decode_failed: content={content[:200]}")
        return _fallback("LLM 输出无法解析，回退 hybrid。")

    policy = str(parsed.get("policy", "")).lower()
    rationale = parsed.get("rationale") or "LLM 未提供理由。"
    if policy not in _VALID_POLICIES:
        logger.warning(f"router.llm_invalid_policy: policy={policy}, content={parsed}")
        return _fallback("LLM 输出非法 policy，回退 hybrid。")
    return RoutingDecision(policy=policy, rationale=str(rationale))


def _fallback(reason: str) -> RoutingDecision:
    """回退策略：使用 hybrid 作为默认策略，因为它会同时尝试两种源。"""
    return RoutingDecision(policy="hybrid", rationale=reason)



