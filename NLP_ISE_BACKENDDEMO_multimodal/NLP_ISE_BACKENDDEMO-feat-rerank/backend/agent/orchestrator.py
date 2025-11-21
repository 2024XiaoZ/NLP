"""Agent 编排逻辑。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.agent import router, synth
from backend.core.config import get_settings
from backend.rag import aggregator
from backend.schemas.common import FinalResponse, LatencyBreakdown, RoutingDecision
from backend.tools import local_rag, web as web_tool
from backend.utils.timing import Timer

logger = logging.getLogger(__name__)


async def answer(query: str) -> FinalResponse:
    """主入口：路由 → 检索 → 聚合 → 生成。"""

    route_decision = await router.route(query)
    latency = {"retrieve": 0, "rerank": 0, "generate": 0}
    local_hits: list[dict[str, Any]] = []
    web_hits: list[dict[str, Any]] = []

    with Timer() as total_timer:
        try:
            local_hits, web_hits, latency = await _execute_policy(query, route_decision, latency)
            agg_result = aggregator.aggregate_evidence(local_hits, web_hits)
            with Timer() as gen_timer:
                synth_result = await synth.generate_answer(
                    query,
                    agg_result["local_block"],
                    agg_result["web_block"],
                )
            latency["generate"] = gen_timer.elapsed_ms
            final_response = FinalResponse(
                answer=synth_result["answer"],
                sources=[*agg_result["local_sources"], *agg_result["web_sources"]],
                routing=route_decision,
                latency_ms=LatencyBreakdown(
                    retrieve=latency["retrieve"],
                    rerank=latency["rerank"],
                    generate=latency["generate"],
                    total=total_timer.elapsed_ms,
                ),
                confidence=float(synth_result.get("confidence", 0.0)),
            )
            logger.info(
                f"orchestrator.success: policy={route_decision.policy}, total_ms={final_response.latency_ms.total}"
            )
            return final_response
        except Exception as exc:
            logger.exception("orchestrator.failure")
            return _fallback_response(
                query=query,
                routing=route_decision,
                latency=latency,
                total_ms=total_timer.elapsed_ms,
                reason=str(exc),
            )


async def _execute_policy(
    query: str,
    routing: RoutingDecision,
    latency: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    policy = routing.policy
    settings = get_settings()

    if policy == "local":
        local_hits, local_latency = await _run_local(query, settings.local_top_k)
        latency["retrieve"] += local_latency["retrieve"]
        latency["rerank"] += local_latency["rerank"]
        return local_hits, [], latency

    if policy == "web":
        web_hits, web_latency = await _run_web(query, settings.local_top_k)
        latency["retrieve"] += web_latency["retrieve"]
        latency["rerank"] += web_latency["rerank"]
        return [], web_hits, latency

    # hybrid 策略：同时执行本地和网络检索
    if policy == "hybrid":
        local_task = asyncio.create_task(_run_local(query, settings.local_top_k))
        web_task = asyncio.create_task(_run_web(query, settings.local_top_k))
        local_result, web_result = await asyncio.gather(local_task, web_task, return_exceptions=True)
        local_hits: list[dict[str, Any]] = []
        web_hits: list[dict[str, Any]] = []
        if isinstance(local_result, Exception):
            logger.error("orchestrator.local_failed", exc_info=local_result)
        else:
            local_hits, local_latency = local_result
            latency["retrieve"] += local_latency["retrieve"]
            latency["rerank"] += local_latency["rerank"]
        if isinstance(web_result, Exception):
            logger.error("orchestrator.web_failed", exc_info=web_result)
        else:
            web_hits, web_latency = web_result
            latency["retrieve"] += web_latency["retrieve"]
            latency["rerank"] += web_latency["rerank"]
        return local_hits, web_hits, latency

    # 兜底策略：如果 policy 不在预期范围内（理论上不会发生，因为 schema 限制了类型）
    # 使用 hybrid 策略作为默认，复用 hybrid 逻辑
    logger.warning(f"orchestrator.unknown_policy: policy={policy}, falling back to hybrid")
    # 复用 hybrid 策略的逻辑
    policy = "hybrid"
    local_task = asyncio.create_task(_run_local(query, settings.local_top_k))
    web_task = asyncio.create_task(_run_web(query, settings.local_top_k))
    local_result, web_result = await asyncio.gather(local_task, web_task, return_exceptions=True)
    local_hits: list[dict[str, Any]] = []
    web_hits: list[dict[str, Any]] = []
    if isinstance(local_result, Exception):
        logger.error("orchestrator.local_failed", exc_info=local_result)
    else:
        local_hits, local_latency = local_result
        latency["retrieve"] += local_latency["retrieve"]
        latency["rerank"] += local_latency["rerank"]
    if isinstance(web_result, Exception):
        logger.error("orchestrator.web_failed", exc_info=web_result)
    else:
        web_hits, web_latency = web_result
        latency["retrieve"] += web_latency["retrieve"]
        latency["rerank"] += web_latency["rerank"]
    return local_hits, web_hits, latency


async def _run_local(query: str, topn: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    result = await local_rag.search_local(query, topn)
    return result["items"], result["latency"]


async def _run_web(query: str, k: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    result = await web_tool.search_web(query, k)
    return result["items"], result["latency"]


def _fallback_response(
    query: str,
    routing: RoutingDecision,
    latency: dict[str, int],
    total_ms: int,
    reason: str,
) -> FinalResponse:
    message = f"系统暂时无法处理您的请求（原因：{reason}）。请稍后再试。"
    return FinalResponse(
        answer=message,
        sources=[],
        routing=routing,
        latency_ms=LatencyBreakdown(
            retrieve=latency.get("retrieve", 0),
            rerank=latency.get("rerank", 0),
            generate=latency.get("generate", 0),
            total=total_ms,
        ),
        confidence=0.0,
    )


