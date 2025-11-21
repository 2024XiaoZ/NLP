"""路由门面：对外暴露统一 route 接口。"""

from __future__ import annotations

from backend.agent.router_llm import llm_route
from backend.schemas.common import RoutingDecision


async def route(query: str) -> RoutingDecision:
    """调用 LLM 路由器，保留原有调用接口。"""

    return await llm_route(query)


