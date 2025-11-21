# -*- coding: utf-8 -*-
"""生成封装：调用 LangChain ChatOpenAI 并解析 JSON。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.llm_client import get_chat_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a rigorous assistant. Answer user questions based on the provided Context."
    "\n\n"
    "Important Principles:"
    "\n1. **Fully utilize Context**: If Context contains relevant information, you must use it to answer the question."
    "\n2. **Parse structured data**: If Context contains JSON, dictionaries, or other structured data, you must parse and extract key information."
    "\n3. **Don't easily say 'insufficient information'**: Only say insufficient information when Context completely cannot answer the question (no relevant content at all)."
    "\n4. **Extract key information**: Even if the data structure is complex, extract useful information (temperature, date, location, numbers, etc.)."
    "\n\n"
    "Answer Rules:"
    "\n1. Prioritize using information from Context to answer questions."
    "\n2. If Context is in JSON/dictionary format, parse and extract key field values."
    "\n3. When citing sources, mark them at the end: use chunk_id for local evidence, URL for web evidence."
    "\n4. All output must be valid JSON, containing answer, sources (text array), and confidence (0-1)."
    "\n\n"
    "Example:"
    "\n- Question: \"What's the weather today?\""
    "\n  Context: \"{'location': {'name': 'Current'}, 'current': {'temp_c': 25.5, 'temp_f': 77.8, 'condition': {'text': 'Clear'}}}\""
    "\n  Answer: \"Today's weather is clear with a temperature of 25.5°C (77.8°F) in the current location.\""
)


async def generate_answer(
    query: str,
    local_block: str,
    web_block: str,
) -> dict[str, Any]:
    """调用 LangChain ChatOpenAI 并返回解析结果。"""

    user_prompt = (
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        "--Local Evidence--\n"
        f"{local_block or 'No local evidence.'}\n\n"
        "--Web Evidence--\n"
        f"{web_block or 'No web evidence.'}\n\n"
        "Instructions:\n"
        "- If Context contains JSON format data (e.g., weather API response), parse the JSON and extract key information to answer.\n"
        "- If Context contains relevant information, fully utilize it to answer the question, even if the format is not plain text.\n"
        "- Only say insufficient information when Context completely cannot answer the question."
    )
    try:
        llm = get_chat_model()
        response = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = _extract_content(response.content)
        parsed = _safe_parse(content)
        logger.info(f"synth.success: confidence={parsed['confidence']}")
        return parsed
    except Exception as exc:  # pragma: no cover - LLM 故障
        logger.exception("synth.failed")
        return {
            "answer": "抱歉，生成模块暂时不可用，请稍后重试。",
            "sources": [],
            "confidence": 0.0,
        }


def _extract_content(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: list[str] = []
        for chunk in payload:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk.get("text", "")))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    return json.dumps(payload, ensure_ascii=False)


def _safe_parse(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"synth.json_decode_failed: content={content[:200]}")
        return {
            "answer": content,
            "sources": [],
            "confidence": 0.2,
        }
    answer = parsed.get("answer") or "抱歉，当前证据不足，无法给出答案。"
    sources = parsed.get("sources") or []
    confidence = float(parsed.get("confidence") or 0.4)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


