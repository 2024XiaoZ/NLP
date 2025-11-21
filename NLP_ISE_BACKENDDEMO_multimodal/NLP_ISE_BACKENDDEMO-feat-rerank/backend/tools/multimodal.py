"""多模态输入处理工具：支持图像+文本的联合理解。"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.utils.timing import Timer

logger = logging.getLogger(__name__)


async def process_image_query(
    image_path: str | Path,
    query: str,
) -> dict[str, Any]:
    """
    处理图像+文本的多模态查询。
    
    Args:
        image_path: 图像文件路径（支持 jpg, png, webp 等）
        query: 用户的文本问题
        
    Returns:
        包含答案和元数据的字典
    """
    settings = get_settings()
    
    with Timer() as timer:
        try:
            # 1. 读取并编码图像
            image_data = _encode_image(image_path)
            
            # 2. 构建多模态消息
            model = _get_vision_model()
            message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            
            # 3. 调用视觉模型
            response = await model.ainvoke([message])
            answer = response.content
            
            logger.info(
                f"multimodal.success: query_len={len(query)}, "
                f"elapsed_ms={timer.elapsed_ms}"
            )
            
            return {
                "answer": answer,
                "image_path": str(image_path),
                "query": query,
                "latency_ms": timer.elapsed_ms,
                "confidence": 0.85,  # 视觉模型的置信度可以根据需求调整
            }
            
        except Exception as exc:
            logger.exception("multimodal.failure")
            raise RuntimeError(f"多模态处理失败: {exc}") from exc


def _encode_image(image_path: str | Path) -> str:
    """将图像编码为 base64 字符串。"""
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 使用 PIL 打开图像并转换为 RGB（确保兼容性）
    with Image.open(path) as img:
        # 如果图像太大，可以调整大小以节省 token
        max_size = 2048
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 转换为 RGB（去除 alpha 通道）
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 编码为 base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_vision_model() -> ChatOpenAI:
    """获取支持视觉的 ChatOpenAI 模型实例。"""
    settings = get_settings()
    
    if not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，无法调用视觉模型。")
    
    # 使用支持视觉的模型（如 gpt-4o, gpt-4-vision-preview 等）
    vision_model = settings.llm_vision_model or "gpt-4o-mini"
    
    return ChatOpenAI(
        model=vision_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        openai_api_key=settings.llm_api_key,
        base_url=str(settings.llm_base_url),
    )