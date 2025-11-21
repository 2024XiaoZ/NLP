"""核心模块初始化。

该包承载配置、日志等运行时基础设施。
"""

from __future__ import annotations

from .config import Settings, get_settings
from .logging import setup_logging

__all__ = ["Settings", "get_settings", "setup_logging"]


