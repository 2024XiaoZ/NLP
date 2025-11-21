"""日志初始化模块。

使用标准 logging + JSON Formatter，方便集中式观测。
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime
from typing import Any

LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "backend.core.logging.JsonFormatter",
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
        }
    },
    "root": {
        "handlers": ["stdout"],
        "level": "INFO",
    },
}


class JsonFormatter(logging.Formatter):
    """简单的 JSON 格式化器，统一输出字段。"""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # 附加额外字段
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload:
                continue
            if key in {"args", "msg", "exc_info", "exc_text", "stack_info"}:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """在应用入口调用，确保日志配置生效。"""

    logging.config.dictConfig(LOGGING_CONFIG)


