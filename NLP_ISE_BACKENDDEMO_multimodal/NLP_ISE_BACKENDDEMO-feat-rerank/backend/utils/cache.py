"""内存级 TTL 缓存实现。"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import threading
import time
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def cache_key(fn_name: str, *args: Any, **kwargs: Any) -> str:
    """根据函数名与参数生成稳定的缓存键。"""

    payload = json.dumps(
        {
            "fn": fn_name,
            "args": args,
            "kwargs": kwargs,
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TTLMemoryCache:
    """线程安全的 TTL Cache，可后续扩展为 Redis。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """读取缓存；过期后自动清理。"""

        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """写入缓存。"""

        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        """清空缓存。"""

        with self._lock:
            self._store.clear()


def async_ttl_cache(ttl_seconds: int) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """装饰异步函数的 TTL 缓存。"""

    cache = TTLMemoryCache()

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = cache_key(fn.__qualname__, *args, **kwargs)
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = await fn(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator


