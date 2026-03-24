import hashlib
import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResultCache:
    """Simple in-memory TTL cache for LLM results."""

    def __init__(self, ttl: int = 3600):
        self._store: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl

    def _key(self, text: str, operation: str) -> str:
        return hashlib.sha256(f"{operation}:{text}".encode()).hexdigest()

    def get(self, text: str, operation: str) -> Optional[Any]:
        key = self._key(text, operation)
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        logger.debug(f"Cache hit for {operation}")
        return value

    def set(self, text: str, operation: str, value: Any) -> None:
        key = self._key(text, operation)
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)




