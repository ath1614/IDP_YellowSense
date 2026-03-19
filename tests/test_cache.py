import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.app.cache import ResultCache


def test_cache_set_and_get():
    cache = ResultCache(ttl=60)
    cache.set("hello", "classify", {"document_type": "APAR"})
    result = cache.get("hello", "classify")
    assert result == {"document_type": "APAR"}


def test_cache_miss():
    cache = ResultCache(ttl=60)
    assert cache.get("missing", "classify") is None


def test_cache_expiry():
    cache = ResultCache(ttl=1)
    cache.set("text", "apar", {"officer_name": "Test"})
    time.sleep(1.1)
    assert cache.get("text", "apar") is None


def test_cache_clear():
    cache = ResultCache(ttl=60)
    cache.set("a", "classify", {"x": 1})
    cache.set("b", "apar", {"y": 2})
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0


def test_different_operations_different_keys():
    cache = ResultCache(ttl=60)
    cache.set("same_text", "classify", {"type": "APAR"})
    cache.set("same_text", "apar", {"officer_name": "John"})
    assert cache.get("same_text", "classify") != cache.get("same_text", "apar")



