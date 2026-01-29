import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import MetricsCollector


def test_record_success():
    m = MetricsCollector()
    m.record("ocr", True, 120.5)
    stats = m.get("ocr")
    assert stats["total_requests"] == 1
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 0
    assert stats["avg_latency_ms"] == 120.5


def test_record_failure():
    m = MetricsCollector()
    m.record("llm", False, 50.0)
    stats = m.get("llm")
    assert stats["failed_requests"] == 1
    assert stats["success_rate"] == 0.0


def test_success_rate():
    m = MetricsCollector()
    m.record("op", True, 10.0)
    m.record("op", True, 20.0)
    m.record("op", False, 30.0)
    stats = m.get("op")
    assert round(stats["success_rate"], 4) == round(2 / 3, 4)


def test_summary_contains_all_operations():
    m = MetricsCollector()
    m.record("ocr", True, 100.0)
    m.record("llm", True, 200.0)
    summary = m.summary()
    assert "ocr" in summary
    assert "llm" in summary


def test_avg_latency():
    m = MetricsCollector()
    m.record("op", True, 100.0)
    m.record("op", True, 200.0)
    assert m.get("op")["avg_latency_ms"] == 150.0

