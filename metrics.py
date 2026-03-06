import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class ServiceMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_requests, 1)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)

    def to_dict(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
        }


class MetricsCollector:
    def __init__(self):
        self._metrics: Dict[str, ServiceMetrics] = defaultdict(ServiceMetrics)

    def record(self, operation: str, success: bool, latency_ms: float) -> None:
        m = self._metrics[operation]
        m.total_requests += 1
        m.total_latency_ms += latency_ms
        if success:
            m.successful_requests += 1
        else:
            m.failed_requests += 1
        logger.debug(f"[metrics] {operation} success={success} latency={latency_ms:.1f}ms")

    def get(self, operation: str) -> Dict:
        return self._metrics[operation].to_dict()

    def summary(self) -> Dict:
        return {op: m.to_dict() for op, m in self._metrics.items()}


metrics = MetricsCollector()


class timer:
    """Context manager to measure elapsed time in ms."""
    def __init__(self, operation: str, success_flag: list):
        self.operation = operation
        self.success_flag = success_flag
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, *_):
        elapsed = (time.time() - self._start) * 1000
        success = exc_type is None and (not self.success_flag or self.success_flag[0])
        metrics.record(self.operation, success, elapsed)



