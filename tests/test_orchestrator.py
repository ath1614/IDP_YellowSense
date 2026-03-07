import sys
import os
import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_orchestrator():
    with patch("orchestrator.IDPOrchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.process_document = AsyncMock(return_value={
            "document_type": "APAR",
            "confidence": 0.85,
            "ocr_result": {"total_pages": 1, "pages": [{"page_number": 1, "text": "test"}]},
            "structured_data": {"officer_name": "Test Officer"},
            "processing_status": "completed",
            "processing_time_ms": 1500.0,
        })
        yield instance


def test_health_endpoint():
    from fastapi.testclient import TestClient
    import orchestrator as orch_module
    client = TestClient(orch_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_process_rejects_non_pdf():
    from fastapi.testclient import TestClient
    import orchestrator as orch_module
    client = TestClient(orch_module.app)
    response = client.post(
        "/process",
        files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
    )
    assert response.status_code == 400


def test_metrics_endpoint():
    from fastapi.testclient import TestClient
    import orchestrator as orch_module
    client = TestClient(orch_module.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


