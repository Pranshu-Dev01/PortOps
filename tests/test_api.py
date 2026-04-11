"""
API integration tests for PortOps-LLM server endpoints.
"""

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_reset_default_payload_shape():
    response = client.post("/reset", json={})
    assert response.status_code == 200
    payload = response.json()
    assert "yard_text" in payload
    assert "inbound_queue" in payload
    assert "outbound_requests" in payload


def test_reset_rejects_invalid_task_id():
    response = client.post("/reset", json={"task_id": 9, "seed": 42})
    assert response.status_code == 422


def test_step_rejects_empty_command():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    response = client.post("/step", json={"command": "   "})
    assert response.status_code == 400


def test_step_rejects_invalid_command_format():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    response = client.post("/step", json={"command": "grab C01"})
    assert response.status_code == 400


def test_state_shape():
    client.post("/reset", json={"task_id": 1, "seed": 42})
    response = client.get("/state")
    assert response.status_code == 200
    payload = response.json()
    assert "yard" in payload
    assert all(f"bay_{i}" in payload["yard"] for i in range(1, 6))
    assert "step_count" in payload
    assert "task_id" in payload
