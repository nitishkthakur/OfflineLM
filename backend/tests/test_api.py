"""Integration tests for the FastAPI endpoints (mocked backends)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# We patch the backend before importing main so the registry is intact.
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    # Patch config to use temp dirs
    import main as app_module
    monkeypatch.setattr(app_module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(app_module, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "EXPORTS_DIR", tmp_path / "exports")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "uploads").mkdir()
    (tmp_path / "exports").mkdir()

    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_get_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "default_model" in data


def test_get_backends(client):
    resp = client.get("/backends")
    assert resp.status_code == 200
    data = resp.json()
    assert "backends" in data
    assert "default_backend" in data
    ids = [b["id"] for b in data["backends"]]
    assert "deep_agent" in ids
    assert "council" in ids


def test_get_artifacts_empty(client):
    resp = client.get("/artifacts")
    assert resp.status_code == 200
    assert resp.json()["artifacts"] == []


def test_chat_unknown_backend(client):
    """Requesting a non-existent backend yields a streaming error event."""
    resp = client.post(
        "/chat",
        json={
            "message": "Hello",
            "model_id": "any/model",
            "backend_id": "does_not_exist",
        },
        stream=True,
    )
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    events = [json.loads(l[6:]) for l in lines]
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1


async def _mock_council_stream(**kwargs):
    yield {"type": "progress", "step": 1, "description": "Phase 1"}
    yield {"type": "content", "content": "Paris is the capital."}
    yield {"type": "done", "conversation_id": kwargs.get("conversation_id", ""), "total_steps": 3}


def test_chat_council_backend_mocked(client):
    """Council backend can be selected via backend_id."""
    with patch("agents.council.CouncilBackend.stream", side_effect=_mock_council_stream):
        resp = client.post(
            "/chat",
            json={
                "message": "What is the capital of France?",
                "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
                "backend_id": "council",
                "backend_config": {
                    "council_models": [
                        "nvidia/nemotron-3-super-120b-a12b:free",
                        "nvidia/nemotron-nano-9b-v2:free",
                    ],
                    "chairman_model": "nvidia/nemotron-3-super-120b-a12b:free",
                },
            },
        )
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    events = [json.loads(l[6:]) for l in lines]
    types = [e["type"] for e in events]
    assert "content" in types
    assert "done" in types
