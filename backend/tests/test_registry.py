"""Tests for the agent registry and base class."""
import pytest
from agents import REGISTRY, get_backend, list_backends, register
from agents.base import AgentBackend


def test_registry_has_expected_backends():
    assert "deep_agent" in REGISTRY
    assert "council" in REGISTRY


def test_get_backend_returns_instance():
    backend = get_backend("deep_agent")
    assert isinstance(backend, AgentBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("does_not_exist")


def test_list_backends_structure():
    backends = list_backends()
    assert isinstance(backends, list)
    assert len(backends) >= 2
    for b in backends:
        assert "id" in b
        assert "name" in b
        assert "description" in b
        assert "config_schema" in b


def test_register_decorator():
    @register("_test_backend")
    class _TestBackend(AgentBackend):
        name = "Test"
        description = "Test backend"

        async def stream(self, messages, model_id, tools, backend_config, conversation_id):
            yield self.ev_content("hello")

    assert "_test_backend" in REGISTRY
    b = get_backend("_test_backend")
    assert b.id == "_test_backend"
    # clean up
    del REGISTRY["_test_backend"]


def test_base_event_helpers():
    b = get_backend("deep_agent")
    assert b.ev_content("hi") == {"type": "content", "content": "hi"}
    assert b.ev_progress(1, "step") == {"type": "progress", "step": 1, "description": "step"}
    assert b.ev_tool_start("search", "desc") == {
        "type": "tool_start", "tool": "search", "description": "desc"
    }
    assert b.ev_tool_end("search") == {"type": "tool_end", "tool": "search"}
    assert b.ev_done("abc", 3) == {"type": "done", "conversation_id": "abc", "total_steps": 3}
    assert b.ev_error("oops") == {"type": "error", "error": "oops"}


def test_council_config_schema():
    backend = get_backend("council")
    schema = backend.config_schema
    assert "council_models" in schema
    assert "chairman_model" in schema
    assert schema["council_models"]["type"] == "model_list"
    assert schema["chairman_model"]["type"] == "model"


def test_deep_agent_config_schema():
    backend = get_backend("deep_agent")
    schema = backend.config_schema
    assert "recursion_limit" in schema
    assert schema["recursion_limit"]["type"] == "integer"
