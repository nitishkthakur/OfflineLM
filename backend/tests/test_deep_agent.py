"""DeepAgent backend tests.

These exercise the full stack against a live Ollama and verify:
  1. The backend is registered and discoverable via /backends.
  2. Config plumbing (subagents, summarization model, recursion_limit) reaches
     the backend.
  3. A tool-capable Ollama model actually runs through the agent and produces
     output + tool calls.
  4. Non-tool-capable models are rejected with a clear error.
  5. NO outbound network calls escape to non-Ollama hosts. A socket-level
     guard fails the test if anything tries to reach Anthropic/OpenAI/OpenRouter.

Skipped automatically if Ollama isn't reachable or required models aren't
pulled locally.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents as agent_registry  # noqa: E402
from tools import make_tools  # noqa: E402


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _ollama_has(model_id: str) -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        return model_id in names
    except Exception:
        return False


# ── Unit: registry ────────────────────────────────────────────────────────────

def test_deep_agent_registered():
    ids = [b["id"] for b in agent_registry.list_backends()]
    assert "deep_agent" in ids, f"deep_agent not in registry: {ids}"


def test_deep_agent_config_schema():
    meta = next(b for b in agent_registry.list_backends() if b["id"] == "deep_agent")
    assert "recursion_limit" in meta["config_schema"]


# ── Unit: helpers ─────────────────────────────────────────────────────────────

def test_subagent_name_from_metadata():
    from agents.deep_agent import _subagent_name_from_metadata
    assert _subagent_name_from_metadata(None) is None
    assert _subagent_name_from_metadata({}) is None
    assert _subagent_name_from_metadata({"tags": ["subagent:researcher"]}) == "researcher"
    assert _subagent_name_from_metadata(
        {"langgraph_checkpoint_ns": "agent/tools/:task:abc123"}
    ) == "subagent"
    # Generic node names are ignored
    assert _subagent_name_from_metadata({"langgraph_node": "agent"}) is None


def test_strip_images():
    from agents.deep_agent import _strip_images
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            {"type": "text", "text": "world"},
        ]},
        {"role": "assistant", "content": "hi there"},
    ]
    out = _strip_images(msgs)
    assert out[0]["content"] == "hello\nworld"
    assert out[1]["content"] == "hi there"


def test_tool_description():
    from agents.deep_agent import _tool_description
    assert "researcher" in _tool_description("task", {"subagent_type": "researcher", "description": "look up bank rates"})
    assert "Searching" in _tool_description("internet_search", {"query": "fed rate 2026"})
    assert "write_todos" in _tool_description("write_todos", {"todos": [{"content": "x"}, {"content": "y"}]})


# ── Integration: live Ollama, no tool-capable model required for rejection ───

def test_rejects_non_tool_model():
    """A non-tool-capable model must fail fast with a clear error."""
    model_id = "gemma3:4b"
    if not _ollama_has(model_id):
        pytest.skip(f"{model_id} not pulled")

    be = agent_registry.get_backend("deep_agent")
    tools = make_tools(Path("artifacts"), web_search_enabled=False)
    errors = []

    async def go():
        async for ev in be.stream(
            messages=[{"role": "user", "content": "hi"}],
            model_id=model_id,
            tools=tools,
            backend_config={
                "ollama_options": {"temperature": 0.1, "num_ctx": 4096},
                "subagents": [],
                "summarization_model": None,
                "subagent_default_model": None,
                "recursion_limit": 50,
            },
            conversation_id="test-reject",
        ):
            if ev.get("type") == "error":
                errors.append(ev["error"])

    asyncio.run(go())
    assert errors, "expected an error event for a non-tool model"
    assert "tool-capable" in errors[0].lower() or "tools" in errors[0].lower()


# ── Integration: live run on a tool-capable Ollama model ─────────────────────

def test_deep_agent_runs_end_to_end():
    """Full turn through DeepAgent. Asserts output, tool usage, and no non-Ollama network access."""
    model_id = "qwen3:4b"
    if not _ollama_has(model_id):
        pytest.skip(f"{model_id} not pulled")

    be = agent_registry.get_backend("deep_agent")
    tools = make_tools(Path("artifacts"), web_search_enabled=False)

    # Network guard: any connection to a host NOT in OLLAMA_HOSTS fails.
    real_create_connection = socket.create_connection
    rogue_calls = []

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in OLLAMA_HOSTS and not host.startswith("127."):
            rogue_calls.append(host)
            raise AssertionError(f"Forbidden outbound connection to {host!r}")
        return real_create_connection(address, *args, **kwargs)

    events_by_type: dict[str, int] = {}
    content = ""
    tool_names: list[str] = []
    errors: list[str] = []

    async def go():
        nonlocal content
        async for ev in be.stream(
            messages=[{
                "role": "user",
                "content": "Compute 123 + 456 + 789 using the calculate tool. Then reply with just the final number."
            }],
            model_id=model_id,
            tools=tools,
            backend_config={
                "ollama_options": {"temperature": 0.1, "reasoning": False, "num_ctx": 8192},
                "subagents": [],               # keep it simple
                "general_purpose_agent": False,
                "summarization_model": None,   # → main model
                "subagent_default_model": None,
                "recursion_limit": 40,
            },
            conversation_id="test-e2e",
        ):
            t = ev.get("type", "?")
            events_by_type[t] = events_by_type.get(t, 0) + 1
            if t == "content":
                content += ev.get("content", "")
            elif t == "tool_start":
                tool_names.append(ev.get("tool", ""))
            elif t == "error":
                errors.append(ev.get("error", ""))

    with patch.object(socket, "create_connection", side_effect=guarded_create_connection):
        asyncio.run(go())

    assert not rogue_calls, f"agent attempted non-Ollama connections: {rogue_calls}"
    assert not errors, f"agent errored: {errors}"
    assert content.strip(), "agent produced no content"
    assert "1368" in content or "1,368" in content, f"wrong answer: {content[:300]}"
    # At least one tool event fired (calculate or planning/filesystem)
    assert events_by_type.get("tool_start", 0) >= 1, f"no tool_start events: {events_by_type}"
    assert "calculate" in tool_names, f"calculate not called. Tools: {tool_names}"


def test_deep_agent_config_wires_through_main():
    """The /chat route must merge config.deep_agent into backend_config."""
    from main import load_config
    cfg = load_config()
    assert "deep_agent" in cfg
    da = cfg["deep_agent"]
    assert "subagents" in da
    names = {s["name"] for s in da["subagents"]}
    assert {"researcher", "critic"}.issubset(names)
    for s in da["subagents"]:
        assert "include" in s
