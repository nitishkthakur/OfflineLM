"""Integration test: verify the ReAct agent actually invokes tools on realistic
bank-statement-style prompts via a live Ollama backend.

Run with:
    backend/.venv/bin/python3 -m pytest backend/tests/test_tool_calling.py -s

These tests require a running Ollama instance with the listed models available.
They are skipped automatically if Ollama is unreachable or the model is missing.
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.react_agent import ReactAgentBackend  # noqa: E402
from tools import make_tools  # noqa: E402


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_has(model_id: str) -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        return model_id in names
    except Exception:
        return False


async def _run(model_id: str, prompt: str, reasoning: bool = False) -> dict:
    be = ReactAgentBackend()
    tools = make_tools(Path("artifacts"), web_search_enabled=False)
    msgs = [{"role": "user", "content": prompt}]
    tool_calls = []
    content = ""
    error = None
    async for ev in be.stream(
        messages=msgs,
        model_id=model_id,
        tools=tools,
        backend_config={
            "ollama_options": {
                "temperature": 0.1,
                "reasoning": reasoning,
                "num_ctx": 8192,
            }
        },
        conversation_id="pytest",
    ):
        t = ev.get("type")
        if t == "tool_start":
            tool_calls.append({"tool": ev.get("tool"), "args": ev.get("args")})
        elif t == "content":
            content += ev.get("content", "")
        elif t == "error":
            error = ev.get("error")
            break
    return {"tool_calls": tool_calls, "content": content, "error": error}


BANK_STATEMENT_PROMPT = (
    "Here are my withdrawals from my bank statement this month:\n"
    "  2026-04-02  Rent            1,450.00\n"
    "  2026-04-05  Groceries         287.34\n"
    "  2026-04-07  Electricity       118.22\n"
    "  2026-04-10  Internet           79.99\n"
    "  2026-04-14  Groceries         142.10\n"
    "  2026-04-18  Fuel               63.45\n"
    "  2026-04-22  Restaurant         56.80\n"
    "  2026-04-27  Pharmacy           38.99\n"
    "Please compute the exact total of all these withdrawals. "
    "You MUST use the calculate tool — do not do arithmetic in your head."
)


@pytest.mark.parametrize("model_id", ["qwen3:4b", "gpt-oss:latest", "qwen2.5:14b"])
def test_calculate_called_on_bank_statement(model_id):
    if not _ollama_has(model_id):
        pytest.skip(f"{model_id} not pulled in local Ollama")

    result = asyncio.run(_run(model_id, BANK_STATEMENT_PROMPT, reasoning=False))

    assert result["error"] is None, f"agent errored: {result['error']}"
    tool_names = [tc["tool"] for tc in result["tool_calls"]]
    assert "calculate" in tool_names, (
        f"{model_id} did NOT call calculate. Tools called: {tool_names}. "
        f"Response: {result['content'][:500]}"
    )
    # Expected total: 2_236.89
    assert "2,236.89" in result["content"] or "2236.89" in result["content"], (
        f"Final answer did not contain correct total. Got: {result['content'][:300]}"
    )


def test_pct_change_called():
    model_id = "qwen3:4b"
    if not _ollama_has(model_id):
        pytest.skip(f"{model_id} not pulled")
    prompt = (
        "My spending last month was $2,100 and this month is $2,450. "
        "What is the percentage change? Use the pct_change tool."
    )
    result = asyncio.run(_run(model_id, prompt))
    assert "pct_change" in [tc["tool"] for tc in result["tool_calls"]]


def test_calculate_primitive():
    """The underlying calculate function is correct regardless of any LLM."""
    from tools import calculate
    # Non-integer floats: general %g formatting, no thousands separator
    assert calculate("1,450.00 + 287.34 + 118.22 + 79.99 + 142.10 + 63.45 + 56.80 + 38.99") == "2236.89"
    # Integers: thousands separator
    assert calculate("2 ** 10") == "1,024"
    # Whole-number float: formatted as int
    assert calculate("sqrt(144)") == "12"
