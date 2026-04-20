"""ReAct Agent backend — fully local LangChain ReAct agent via Ollama.

Supports Ollama generation hyperparameters forwarded from the frontend:
  ollama_options.num_predict  – max output tokens  (-1 = unlimited)
  ollama_options.num_ctx      – context window size
  ollama_options.temperature  – sampling temperature
  ollama_options.reasoning    – True / False / None  (thinking mode)

Safety guards (both checked via a single /api/show call per request):
  • If reasoning=True is requested but the model lacks 'thinking' capability,
    reasoning is silently forced to False to avoid Ollama 400.
  • If the model lacks 'tools' capability, lc_tools is silently emptied so
    that create_agent doesn't get a 400 "does not support tools" error.
"""
import os
from typing import AsyncGenerator

import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool

from .base import AgentBackend
from . import register
from tools import SYSTEM_PROMPT


# ── Streaming <think> parser ──────────────────────────────────────────────────

class _ThinkParser:
    """Split a streamed token sequence into 'content' and 'think' segments.

    Used when reasoning=None (model default) to catch raw <think>…</think>
    tags in the main content stream.  When reasoning=True, ChatOllama already
    routes thinking tokens into chunk.additional_kwargs['reasoning_content'],
    so this parser sees clean text and is effectively a no-op.
    """

    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._state = "normal"
        self._buf   = ""

    def feed(self, chunk: str):
        self._buf += chunk
        while True:
            if self._state == "normal":
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    safe = max(0, len(self._buf) - len(self._OPEN))
                    if safe:
                        yield "content", self._buf[:safe]
                        self._buf = self._buf[safe:]
                    break
                if idx:
                    yield "content", self._buf[:idx]
                self._buf  = self._buf[idx + len(self._OPEN):]
                self._state = "think"
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    safe = max(0, len(self._buf) - len(self._CLOSE))
                    if safe:
                        yield "think", self._buf[:safe]
                        self._buf = self._buf[safe:]
                    break
                if idx:
                    yield "think", self._buf[:idx]
                self._buf  = self._buf[idx + len(self._CLOSE):]
                self._state = "normal"

    def flush(self):
        if self._buf:
            kind = "think" if self._state == "think" else "content"
            yield kind, self._buf
            self._buf   = ""
            self._state = "normal"


# ── Capability check ──────────────────────────────────────────────────────────

async def _get_model_capabilities(model_id: str, ollama_url: str) -> set[str]:
    """Return the set of capability strings from Ollama /api/show.

    Returns an empty set on any error so callers get safe defaults:
      - thinking not in caps  → don't enable reasoning
      - tools    not in caps  → don't bind tools
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/show",
                json={"name": model_id},
            )
            resp.raise_for_status()
            return set(resp.json().get("capabilities", []))
    except Exception:
        return set()


# ── Agent backend ─────────────────────────────────────────────────────────────

@register("react_agent")
class ReactAgentBackend(AgentBackend):
    id = "react_agent"
    name = "ReAct Agent"
    description = (
        "Local LangChain ReAct agent via Ollama. "
        "Reasons step-by-step, calls tools as needed, "
        "fully offline — no data leaves your machine."
    )
    config_schema = {}

    async def stream(
        self,
        messages: list[dict],
        model_id: str,
        tools: list,
        backend_config: dict,
        conversation_id: str,
    ) -> AsyncGenerator[dict, None]:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # ── Resolve Ollama generation options ─────────────────────────────
        opts: dict = backend_config.get("ollama_options", {})
        num_predict: int | None   = opts.get("num_predict")   # None → Ollama default
        num_ctx:     int | None   = opts.get("num_ctx")
        temperature: float | None = opts.get("temperature")
        reasoning_req             = opts.get("reasoning")      # True/False/None

        # Single /api/show call — fetch all capabilities once.
        caps = await _get_model_capabilities(model_id, ollama_url)
        model_supports_tools    = "tools"    in caps
        model_supports_thinking = "thinking" in caps
        model_supports_vision   = "vision"   in caps

        # Safety: strip image_url blocks if the model has no vision capability.
        # The frontend gates this too, but we enforce it server-side as well so
        # that a non-vision model never receives image content and errors out.
        if not model_supports_vision:
            messages = _strip_images(messages)

        # Safety: only allow reasoning=True if the model actually supports it.
        # Sending reasoning=True to an unsupported model causes a 400 from Ollama.
        if reasoning_req is True and not model_supports_thinking:
            reasoning_req = False   # silently downgrade

        # Wrap plain functions as LangChain StructuredTool instances.
        # If the model has no 'tools' capability, pass an empty list —
        # create_agent(tools=[]) still works (plain chat agent) and avoids
        # the Ollama 400 "does not support tools" error.
        lc_tools = []
        if model_supports_tools:
            for fn in tools:
                try:
                    lc_tools.append(StructuredTool.from_function(fn))
                except Exception as e:
                    fn_name = getattr(fn, "__name__", str(fn))
                    yield self.ev_progress(0, f"Warning: could not wrap tool {fn_name}: {e}")
        else:
            # Make this loud: if the user asked a question that needs tools
            # (e.g. arithmetic on a bank statement, web search, artifact save),
            # a silent fallback to plain chat will produce wrong answers.
            warning = (
                f"⚠ Model '{model_id}' has no tool-calling capability in Ollama "
                f"(capabilities={sorted(caps) or 'unknown'}). "
                f"calculate / pct_change / internet_search / save_artifact WILL NOT run — "
                f"the model may emit tool-markup text (e.g. ```tool_code```) that is never executed. "
                f"For arithmetic accuracy switch to a tool-capable model "
                f"(e.g. qwen3:4b, qwen2.5:14b, gpt-oss:latest)."
            )
            yield self.ev_progress(0, warning)
            yield self.ev_think(warning)

        # Build ChatOllama — pass only the params that were explicitly set
        chatopts: dict = dict(model=model_id, base_url=ollama_url)
        if num_predict is not None:
            chatopts["num_predict"] = num_predict
        if num_ctx is not None:
            chatopts["num_ctx"] = num_ctx
        if temperature is not None:
            chatopts["temperature"] = temperature
        if reasoning_req is not None:
            chatopts["reasoning"] = reasoning_req

        model = ChatOllama(**chatopts)

        agent = create_agent(
            model=model,
            tools=lc_tools,
            system_prompt=SYSTEM_PROMPT,
        )

        step_count     = 0
        full_response  = ""
        llm_call_count = 0
        parser         = _ThinkParser()

        try:
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": backend_config.get("recursion_limit", 50)},
            ):
                kind = event.get("event")

                # ── New LLM call ───────────────────────────────────────────
                if kind == "on_chat_model_start":
                    llm_call_count += 1
                    for t, text in parser.flush():
                        if t == "think" and text:
                            yield self.ev_think(text)
                        elif text:
                            full_response += text
                            yield self.ev_content(text)
                    yield self.ev_agent_turn(llm_call_count)
                    yield self.ev_progress(
                        0,
                        "Generating response — if the model is large or CPU-offloaded, "
                        "the first token may take 30–60 s…",
                    )

                # ── Streaming tokens ───────────────────────────────────────
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    # Path 1: ChatOllama with reasoning=True separates thinking
                    # into additional_kwargs['reasoning_content']
                    rc = ""
                    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                        rc = chunk.additional_kwargs.get("reasoning_content", "") or ""
                    if rc:
                        yield self.ev_think(rc)
                        continue   # thinking chunk has no main content

                    # Path 2: raw content (may have <think> tags if reasoning=None)
                    content = getattr(chunk, "content", "") or ""
                    raw_texts: list[str] = []

                    if isinstance(content, str):
                        if content:
                            raw_texts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                t = block.get("text", "")
                                if t:
                                    raw_texts.append(t)
                            elif hasattr(block, "text") and block.text:
                                raw_texts.append(block.text)

                    for raw in raw_texts:
                        for t, text in parser.feed(raw):
                            if t == "think" and text:
                                yield self.ev_think(text)
                            elif text:
                                full_response += text
                                yield self.ev_content(text)

                # ── Tool invocation starts ─────────────────────────────────
                elif kind == "on_tool_start":
                    for t, text in parser.flush():
                        if t == "think" and text:
                            yield self.ev_think(text)
                        elif text:
                            full_response += text
                            yield self.ev_content(text)

                    step_count += 1
                    tool_name  = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    desc       = _tool_description(tool_name, tool_input)
                    yield self.ev_progress(step_count, f"Step {step_count}: {desc}")
                    yield self.ev_tool_start(tool_name, desc, tool_input)

                # ── Tool returns ───────────────────────────────────────────
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    raw_out   = event.get("data", {}).get("output", "")
                    if hasattr(raw_out, "content"):
                        result_str = str(raw_out.content)
                    else:
                        result_str = str(raw_out) if raw_out else ""
                    result_str = result_str[:5000]

                    yield self.ev_tool_end(tool_name)
                    yield self.ev_tool_result(tool_name, result_str)

        except Exception as e:
            yield self.ev_error(f"Agent error: {e}")
            return

        for t, text in parser.flush():
            if t == "think" and text:
                yield self.ev_think(text)
            elif text:
                full_response += text
                yield self.ev_content(text)

        yield self.ev_done(conversation_id, step_count)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_images(messages: list[dict]) -> list[dict]:
    """Remove image_url content blocks from all messages (for non-vision models)."""
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            cleaned.append({"role": msg["role"], "content": "\n".join(text_parts)})
        else:
            cleaned.append(msg)
    return cleaned


def _tool_description(tool_name: str, tool_input) -> str:
    if isinstance(tool_input, str):
        s = tool_input[:60]
        return f"Using {tool_name}: {s}{'...' if len(tool_input) > 60 else ''}"
    if isinstance(tool_input, dict):
        if tool_name == "internet_search":
            q = tool_input.get("query", "")
            return f"Searching: {q[:50]}{'...' if len(q) > 50 else ''}"
        if tool_name == "save_artifact":
            return f"Saving artifact: {tool_input.get('filename', '')}"
        if tool_name == "read_artifact":
            return f"Reading: {tool_input.get('filename', '')}"
        if tool_name == "list_artifacts":
            return "Listing artifacts"
    return f"Using {tool_name}"
