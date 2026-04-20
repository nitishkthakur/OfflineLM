"""DeepAgent backend — `langchain-ai/deepagents` wired to local Ollama only.

Every LLM call site inside deepagents is explicitly pinned to a ChatOllama
instance. The upstream defaults (`ChatAnthropic(claude-sonnet-4-5)`) and the
OpenRouter/OpenAI code paths are NEVER reached — there is no code path in this
backend that can leak a request to a non-Ollama endpoint.

Four LLM slots are controlled:
  1. Main agent model  — the `model_id` selected in the sidebar dropdown.
  2. Summarization model (`SummarizationMiddleware`) — `config.deep_agent.summarization_model`,
     falls back to the main model.
  3. Subagent default model — `config.deep_agent.subagent_default_model`,
     falls back to the main model.
  4. Per-subagent model — `subagents[i].model`, falls back to subagent default.

Trace surfacing:
  - Every `on_chat_model_*` event is streamed (main, subagents, summarization).
  - Every `on_tool_*` event is streamed, including `write_todos` (planning),
    `task` (subagent dispatch), and filesystem tools (`ls`, `read_file`, …).
  - Subagent context is inferred from LangGraph `metadata.langgraph_checkpoint_ns`
    so nested events are labelled with the subagent name in the trace.
"""
from __future__ import annotations

import os
from typing import Any, AsyncGenerator

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

from .base import AgentBackend
from . import register
from tools import SYSTEM_PROMPT


# ── Capability check (shared shape with react_agent) ──────────────────────────

async def _get_model_capabilities(model_id: str, ollama_url: str) -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{ollama_url}/api/show", json={"name": model_id})
            resp.raise_for_status()
            return set(resp.json().get("capabilities", []))
    except Exception:
        return set()


def _build_chat_ollama(
    model_id: str,
    ollama_url: str,
    *,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    reasoning: bool | None = None,
) -> ChatOllama:
    """Build a ChatOllama with only the opts the caller set (mirrors react_agent)."""
    opts: dict[str, Any] = dict(model=model_id, base_url=ollama_url)
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if temperature is not None:
        opts["temperature"] = temperature
    if reasoning is not None:
        opts["reasoning"] = reasoning
    return ChatOllama(**opts)


def _strip_images(messages: list[dict]) -> list[dict]:
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


def _subagent_name_from_metadata(metadata: dict | None) -> str | None:
    """Infer the subagent name from LangGraph stream-event metadata.

    deepagents launches each subagent via the `task` tool, which runs the
    subagent as a nested Runnable. LangGraph tags the nested graph with the
    subagent runnable's name, surfaced via `metadata.langgraph_node`,
    `langgraph_checkpoint_ns`, or `tags`. We look across these to find the
    subagent label.
    """
    if not metadata:
        return None
    # Direct node name (rare — deepagents subagents don't show here cleanly).
    node = metadata.get("langgraph_node")
    # Namespace trails the subagent invocation: "... :task:<uuid>". Also check
    # tags which sometimes carry the subagent name.
    ns = metadata.get("langgraph_checkpoint_ns", "") or ""
    tags = metadata.get("tags") or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("subagent:"):
            return tag.split(":", 1)[1]
    if ":task:" in ns or "/task:" in ns:
        return "subagent"
    if node and node not in ("agent", "tools", "model", "__start__", "__end__"):
        return node
    return None


# ── Agent backend ─────────────────────────────────────────────────────────────

@register("deep_agent")
class DeepAgentBackend(AgentBackend):
    id = "deep_agent"
    name = "Deep Agent"
    description = (
        "Hierarchical agent (LangChain deepagents) with planning, filesystem, "
        "subagents, and history summarization — all via local Ollama. "
        "Configure subagents and summarization model in config.json → deep_agent."
    )
    config_schema = {
        "recursion_limit": {
            "type": "integer",
            "label": "Recursion limit",
            "description": "Maximum agent reasoning steps per turn.",
            "default": 250,
            "min": 10,
            "max": 1000,
        },
    }

    async def stream(
        self,
        messages: list[dict],
        model_id: str,
        tools: list,
        backend_config: dict,
        conversation_id: str,
    ) -> AsyncGenerator[dict, None]:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        opts = backend_config.get("ollama_options", {}) or {}
        num_predict = opts.get("num_predict")
        num_ctx = opts.get("num_ctx")
        temperature = opts.get("temperature")
        reasoning_req = opts.get("reasoning")

        caps = await _get_model_capabilities(model_id, ollama_url)
        model_supports_tools = "tools" in caps
        model_supports_thinking = "thinking" in caps
        model_supports_vision = "vision" in caps

        if not model_supports_vision:
            messages = _strip_images(messages)

        if reasoning_req is True and not model_supports_thinking:
            reasoning_req = False

        if not model_supports_tools:
            yield self.ev_error(
                f"DeepAgent requires a tool-capable model. '{model_id}' capabilities="
                f"{sorted(caps) or 'unknown'}. Pick a model with the 'tools' capability "
                f"(e.g. qwen3:4b, qwen2.5:14b, gpt-oss:latest)."
            )
            return

        # ── Build main model ──────────────────────────────────────────────
        main_model = _build_chat_ollama(
            model_id, ollama_url,
            num_ctx=num_ctx, num_predict=num_predict,
            temperature=temperature, reasoning=reasoning_req,
        )

        # ── Summarization model ───────────────────────────────────────────
        summ_model_id = backend_config.get("summarization_model") or model_id
        summ_model = _build_chat_ollama(
            summ_model_id, ollama_url,
            num_ctx=num_ctx, temperature=0.0,
        )

        # ── Subagents ─────────────────────────────────────────────────────
        sub_default_id = backend_config.get("subagent_default_model") or model_id
        sub_default_model = _build_chat_ollama(
            sub_default_id, ollama_url,
            num_ctx=num_ctx, num_predict=num_predict,
            temperature=temperature,
        )

        raw_subagents = backend_config.get("subagents", []) or []
        subagents: list[dict] = []
        resolved_subagent_names: list[str] = []
        for sa in raw_subagents:
            if not sa.get("include", True):
                continue
            name = sa.get("name")
            if not name:
                continue
            sa_model_id = sa.get("model") or sub_default_id
            sa_model = (
                sub_default_model
                if sa_model_id == sub_default_id
                else _build_chat_ollama(
                    sa_model_id, ollama_url,
                    num_ctx=num_ctx, num_predict=num_predict,
                    temperature=temperature,
                )
            )
            subagents.append({
                "name": name,
                "description": sa.get("description", ""),
                "system_prompt": sa.get("prompt") or sa.get("system_prompt") or "",
                "tools": tools,
                "model": sa_model,
            })
            resolved_subagent_names.append(f"{name}→{sa_model_id}")

        # ── Wrap tools for LangChain ─────────────────────────────────────
        lc_tools = []
        for fn in tools:
            try:
                lc_tools.append(StructuredTool.from_function(fn))
            except Exception as e:
                yield self.ev_progress(0, f"Warning: could not wrap tool {getattr(fn,'__name__',fn)}: {e}")

        # ── Middleware stack (explicit, no upstream defaults) ─────────────
        backend_factory = lambda rt: StateBackend(rt)  # noqa: E731
        subagent_default_middleware = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=backend_factory),
            SummarizationMiddleware(
                model=summ_model,
                trigger=("tokens", 120_000),
                keep=("messages", 6),
                trim_tokens_to_summarize=None,
            ),
            PatchToolCallsMiddleware(),
        ]
        main_middleware = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=backend_factory),
            SubAgentMiddleware(
                default_model=sub_default_model,
                default_tools=lc_tools,
                subagents=subagents,
                default_middleware=subagent_default_middleware,
                general_purpose_agent=bool(backend_config.get("general_purpose_agent", True)),
            ),
            SummarizationMiddleware(
                model=summ_model,
                trigger=("tokens", 120_000),
                keep=("messages", 6),
                trim_tokens_to_summarize=None,
            ),
            PatchToolCallsMiddleware(),
        ]

        agent = create_agent(
            main_model,
            system_prompt=SYSTEM_PROMPT,
            tools=lc_tools,
            middleware=main_middleware,
        )

        # Surface resolved config in the trace so the user can verify no
        # silent fallbacks happened.
        yield self.ev_progress(0, (
            f"DeepAgent → main={model_id} · summarization={summ_model_id} · "
            f"subagent_default={sub_default_id} · subagents=[{', '.join(resolved_subagent_names) or 'none'}]"
        ))

        recursion_limit = int(backend_config.get("recursion_limit", 250))
        step_count = 0
        llm_call_count = 0
        full_response = ""

        try:
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": recursion_limit},
            ):
                kind = event.get("event")
                metadata = event.get("metadata") or {}
                sub_name = _subagent_name_from_metadata(metadata)
                prefix = f"[{sub_name}] " if sub_name else ""

                if kind == "on_chat_model_start":
                    llm_call_count += 1
                    label = event.get("name") or "LLM"
                    yield self.ev_agent_turn(llm_call_count)
                    yield self.ev_progress(0, f"{prefix}{label} call #{llm_call_count} starting…")

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    rc = ""
                    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                        rc = chunk.additional_kwargs.get("reasoning_content", "") or ""
                    if rc:
                        yield self.ev_think(f"{prefix}{rc}" if prefix else rc)
                        continue

                    content = getattr(chunk, "content", "") or ""
                    if isinstance(content, str) and content:
                        # Subagent / summarization output stays in the trace
                        # (as think bubbles) so the final answer shown to the
                        # user comes only from the top-level agent.
                        if sub_name:
                            yield self.ev_think(f"[{sub_name}] {content}")
                        else:
                            full_response += content
                            yield self.ev_content(content)
                    elif isinstance(content, list):
                        for block in content:
                            text = ""
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "") or ""
                            elif hasattr(block, "text"):
                                text = block.text or ""
                            if not text:
                                continue
                            if sub_name:
                                yield self.ev_think(f"[{sub_name}] {text}")
                            else:
                                full_response += text
                                yield self.ev_content(text)

                elif kind == "on_tool_start":
                    step_count += 1
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    desc = _tool_description(tool_name, tool_input)
                    labelled_name = f"{prefix}{tool_name}" if prefix else tool_name
                    yield self.ev_progress(step_count, f"Step {step_count}: {prefix}{desc}")
                    yield self.ev_tool_start(labelled_name, desc, tool_input)

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    raw = event.get("data", {}).get("output", "")
                    result = str(getattr(raw, "content", raw) or "")
                    result = result[:5000]
                    labelled_name = f"{prefix}{tool_name}" if prefix else tool_name
                    yield self.ev_tool_end(labelled_name)
                    yield self.ev_tool_result(labelled_name, result)

        except Exception as e:
            yield self.ev_error(f"DeepAgent error: {e}")
            return

        yield self.ev_done(conversation_id, step_count)


def _tool_description(tool_name: str, tool_input) -> str:
    if isinstance(tool_input, str):
        s = tool_input[:80]
        return f"{tool_name}: {s}{'…' if len(tool_input) > 80 else ''}"
    if isinstance(tool_input, dict):
        if tool_name == "task":
            sub = tool_input.get("subagent_type", "")
            d = tool_input.get("description", "")
            return f"task → {sub}: {d[:80]}{'…' if len(d) > 80 else ''}"
        if tool_name == "write_todos":
            todos = tool_input.get("todos", [])
            return f"write_todos ({len(todos)} items)"
        if tool_name == "internet_search":
            q = tool_input.get("query", "")
            return f"Searching: {q[:60]}{'…' if len(q) > 60 else ''}"
        if tool_name in ("save_artifact", "read_artifact"):
            return f"{tool_name}: {tool_input.get('filename', '')}"
        if tool_name == "list_artifacts":
            return "Listing artifacts"
        if tool_name in ("read_file", "write_file", "edit_file", "ls", "glob", "grep"):
            # deepagents filesystem tools
            target = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("pattern") or ""
            return f"{tool_name}: {str(target)[:80]}"
    return f"{tool_name}"
