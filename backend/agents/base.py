"""Abstract base class for all agent backends."""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class AgentBackend(ABC):
    """
    Base class for pluggable agent backends.

    Each backend must implement `stream()`, an async generator that yields
    SSE-compatible dict events as it processes the request.

    Event shapes:
        {"type": "content",    "content": str}
        {"type": "progress",   "step": int, "description": str}
        {"type": "tool_start", "tool": str, "description": str}
        {"type": "tool_end",   "tool": str}
        {"type": "chain_start","name": str}
        {"type": "chain_end",  "name": str}
        {"type": "done",       "conversation_id": str, "total_steps": int}
        {"type": "error",      "error": str}
    """

    # Metadata that every subclass should override.
    id: str = ""
    name: str = ""
    description: str = ""

    # JSON-serialisable schema describing per-request config knobs.
    # Format: {field_name: {type, label, description, default, ...}}
    # Supported types: "integer", "string", "model", "model_list"
    config_schema: dict[str, Any] = {}

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        model_id: str,
        tools: list,
        backend_config: dict,
        conversation_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Yield SSE event dicts until the request is complete.

        Args:
            messages:         Full conversation history (role/content dicts).
            model_id:         OpenRouter model ID selected by the user.
            tools:            Ready-to-use LangChain tool functions.
            backend_config:   Runtime config values from the frontend panel.
            conversation_id:  Stable identifier for the conversation.
        """
        # This is an abstract async generator – subclasses must `yield`.
        raise NotImplementedError
        yield  # makes Python treat this as an async generator

    # ── convenience helpers ────────────────────────────────────────────────

    @staticmethod
    def ev_content(text: str) -> dict:
        return {"type": "content", "content": text}

    @staticmethod
    def ev_progress(step: int, description: str) -> dict:
        return {"type": "progress", "step": step, "description": description}

    @staticmethod
    def ev_agent_turn(turn: int) -> dict:
        return {"type": "agent_turn", "turn": turn}

    @staticmethod
    def ev_think(content: str) -> dict:
        return {"type": "think", "content": content}

    @staticmethod
    def ev_tool_start(tool: str, description: str = "", args: dict = None) -> dict:
        return {"type": "tool_start", "tool": tool, "description": description, "args": args or {}}

    @staticmethod
    def ev_tool_end(tool: str) -> dict:
        return {"type": "tool_end", "tool": tool}

    @staticmethod
    def ev_tool_result(tool: str, result: str) -> dict:
        return {"type": "tool_result", "tool": tool, "result": result}

    @staticmethod
    def ev_done(conversation_id: str, total_steps: int = 0) -> dict:
        return {"type": "done", "conversation_id": conversation_id, "total_steps": total_steps}

    @staticmethod
    def ev_error(message: str) -> dict:
        return {"type": "error", "error": message}
