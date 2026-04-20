"""Agent backend registry.

Usage:
    from agents import get_backend, list_backends, register

    @register("my_backend")
    class MyBackend(AgentBackend):
        ...

    backend = get_backend("my_backend")
"""
from typing import Type

from .base import AgentBackend

REGISTRY: dict[str, Type[AgentBackend]] = {}


def register(backend_id: str):
    """Class decorator that adds a backend to the registry."""
    def decorator(cls: Type[AgentBackend]):
        cls.id = backend_id
        REGISTRY[backend_id] = cls
        return cls
    return decorator


def get_backend(backend_id: str) -> AgentBackend:
    """Return an instantiated backend for the given id, or raise KeyError."""
    cls = REGISTRY.get(backend_id)
    if cls is None:
        raise KeyError(f"Unknown backend: {backend_id!r}. Available: {list(REGISTRY)}")
    return cls()


def list_backends() -> list[dict]:
    """Return a list of backend metadata dicts suitable for the /backends endpoint."""
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "config_schema": cls.config_schema,
        }
        for cls in REGISTRY.values()
    ]


# Auto-import all backend modules so their @register decorators run.
from . import react_agent  # noqa: E402, F401
from . import deep_agent   # noqa: E402, F401
