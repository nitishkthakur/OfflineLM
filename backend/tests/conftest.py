"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure the backend directory is on the path so imports resolve.
BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest


# ── Free OpenRouter models used for any integration tests ─────────────────────
FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-nano-9b-v2:free",
]


@pytest.fixture
def free_models():
    return FREE_MODELS


@pytest.fixture
def council_config():
    """Default council config using only free models."""
    return {
        "council_models": FREE_MODELS,
        "chairman_model": FREE_MODELS[0],
    }
