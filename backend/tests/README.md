# Backend Tests

All tests are **fully mocked** — no real API calls, no credentials required.

## Running

```bash
cd devin_app/backend
source .venv/bin/activate   # or venv/bin/activate
pip install pytest pytest-asyncio   # one-time
pytest tests/ -v
```

## Test files

| File | What it tests |
|------|--------------|
| `test_registry.py` | Registry `@register` decorator, `get_backend()`, `list_backends()`, event helpers, config schemas |
| `test_council.py` | Council 3-phase flow (mocked OpenAI), error propagation, comma-separated model strings |
| `test_api.py` | FastAPI endpoints: `/models`, `/backends`, `/artifacts`, `/chat` routing |

## Free models for integration tests

If you add integration tests that hit the real API, use only these free OpenRouter models:

- `nvidia/nemotron-3-super-120b-a12b:free`
- `nvidia/nemotron-nano-9b-v2:free`

These are available via `conftest.py`'s `free_models` fixture.
