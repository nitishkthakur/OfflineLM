"""Unit tests for the Council backend — fully mocked, no real API calls."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.council import CouncilBackend, _extract_last_user_message


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_completion(text: str):
    """Build a minimal fake chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_stream_chunk(text: str | None):
    """Build a fake streaming chunk."""
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _async_iter(items):
    for item in items:
        yield item


# ── tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_last_user_message():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert _extract_last_user_message(messages) == "What is the capital of France?"


@pytest.mark.asyncio
async def test_extract_last_user_message_content_block():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "block question"}]},
    ]
    assert _extract_last_user_message(messages) == "block question"


@pytest.mark.asyncio
async def test_council_no_api_key(monkeypatch):
    """Council yields error event when OPENROUTER_API_KEY is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    backend = CouncilBackend()
    events = []
    async for ev in backend.stream(
        messages=[{"role": "user", "content": "hi"}],
        model_id="any",
        tools=[],
        backend_config={},
        conversation_id="c1",
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "OPENROUTER_API_KEY" in events[0]["error"]


@pytest.mark.asyncio
async def test_council_full_flow_mocked(monkeypatch):
    """
    Test the 3-phase council flow with mocked OpenAI client.
    Uses free models as specified: nvidia/nemotron-3-super-120b-a12b:free
    and nvidia/nemotron-nano-9b-v2:free.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Phase 1 completions — one per council model
    phase1_resp_a = _make_completion("Paris is the capital of France.")
    phase1_resp_b = _make_completion("The capital is Paris.")

    # Phase 2 review completions — one per council model
    review_a = _make_completion("Response A is concise. Response B is also correct.")
    review_b = _make_completion("Both are accurate; A is slightly clearer.")

    # Phase 3 streaming chunks
    stream_chunks = [
        _make_stream_chunk("Paris"),
        _make_stream_chunk(" is"),
        _make_stream_chunk(" the capital of France."),
        _make_stream_chunk(None),  # end chunk with no content
    ]

    mock_create = AsyncMock(
        side_effect=[
            # Phase 1: two parallel non-streaming calls
            phase1_resp_a,
            phase1_resp_b,
            # Phase 2: two parallel non-streaming calls
            review_a,
            review_b,
            # Phase 3: one streaming call
            _async_iter(stream_chunks),
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("agents.council.AsyncOpenAI", return_value=mock_client):
        backend = CouncilBackend()
        events = []
        async for ev in backend.stream(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model_id="nvidia/nemotron-3-super-120b-a12b:free",
            tools=[],
            backend_config={
                "council_models": [
                    "nvidia/nemotron-3-super-120b-a12b:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                ],
                "chairman_model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
            conversation_id="test-conv",
        ):
            events.append(ev)

    event_types = [e["type"] for e in events]

    # Should have 3 progress events, content events, and a done event
    assert event_types.count("progress") == 3
    assert "content" in event_types
    assert event_types[-1] == "done"
    assert events[-1]["conversation_id"] == "test-conv"
    assert events[-1]["total_steps"] == 3

    # Content should match streamed chunks
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "Paris" in content


@pytest.mark.asyncio
async def test_council_phase1_all_fail_yields_error(monkeypatch):
    """If ALL Phase 1 models fail, an error event is yielded."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mock_create = AsyncMock(side_effect=Exception("connection refused"))
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("agents.council.AsyncOpenAI", return_value=mock_client):
        backend = CouncilBackend()
        events = []
        async for ev in backend.stream(
            messages=[{"role": "user", "content": "hi"}],
            model_id="any",
            tools=[],
            backend_config={
                "council_models": [
                    "nvidia/nemotron-3-super-120b-a12b:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                ],
                "chairman_model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
            conversation_id="c2",
        ):
            events.append(ev)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1
    assert "Phase 1 failed" in error_events[0]["error"]


@pytest.mark.asyncio
async def test_council_phase1_partial_failure_continues(monkeypatch):
    """If only some Phase 1 models fail, the council continues with successful ones."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # First model fails, second succeeds
    good_resp = _make_completion("Paris is the capital of France.")
    review = _make_completion("Good answer.")
    stream_chunks = [_make_stream_chunk("Paris is the capital.")]

    mock_create = AsyncMock(
        side_effect=[Exception("model unavailable"), good_resp, review, _async_iter(stream_chunks)]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("agents.council.AsyncOpenAI", return_value=mock_client):
        backend = CouncilBackend()
        events = []
        async for ev in backend.stream(
            messages=[{"role": "user", "content": "Capital of France?"}],
            model_id="any",
            tools=[],
            backend_config={
                "council_models": [
                    "nvidia/nemotron-3-super-120b-a12b:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                ],
                "chairman_model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
            conversation_id="c5",
        ):
            events.append(ev)

    # Should still produce content and done (not a hard error)
    types = [e["type"] for e in events]
    assert "content" in types
    assert "done" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_council_comma_separated_models(monkeypatch):
    """council_models can be a comma-separated string (2 models → 2 calls per phase)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Phase 1: 2 models → 2 completions
    # Phase 2: 2 models → 2 review completions
    # Phase 3: 1 streaming call
    phase1_a = _make_completion("Paris.")
    phase1_b = _make_completion("The capital is Paris.")
    review_a = _make_completion("Good.")
    review_b = _make_completion("Also good.")
    stream_chunks = [_make_stream_chunk("Paris is the capital.")]

    mock_create = AsyncMock(
        side_effect=[phase1_a, phase1_b, review_a, review_b, _async_iter(stream_chunks)]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("agents.council.AsyncOpenAI", return_value=mock_client):
        backend = CouncilBackend()
        events = []
        async for ev in backend.stream(
            messages=[{"role": "user", "content": "Capital of France?"}],
            model_id="any",
            tools=[],
            backend_config={
                "council_models": "nvidia/nemotron-3-super-120b-a12b:free,nvidia/nemotron-nano-9b-v2:free",
                "chairman_model": "nvidia/nemotron-3-super-120b-a12b:free",
            },
            conversation_id="c3",
        ):
            events.append(ev)

    # No error events expected
    assert not any(e["type"] == "error" for e in events)
