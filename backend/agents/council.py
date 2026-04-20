"""Council of LLMs backend — inspired by Karpathy's llm-council.

Architecture (3 phases):
  Phase 1 — Parallel inference: every council model answers independently.
  Phase 2 — Anonymous peer review: each council model reviews all others'
             answers (labelled Response A / B / C …) without knowing which
             model wrote which.
  Phase 3 — Chairman synthesis: a designated chairman model reads all answers
             and all reviews then produces the final, streamed response.
"""
import asyncio
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .base import AgentBackend
from . import register

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_REVIEW_SYSTEM = (
    "You are an expert evaluator. You will be given several responses to a "
    "user question, labelled Response A, Response B, … You must critically "
    "assess each response for accuracy, completeness, and clarity. "
    "Be concise and constructive."
)

_CHAIRMAN_SYSTEM = (
    "You are the chairman of a council of AI models. You have received "
    "multiple responses to a user question and peer reviews of those "
    "responses. Synthesise the best answer, incorporating the strongest "
    "points and avoiding the weaknesses identified in the reviews. "
    "Respond directly to the user — do not mention the council process."
)


@register("council")
class CouncilBackend(AgentBackend):
    id = "council"
    name = "Council of LLMs"
    description = (
        "Multiple models answer in parallel, peer-review each other, then a "
        "chairman synthesises the final response (Karpathy's LLM Council)."
    )

    config_schema = {
        "council_models": {
            "type": "model_list",
            "label": "Council Models",
            "description": "Models that answer and review (comma-separated OpenRouter IDs).",
            "default": [
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-nano-9b-v2:free",
            ],
        },
        "chairman_model": {
            "type": "model",
            "label": "Chairman Model",
            "description": "Model that synthesises the final answer.",
            "default": "nvidia/nemotron-3-super-120b-a12b:free",
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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            yield self.ev_error("OPENROUTER_API_KEY is not set")
            return

        client = AsyncOpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)

        # Resolve config values
        raw_council = backend_config.get(
            "council_models",
            self.config_schema["council_models"]["default"],
        )
        if isinstance(raw_council, str):
            council_models = [m.strip() for m in raw_council.split(",") if m.strip()]
        else:
            council_models = list(raw_council)

        chairman_model = backend_config.get(
            "chairman_model",
            self.config_schema["chairman_model"]["default"],
        )

        # ── Phase 1: Parallel inference ───────────────────────────────────
        yield self.ev_progress(1, f"Phase 1 — {len(council_models)} models answering in parallel…")

        async def _infer(model: str) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            choices = resp.choices if resp else None
            if not choices:
                raise ValueError(f"No choices returned from {model}")
            return choices[0].message.content or ""

        tasks = [_infer(m) for m in council_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Keep successful answers; skip failed models
        paired = list(zip(council_models, results))
        successful = [(m, r) for m, r in paired if not isinstance(r, Exception)]
        failed = [(m, r) for m, r in paired if isinstance(r, Exception)]
        for m, exc in failed:
            yield self.ev_progress(1, f"⚠ {m} skipped: {exc}")

        if not successful:
            yield self.ev_error(f"Phase 1 failed: all council models returned errors. Last: {results[-1]}")
            return

        labels = [chr(ord("A") + i) for i in range(len(successful))]
        answer_block = "\n\n".join(
            f"Response {lbl}:\n{ans}" for lbl, (_, ans) in zip(labels, successful)
        )

        # ── Phase 2: Anonymous peer review ────────────────────────────────
        yield self.ev_progress(2, "Phase 2 — models reviewing each other's responses…")

        original_question = _extract_last_user_message(messages)

        review_messages = [
            {"role": "system", "content": _REVIEW_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original question:\n{original_question}\n\n"
                    f"Responses to review:\n{answer_block}\n\n"
                    "Please evaluate each response."
                ),
            },
        ]

        async def _review(model: str) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=review_messages,
            )
            choices = resp.choices if resp else None
            if not choices:
                raise ValueError(f"No choices returned from {model}")
            return choices[0].message.content or ""

        review_tasks = [_review(m) for m in [m for m, _ in successful]]
        review_results = await asyncio.gather(*review_tasks, return_exceptions=True)
        reviews = [r if not isinstance(r, Exception) else f"(review unavailable: {r})"
                   for r in review_results]

        review_block = "\n\n".join(
            f"Review from Model {lbl}:\n{rev}" for lbl, rev in zip(labels, reviews)
        )

        # ── Phase 3: Chairman synthesis (streamed) ────────────────────────
        yield self.ev_progress(3, f"Phase 3 — {chairman_model} synthesising final answer…")

        chairman_messages = [
            {"role": "system", "content": _CHAIRMAN_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original question:\n{original_question}\n\n"
                    f"Council responses:\n{answer_block}\n\n"
                    f"Peer reviews:\n{review_block}\n\n"
                    "Please provide the best synthesised answer."
                ),
            },
        ]

        try:
            stream = await client.chat.completions.create(
                model=chairman_model,
                messages=chairman_messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield self.ev_content(delta)
        except Exception as e:
            yield self.ev_error(f"Phase 3 failed: {e}")
            return

        yield self.ev_done(conversation_id, total_steps=3)


def _extract_last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
    return ""
