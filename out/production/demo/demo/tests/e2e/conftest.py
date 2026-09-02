"""
Canary smoke-test fixtures.

The agent loop in `quant_agent.stream_quant_agent` calls
`client.chat.completions.create(..., stream=True)` and iterates over OpenAI-shaped
streaming chunks. To make the loop testable without any network we patch that
method to yield a scripted sequence of chunks.

We also patch `_generate_followups` to return [] — it makes a *second* LLM call
after `finish_reason="stop"`, which we don't want to script in the canary.
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

# Make project root importable when pytest is invoked from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── OpenAI streaming chunk builders ────────────────────────────────────────────
# Match the attribute shape that quant_agent.stream_quant_agent reads from each
# chunk: chunk.choices[0].delta.{content, tool_calls} and choices[0].finish_reason.

def content_chunk(text: str) -> NS:
    """A chunk that streams plain assistant text (no finish)."""
    return NS(choices=[NS(
        delta=NS(content=text, tool_calls=None),
        finish_reason=None,
    )])


def stop_chunk() -> NS:
    """The final chunk that ends a streaming response with finish_reason='stop'."""
    return NS(choices=[NS(
        delta=NS(content=None, tool_calls=None),
        finish_reason="stop",
    )])


def tool_call_chunk(call_id: str, name: str, arguments_json: str) -> NS:
    """A chunk containing one complete tool_call delta (id + name + full args)."""
    tc = NS(
        index=0,
        id=call_id,
        function=NS(name=name, arguments=arguments_json),
    )
    return NS(choices=[NS(
        delta=NS(content=None, tool_calls=[tc]),
        finish_reason=None,
    )])


def tool_calls_finish_chunk() -> NS:
    """Final chunk that signals the assistant turn ended with tool_calls."""
    return NS(choices=[NS(
        delta=NS(content=None, tool_calls=None),
        finish_reason="tool_calls",
    )])


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def scripted_llm(monkeypatch):
    """
    Patch `quant_agent.client.chat.completions.create` to return scripted chunks.

    Yields a callable: `script(*responses)` where each response is a list of
    chunks. Each LLM call (each iteration of the agent loop) consumes the next
    response list in order.
    """
    import quant_agent

    # Suppress the followup LLM call — out of scope for the canary
    monkeypatch.setattr(quant_agent, "_generate_followups", lambda messages: [])

    scripted_responses: list[list] = []
    call_count = {"n": 0}

    def fake_create(*args, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        if i >= len(scripted_responses):
            raise AssertionError(
                f"LLM was called {i + 1} times but only "
                f"{len(scripted_responses)} responses were scripted"
            )
        return iter(scripted_responses[i])

    monkeypatch.setattr(
        quant_agent.client.chat.completions, "create", fake_create
    )

    def script(*responses):
        scripted_responses.extend(responses)
        return call_count

    return script
