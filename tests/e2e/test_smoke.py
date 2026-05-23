"""
Canary smoke test for the quant agent loop.

This is the safety net for refactoring `quant_agent.py` (G2 in CLAUDE.md).
If these two tests stay green through a refactor, the core machinery —
system prompt building, message handling, streaming accumulation,
tool dispatch via TOOL_REGISTRY, and the generator event contract — still works.

The tests are hermetic: the LLM is patched to return scripted chunks, and the
only tool exercised is `black_scholes`, which is pure math (no network).
"""

import json

from .conftest import (
    content_chunk,
    stop_chunk,
    tool_call_chunk,
    tool_calls_finish_chunk,
)


def _drain(generator) -> list:
    """Collect all events from the agent generator."""
    return list(generator)


def _events_of(events: list, event_type: str) -> list:
    return [e for e in events if e.get("type") == event_type]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — simple chat, no tool dispatch.
# Protects: system prompt injection, message handling, streaming accumulator,
# finish_reason="stop" path, markdown rendering, generator event contract.
# ─────────────────────────────────────────────────────────────────────────────

def test_simple_chat_no_tools(scripted_llm):
    from quant_agent import stream_quant_agent

    scripted_llm([
        content_chunk("Hello "),
        content_chunk("world."),
        stop_chunk(),
    ])

    messages = [{"role": "user", "content": "say hello"}]
    events = _drain(stream_quant_agent(messages, max_iterations=3))

    # Generator must produce at least one content_delta and exactly one final.
    deltas = _events_of(events, "content_delta")
    finals = _events_of(events, "final")
    errors = _events_of(events, "error")

    assert not errors, f"unexpected errors: {errors}"
    assert deltas, "expected at least one content_delta event"
    assert len(finals) == 1, f"expected exactly one final event, got {len(finals)}"

    final = finals[0]
    assert final["text"] == "Hello world."
    assert final["iterations"] == 1
    assert "text_html" in final  # markdown was rendered

    # Streamed content reassembles correctly
    assert "".join(d["text"] for d in deltas) == "Hello world."

    # Conversation log has the assistant turn appended
    assert messages[-1] == {"role": "assistant", "content": "Hello world."}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — one tool-call round-trip with a pure-math tool.
# Protects: TOOL_REGISTRY dispatch, tool_call → tool_result event emission,
# assistant + tool message atomic append, multi-iteration loop continuation.
# ─────────────────────────────────────────────────────────────────────────────

def test_tool_dispatch_roundtrip(scripted_llm):
    from quant_agent import stream_quant_agent

    bs_args = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 0.25,
        "risk_free_rate": 0.03,
        "volatility": 0.20,
        "option_type": "call",
    }

    # Iteration 1: LLM asks to call black_scholes.
    # Iteration 2: LLM produces final answer.
    scripted_llm(
        [
            tool_call_chunk(
                call_id="call_abc",
                name="black_scholes",
                arguments_json=json.dumps(bs_args),
            ),
            tool_calls_finish_chunk(),
        ],
        [
            content_chunk("The call price is ~4.6."),
            stop_chunk(),
        ],
    )

    messages = [{"role": "user", "content": "price a 3-month ATM call on a $100 stock"}]
    events = _drain(stream_quant_agent(messages, max_iterations=5))

    errors = _events_of(events, "error")
    tool_calls = _events_of(events, "tool_call")
    tool_results = _events_of(events, "tool_result")
    finals = _events_of(events, "final")

    assert not errors, f"unexpected errors: {errors}"
    assert len(tool_calls) == 1, f"expected 1 tool_call, got {len(tool_calls)}"
    assert len(tool_results) == 1, f"expected 1 tool_result, got {len(tool_results)}"
    assert len(finals) == 1, f"expected 1 final, got {len(finals)}"

    # The dispatched tool was the one the LLM asked for, with the args passed through.
    assert tool_calls[0]["name"] == "black_scholes"
    assert tool_calls[0]["input"]["spot"] == 100.0

    # black_scholes returned a successful result (pure math, deterministic).
    result = tool_results[0]["result"]
    assert tool_results[0]["is_error"] is False
    assert result.get("success") is True
    assert "theoretical_price" in result
    assert 4.0 < result["theoretical_price"] < 5.5  # ATM 3M @ 20% vol, r=3% → ~$4.35

    # Final answer arrived on iteration 2.
    assert finals[0]["iterations"] == 2
    assert finals[0]["text"] == "The call price is ~4.6."

    # Message log records: user → assistant(tool_calls) → tool(result) → assistant(final)
    roles = [m["role"] for m in messages if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant"]
