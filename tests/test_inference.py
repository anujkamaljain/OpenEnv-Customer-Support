"""Tests for inference.py — validates exact stdout format and safety guarantees.

Uses a mock OpenAI client so no real API key is needed.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import inference


@pytest.fixture(autouse=True)
def _single_difficulty_for_tests() -> Any:
    """Full baseline runs easy+medium+hard; unit tests keep one episode."""
    with patch.object(inference, "EPISODE_DIFFICULTIES", ["easy"]):
        yield


# ── mock helpers ─────────────────────────────────────────────────────────────

def _make_mock_client(responses: list[str]) -> AsyncMock:
    """Build a mock AsyncOpenAI that returns *responses* in order."""
    client = AsyncMock()
    side_effects = []
    for text in responses:
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        side_effects.append(resp)
    while len(side_effects) < 15:
        side_effects.append(side_effects[-1])
    client.chat.completions.create = AsyncMock(side_effect=side_effects)
    return client


OPTIMAL_EASY_RESPONSES = [
    '{"action_type":"classify","category":"account_access","priority":"medium"}',
    '{"action_type":"route","department":"account"}',
    '{"action_type":"resolve","resolution_summary":"Password reset link sent. Access to the account is now restored."}',
]


# ── format validation ───────────────────────────────────────────────────────

START_RE = re.compile(
    r"^\[START\] task=\S+ env=\S+ model=\S+$"
)
STEP_RE = re.compile(
    r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(?:true|false) error=.+$"
)
END_RE = re.compile(
    r"^\[END\] success=(?:true|false) steps=\d+ score=\d+\.\d{3} rewards=.+$"
)


def _parse_output(captured: str) -> dict[str, Any]:
    lines = [l for l in captured.strip().splitlines() if l.strip()]
    assert len(lines) >= 2, f"Expected at least 2 lines, got:\n{captured}"

    start_line = lines[0]
    end_line = lines[-1]
    step_lines = lines[1:-1]

    assert START_RE.match(start_line), f"Bad START line: {start_line}"
    assert END_RE.match(end_line), f"Bad END line: {end_line}"
    for sl in step_lines:
        assert STEP_RE.match(sl), f"Bad STEP line: {sl}"

    end_parts = end_line.split()
    success = end_parts[1].split("=")[1]
    steps = int(end_parts[2].split("=")[1])
    score = float(end_parts[3].split("=")[1])
    rewards_str = end_parts[4].split("=")[1]

    assert success in ("true", "false")
    assert steps == len(step_lines)

    if rewards_str:
        rewards = [float(r) for r in rewards_str.split(",")]
    else:
        rewards = []

    return {
        "success": success == "true",
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "step_lines": step_lines,
    }


# ── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_optimal_easy_episode_format(capsys: pytest.CaptureFixture[str]) -> None:
    """A full easy episode must produce valid [START]/[STEP]/[END] lines."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    parsed = _parse_output(out)

    assert parsed["success"] is True
    assert parsed["steps"] == 3
    assert parsed["score"] > 0.1
    assert len(parsed["rewards"]) == 3
    for r in parsed["rewards"]:
        assert -0.25 <= r <= 0.30


@pytest.mark.asyncio
async def test_step_numbering_starts_at_one(capsys: pytest.CaptureFixture[str]) -> None:
    """Step numbers must be 1-based."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    step_nums = []
    for line in out.strip().splitlines():
        m = re.match(r"^\[STEP\] step=(\d+)", line)
        if m:
            step_nums.append(int(m.group(1)))

    assert step_nums == [1, 2, 3]


@pytest.mark.asyncio
async def test_model_returns_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    """When the model returns non-JSON, inference uses fallback and never crashes."""
    garbage = ["lol not json"] * 12
    mock_client = _make_mock_client(garbage)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    parsed = _parse_output(out)

    assert parsed["steps"] >= 1
    for sl in parsed["step_lines"]:
        assert "error=" in sl


@pytest.mark.asyncio
async def test_model_exception(capsys: pytest.CaptureFixture[str]) -> None:
    """If the model call raises, inference still emits [END]."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("API down")
    )

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert any(l.startswith("[START]") for l in lines)
    assert any(l.startswith("[END]") for l in lines)


@pytest.mark.asyncio
async def test_reward_format_two_decimals(capsys: pytest.CaptureFixture[str]) -> None:
    """Rewards in STEP lines must have exactly 2 decimal places."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    for line in out.strip().splitlines():
        if line.startswith("[STEP]"):
            match = re.search(r"reward=(-?\d+\.\d+)", line)
            assert match is not None
            decimals = match.group(1).split(".")[1]
            assert len(decimals) == 2, f"Expected 2 decimal places, got: {match.group(1)}"


@pytest.mark.asyncio
async def test_score_format_three_decimals(capsys: pytest.CaptureFixture[str]) -> None:
    """Score in END line must have exactly 3 decimal places."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    for line in out.strip().splitlines():
        if line.startswith("[END]"):
            match = re.search(r"score=(\d+\.\d+)", line)
            assert match is not None
            decimals = match.group(1).split(".")[1]
            assert len(decimals) == 3, f"Expected 3 decimal places, got: {match.group(1)}"


@pytest.mark.asyncio
async def test_booleans_lowercase(capsys: pytest.CaptureFixture[str]) -> None:
    """All boolean fields must use lowercase true/false."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    for line in out.strip().splitlines():
        assert "True" not in line, f"Python-style True found in: {line}"
        assert "False" not in line, f"Python-style False found in: {line}"


@pytest.mark.asyncio
async def test_end_always_emitted_on_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """Even if env.reset() throws, [END] must be emitted."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with (
        patch.object(inference, "_build_client", return_value=mock_client),
        patch(
            "inference.CustomerSupportEnv.reset",
            side_effect=RuntimeError("env broken"),
        ),
    ):
        await inference.run()

    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert lines[-1].startswith("[END]")


@pytest.mark.asyncio
async def test_missing_hf_token_emits_end(capsys: pytest.CaptureFixture[str]) -> None:
    """If HF_TOKEN is empty, _build_client raises but [END] is still emitted."""
    with patch.object(inference, "HF_TOKEN", ""):
        await inference.run()

    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert lines[-1].startswith("[END]")


@pytest.mark.asyncio
async def test_action_field_is_pure_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The action= field must be unmodified JSON (no underscore replacement)."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    import json as _json

    for line in out.strip().splitlines():
        if line.startswith("[STEP]"):
            m = re.search(r"action=(.+?) reward=", line)
            assert m is not None, f"No action field in: {line}"
            action_raw = m.group(1)
            parsed = _json.loads(action_raw)
            assert "action_type" in parsed


@pytest.mark.asyncio
async def test_error_field_unmodified(capsys: pytest.CaptureFixture[str]) -> None:
    """Error messages must be output verbatim (no underscore replacement)."""
    garbage = ["not json at all"] * 12
    mock_client = _make_mock_client(garbage)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    found_error = False
    for line in out.strip().splitlines():
        if line.startswith("[STEP]"):
            m = re.search(r"error=(.+)$", line)
            if m and m.group(1) != "null":
                found_error = True
                assert "_" not in m.group(1) or " " in m.group(1), (
                    f"Error should be unmodified: {m.group(1)}"
                )
    assert found_error, "Expected at least one step with an error"


@pytest.mark.asyncio
async def test_default_baseline_runs_three_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    """With easy+medium+hard, inference emits one [START]/[END] block per task."""
    garbage = ["not json"] * 60
    mock_client = _make_mock_client(garbage)

    with patch.object(inference, "EPISODE_DIFFICULTIES", ["easy", "medium", "hard"]):
        with patch.object(inference, "_build_client", return_value=mock_client):
            await inference.run()

    out = capsys.readouterr().out
    assert out.count("[START]") == 3
    assert out.count("[END]") == 3


@pytest.mark.asyncio
async def test_success_is_score_based(capsys: pytest.CaptureFixture[str]) -> None:
    """success=true iff score > 0.1."""
    mock_client = _make_mock_client(OPTIMAL_EASY_RESPONSES)

    with patch.object(inference, "_build_client", return_value=mock_client):
        await inference.run()

    out = capsys.readouterr().out
    parsed = _parse_output(out)
    if parsed["score"] > 0.1:
        assert parsed["success"] is True
    else:
        assert parsed["success"] is False
