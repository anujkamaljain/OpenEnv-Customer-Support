from __future__ import annotations

import pytest

from sandbox.env_adapter import SandboxEnv
from sandbox.env_adapter.bridge import SandboxConnectionError, SandboxValidationError
from sandbox.env_adapter.drill import build_curriculum_schedule


@pytest.mark.asyncio
async def test_sandbox_env_ticket_mode_uses_simulated_flow() -> None:
    env = SandboxEnv(cluster_base_url="http://127.0.0.1", chaos_url="http://127.0.0.1:6660")
    try:
        reset = await env.reset(mode="ticket", seed=0)
        assert reset.observation.mode == "ticket"
        # Ticket mode must NOT add sandbox info (drop-in compatibility).
        assert "sandbox" not in reset.info
        step = await env.step(
            {
                "action_type": "classify",
                "category": "general_inquiry",
                "priority": "medium",
            }
        )
        assert isinstance(step.reward, float)
        assert "sandbox" not in step.info
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_sandbox_env_incident_reset_falls_back_without_cluster() -> None:
    env = SandboxEnv(cluster_base_url="http://127.0.0.1", chaos_url="http://127.0.0.1:65531")
    try:
        reset = await env.reset(mode="incident", difficulty="easy", seed=0)
        sandbox = reset.info.get("sandbox", {})
        assert isinstance(sandbox, dict)
        # Without a cluster, must gracefully fall back to simulated backend.
        assert sandbox.get("backend") == "sim_fallback"
        # Reward and observation should still be valid.
        assert reset.observation.mode == "incident"
        assert reset.observation.incident_phase is not None
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_sandbox_env_drill_mode_initializes_schedule() -> None:
    env = SandboxEnv(cluster_base_url="http://127.0.0.1", chaos_url="http://127.0.0.1:65531")
    try:
        reset = await env.reset(
            mode="incident", difficulty="hard", seed=0, drill_mode=True, drill_seed=7
        )
        sandbox = reset.info.get("sandbox", {})
        # In sim_fallback (cluster unreachable), drill schedule is still
        # tracked but not executable. The api still must accept the call.
        assert isinstance(sandbox, dict)
    finally:
        await env.close()


def test_drill_schedule_is_deterministic_for_same_seed() -> None:
    first = build_curriculum_schedule(seed=11, difficulty="hard", max_steps=70)
    second = build_curriculum_schedule(seed=11, difficulty="hard", max_steps=70)
    assert [event.key for event in first] == [event.key for event in second]
    assert [event.deadline_step for event in first] == [event.deadline_step for event in second]


def test_drill_schedule_varies_by_seed() -> None:
    first = build_curriculum_schedule(seed=11, difficulty="hard", max_steps=70)
    second = build_curriculum_schedule(seed=12, difficulty="hard", max_steps=70)
    assert [event.key for event in first] != [event.key for event in second]


def test_drill_schedule_event_count_per_difficulty() -> None:
    easy = build_curriculum_schedule(seed=1, difficulty="easy", max_steps=40)
    medium = build_curriculum_schedule(seed=1, difficulty="medium", max_steps=50)
    hard = build_curriculum_schedule(seed=1, difficulty="hard", max_steps=70)
    nightmare = build_curriculum_schedule(seed=1, difficulty="nightmare", max_steps=80)
    assert len(easy) == 1
    assert len(medium) == 2
    assert len(hard) == 3
    assert len(nightmare) == 4


def test_drill_schedule_steps_within_episode_bounds() -> None:
    schedule = build_curriculum_schedule(seed=42, difficulty="nightmare", max_steps=80)
    for event in schedule:
        assert 0 <= event.step < 80
        assert event.deadline_step <= 80
        assert event.deadline_step > event.step


def test_drill_schedule_short_episode_is_empty() -> None:
    """Episodes shorter than 8 steps are too short for a meaningful drill."""
    schedule = build_curriculum_schedule(seed=1, difficulty="easy", max_steps=4)
    assert schedule == []


def test_sandbox_bridge_exception_hierarchy() -> None:
    """Validation errors are recoverable; connection errors are not."""
    val_err = SandboxValidationError(400, "bad input")
    conn_err = SandboxConnectionError("unreachable")
    # Both subclass SandboxBridgeError for unified handling.
    from sandbox.env_adapter.bridge import SandboxBridgeError
    assert isinstance(val_err, SandboxBridgeError)
    assert isinstance(conn_err, SandboxBridgeError)
    assert val_err.status_code == 400

