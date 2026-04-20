"""HTTP API contract tests for server reset behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_reset_defaults_to_ticket_mode() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"seed": 0, "difficulty": "easy"})
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["mode"] == "ticket"
    assert body["observation"]["incident_id"] is None
    assert body["info"]["session_id"] == "default"
    client.post("/close")


def test_reset_accepts_incident_mode() -> None:
    client = TestClient(app)
    response = client.post(
        "/reset",
        json={"seed": 0, "difficulty": "easy", "mode": "incident"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["mode"] == "incident"
    assert body["observation"]["incident_id"] is not None
    assert body["info"]["mode"] == "incident"
    assert body["info"]["session_id"] == "default"
    client.post("/close")


def test_reset_accepts_nightmare_for_incident_mode() -> None:
    client = TestClient(app)
    response = client.post(
        "/reset",
        json={"seed": 0, "difficulty": "nightmare", "mode": "incident"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["mode"] == "incident"
    assert body["observation"]["incident_phase"] == "triage"
    client.post("/close")


def test_sessions_are_isolated_by_session_id() -> None:
    client = TestClient(app)
    left = "session-left"
    right = "session-right"

    reset_left = client.post(
        "/reset",
        json={"seed": 0, "difficulty": "easy", "mode": "incident", "session_id": left},
    )
    assert reset_left.status_code == 200
    assert reset_left.json()["info"]["session_id"] == left

    reset_right = client.post(
        "/reset",
        json={"seed": 0, "difficulty": "easy", "mode": "ticket", "session_id": right},
    )
    assert reset_right.status_code == 200
    assert reset_right.json()["info"]["session_id"] == right

    state_left = client.get("/state", headers={"X-Session-ID": left})
    state_right = client.get("/state", headers={"X-Session-ID": right})
    assert state_left.status_code == 200
    assert state_right.status_code == 200
    assert state_left.json()["observation"]["mode"] == "incident"
    assert state_right.json()["observation"]["mode"] == "ticket"

    client.post("/close", headers={"X-Session-ID": left})
    client.post("/close", headers={"X-Session-ID": right})
