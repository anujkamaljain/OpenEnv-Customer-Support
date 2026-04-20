"""Deterministic incident bank for world simulation scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from models.incident import Difficulty, IncidentScenario

_INCIDENTS_DIR = Path(__file__).parent / "incidents"


class IncidentBank:
    """Loads pre-authored incidents and selects deterministically by seed."""

    def __init__(self, incidents_dir: Path | None = None) -> None:
        root = incidents_dir or _INCIDENTS_DIR
        self._by_difficulty: dict[Difficulty, list[IncidentScenario]] = {
            "easy": self._load(root / "easy.json"),
            "medium": self._load(root / "medium.json"),
            "hard": self._load(root / "hard.json"),
            "nightmare": self._load(root / "nightmare.json"),
        }
        self._all: list[IncidentScenario] = (
            self._by_difficulty["easy"]
            + self._by_difficulty["medium"]
            + self._by_difficulty["hard"]
            + self._by_difficulty["nightmare"]
        )
        if not self._all:
            raise ValueError(f"No incidents found in {root}")

    @staticmethod
    def _load(path: Path) -> list[IncidentScenario]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [IncidentScenario.model_validate(item) for item in raw]

    def get_incident(
        self, seed: int = 0, difficulty: str | None = None
    ) -> IncidentScenario:
        """Select one incident deterministically from a pool."""
        if difficulty is not None:
            pool = self._by_difficulty.get(difficulty)  # type: ignore[arg-type]
            if not pool:
                raise ValueError(f"No incidents for difficulty '{difficulty}'")
            return pool[seed % len(pool)]
        return self._all[seed % len(self._all)]

    def list_incidents(self, difficulty: str | None = None) -> Sequence[IncidentScenario]:
        """List incidents, optionally filtered by difficulty."""
        if difficulty is not None:
            return list(self._by_difficulty.get(difficulty, []))  # type: ignore[arg-type]
        return list(self._all)
