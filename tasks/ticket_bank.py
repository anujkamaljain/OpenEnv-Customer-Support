"""Deterministic ticket bank — loads tickets from JSON and selects by seed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from models.ticket import Difficulty, TicketData

_TICKETS_DIR = Path(__file__).parent / "tickets"


class TicketBank:
    """Loads pre-authored tickets and provides deterministic selection."""

    def __init__(self, tickets_dir: Path | None = None) -> None:
        root = tickets_dir or _TICKETS_DIR
        self._by_difficulty: dict[Difficulty, list[TicketData]] = {
            "easy": self._load(root / "easy.json"),
            "medium": self._load(root / "medium.json"),
            "hard": self._load(root / "hard.json"),
        }
        self._all: list[TicketData] = (
            self._by_difficulty["easy"]
            + self._by_difficulty["medium"]
            + self._by_difficulty["hard"]
        )
        if not self._all:
            raise ValueError(f"No tickets found in {root}")

    @staticmethod
    def _load(path: Path) -> list[TicketData]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [TicketData.model_validate(item) for item in raw]

    def get_ticket(
        self,
        seed: int = 0,
        difficulty: str | None = None,
    ) -> TicketData:
        """Select a ticket deterministically.  Same seed → same ticket."""
        if difficulty is not None:
            pool = self._by_difficulty.get(difficulty)  # type: ignore[arg-type]
            if not pool:
                raise ValueError(f"No tickets for difficulty '{difficulty}'")
        else:
            pool = self._all
        return pool[seed % len(pool)]

    def list_tickets(self, difficulty: str | None = None) -> Sequence[TicketData]:
        """Return all tickets, optionally filtered by difficulty."""
        if difficulty is not None:
            return list(self._by_difficulty.get(difficulty, []))  # type: ignore[arg-type]
        return list(self._all)
