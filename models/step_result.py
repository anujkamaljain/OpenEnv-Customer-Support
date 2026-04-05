"""Step result returned by ``env.step()`` and ``env.reset()``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from models.observation import Observation


class StepResult(BaseModel):
    """Uniform return value for reset / step / state."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
