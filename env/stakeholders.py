"""Stakeholder patience and notification dynamics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StakeholderName = Literal["vp_engineering", "legal", "support_lead"]


class StakeholderState(BaseModel):
    """Internal state for one stakeholder."""

    stakeholder: StakeholderName
    patience: float = Field(default=1.0, ge=0.0, le=1.0)
    patience_decay: float = Field(ge=0.0, le=1.0)
    wants: str


class NotifyResult(BaseModel):
    """Notification action result."""

    model_config = {"frozen": True}

    stakeholder: StakeholderName
    accepted: bool
    patience_after: float
    message: str


class StakeholderManager:
    """Tracks patience of key stakeholders during incidents."""

    def __init__(self) -> None:
        self._stakeholders: dict[StakeholderName, StakeholderState] = {
            "vp_engineering": StakeholderState(
                stakeholder="vp_engineering",
                patience_decay=0.04,
                wants="status_updates",
            ),
            "legal": StakeholderState(
                stakeholder="legal",
                patience_decay=0.02,
                wants="compliance_assurance",
            ),
            "support_lead": StakeholderState(
                stakeholder="support_lead",
                patience_decay=0.03,
                wants="customer_resolution",
            ),
        }

    def tick(self) -> list[str]:
        """Decay patience and return critical warnings."""
        warnings: list[str] = []
        for state in self._stakeholders.values():
            state.patience = round(max(0.0, state.patience - state.patience_decay), 3)
            if state.patience <= 0.25:
                warnings.append(f"{state.stakeholder} patience critical")
        return warnings

    def notify(self, stakeholder: StakeholderName, message: str) -> NotifyResult:
        """Send stakeholder update and recover patience."""
        state = self._stakeholders[stakeholder]
        contains_wanted_signal = state.wants.split("_")[0] in message.lower()
        delta = 0.15 if contains_wanted_signal else 0.08
        state.patience = round(min(1.0, state.patience + delta), 3)
        return NotifyResult(
            stakeholder=stakeholder,
            accepted=True,
            patience_after=state.patience,
            message="Stakeholder updated.",
        )

    def get_patience_levels(self) -> dict[str, float]:
        """Return visible patience levels."""
        return {
            stakeholder: state.patience
            for stakeholder, state in self._stakeholders.items()
        }
