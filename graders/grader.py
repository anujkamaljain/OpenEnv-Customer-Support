"""Deterministic grader — all scoring uses exact string / numeric comparisons.

No LLM, no embeddings, no randomness.

v2 additions:
    - ``weighted_keyword_score`` with required / optional / forbidden keywords
    - ``sla_penalty`` for deadline overage
    - Multi-objective ``grade_episode`` with urgency + SLA dimensions
"""

from __future__ import annotations

import re

from models.ticket import KeywordSpec


class DeterministicGrader:
    """Stateless scoring utilities for actions and full episodes."""

    # ------------------------------------------------------------------
    # Keyword scoring
    # ------------------------------------------------------------------

    @staticmethod
    def keyword_overlap(text: str, keywords: list[str]) -> float:
        """Legacy helper — fraction of *keywords* found via substring match."""
        if not keywords:
            return 1.0
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched / len(keywords)

    @staticmethod
    def weighted_keyword_score(text: str, spec: KeywordSpec) -> tuple[float, float]:
        """Score *text* against a :class:`KeywordSpec`.

        Returns ``(quality, forbidden_penalty)`` where:
        - *quality* ∈ [0, 1] — weighted hit ratio (60 % required, 40 % optional).
          Halved when ``min_required_hits`` is not met.
        - *forbidden_penalty* ≤ 0 — ``-0.03`` per forbidden keyword found.
        """
        text_lower = text.lower()

        # --- required (60 % weight) ---
        if spec.required:
            req_hits = sum(1 for kw in spec.required if kw.lower() in text_lower)
            req_ratio = req_hits / len(spec.required)
            meets_min = req_hits >= spec.min_required_hits
        else:
            req_ratio = 1.0
            meets_min = True

        # --- optional (40 % weight) ---
        if spec.optional:
            opt_hits = sum(1 for kw in spec.optional if kw.lower() in text_lower)
            opt_ratio = opt_hits / len(spec.optional)
        else:
            opt_ratio = 1.0

        quality = 0.60 * req_ratio + 0.40 * opt_ratio
        if not meets_min:
            quality *= 0.5

        # --- forbidden ---
        penalty = 0.0
        if spec.forbidden:
            forbid_hits = sum(1 for kw in spec.forbidden if kw.lower() in text_lower)
            penalty = -0.03 * forbid_hits

        return round(quality, 4), round(penalty, 4)

    # ------------------------------------------------------------------
    # SLA
    # ------------------------------------------------------------------

    @staticmethod
    def sla_penalty(current_step: int, sla_steps: int) -> float:
        """Increasing penalty for every step beyond the SLA deadline.

        Returns 0.0 when within SLA, otherwise ``-0.02 * overage``.
        """
        if current_step < sla_steps:
            return 0.0
        overage = current_step - sla_steps + 1
        return round(-0.02 * overage, 4)

    # ------------------------------------------------------------------
    # Compensation
    # ------------------------------------------------------------------

    @staticmethod
    def compensation_accuracy(
        offered: float | None,
        expected_range: tuple[float, float] | None,
    ) -> float:
        """Score how well the offered compensation fits the expected range."""
        if expected_range is None:
            return 1.0 if offered is None else 0.5
        if offered is None:
            return 0.0
        lo, hi = expected_range
        if lo <= offered <= hi:
            return 1.0
        return 0.3

    @staticmethod
    def check_refund_constraint(constraint: str, offered: float | None) -> bool:
        """Return ``True`` if the refund *constraint* is violated."""
        if offered is None:
            return False
        match = re.search(r">\s*\$?(\d+(?:\.\d+)?)", constraint)
        if match:
            cap = float(match.group(1))
            return offered > cap
        return False

    # ------------------------------------------------------------------
    # Episode-level scoring
    # ------------------------------------------------------------------

    @staticmethod
    def grade_episode(
        *,
        classification_correct: bool | None,
        routing_correct: bool | None,
        response_quality: float | None,
        resolution_quality: float | None,
        escalation_score: float,
        urgency_handled: bool,
        steps_taken: int,
        max_steps: int,
        sla_steps: int,
        constraints_violated: int = 0,
    ) -> float:
        """Compute the final multi-objective episode score in ``[0, 1]``.

        Dimensions (weights sum to 1.0):

        ============================  ======
        Classification accuracy        15 %
        Routing accuracy               10 %
        Response quality               20 %
        Resolution quality             20 %
        Escalation correctness         10 %
        Urgency handling               10 %
        Step efficiency                 5 %
        SLA compliance                 10 %
        ============================  ======
        """
        cls_s = 1.0 if classification_correct else 0.0
        rte_s = 1.0 if routing_correct else 0.0
        rsp_s = response_quality if response_quality is not None else 0.0
        res_s = resolution_quality if resolution_quality is not None else 0.0
        esc_s = max(0.0, escalation_score)
        urg_s = 1.0 if urgency_handled else 0.0
        eff_s = max(0.0, 1.0 - steps_taken / max_steps) if max_steps > 0 else 0.0

        sla_overage = max(0, steps_taken - sla_steps)
        sla_s = max(0.0, 1.0 - sla_overage * 0.2)

        raw = (
            0.15 * cls_s
            + 0.10 * rte_s
            + 0.20 * rsp_s
            + 0.20 * res_s
            + 0.10 * esc_s
            + 0.10 * urg_s
            + 0.05 * eff_s
            + 0.10 * sla_s
        )

        penalty = constraints_violated * 0.05
        return round(max(0.0, min(raw - penalty, 1.0)), 4)
