"""Tests for the deterministic grader (v2 — weighted keywords, SLA, multi-objective)."""

import pytest

from graders.grader import DeterministicGrader
from models.ticket import KeywordSpec


# -- weighted keyword scoring --------------------------------------------


class TestWeightedKeywordScore:
    def test_all_required_and_optional(self) -> None:
        spec = KeywordSpec(
            required=["refund", "duplicate"],
            optional=["charge", "process"],
            min_required_hits=2,
        )
        quality, pen = DeterministicGrader.weighted_keyword_score(
            "We will process your refund for the duplicate charge", spec
        )
        assert quality == pytest.approx(1.0)
        assert pen == 0.0

    def test_partial_required(self) -> None:
        spec = KeywordSpec(
            required=["refund", "duplicate", "credit", "apology"],
            optional=[],
            min_required_hits=2,
        )
        quality, pen = DeterministicGrader.weighted_keyword_score(
            "We will process your refund", spec
        )
        # 1/4 required hit, meets min (1 >= 2 is false) → halved
        assert quality < 0.5

    def test_min_required_not_met_halves_score(self) -> None:
        spec = KeywordSpec(
            required=["refund", "duplicate", "apology"],
            optional=["process"],
            min_required_hits=3,
        )
        full_q, _ = DeterministicGrader.weighted_keyword_score(
            "refund duplicate apology process", spec
        )
        partial_q, _ = DeterministicGrader.weighted_keyword_score(
            "refund duplicate process", spec  # only 2 of 3 required
        )
        assert partial_q < full_q
        assert partial_q <= full_q * 0.6  # halving kicks in

    def test_forbidden_penalty(self) -> None:
        spec = KeywordSpec(
            required=["refund"],
            optional=[],
            forbidden=["your fault", "no error"],
            min_required_hits=1,
        )
        _, pen = DeterministicGrader.weighted_keyword_score(
            "This is your fault and there is no error", spec
        )
        assert pen == pytest.approx(-0.06)  # 2 forbidden * -0.03

    def test_no_forbidden_hit(self) -> None:
        spec = KeywordSpec(
            required=["refund"],
            forbidden=["your fault"],
            min_required_hits=1,
        )
        _, pen = DeterministicGrader.weighted_keyword_score(
            "We will refund the amount", spec
        )
        assert pen == 0.0

    def test_empty_spec(self) -> None:
        spec = KeywordSpec()
        quality, pen = DeterministicGrader.weighted_keyword_score("anything", spec)
        assert quality == pytest.approx(1.0)
        assert pen == 0.0

    def test_case_insensitive(self) -> None:
        spec = KeywordSpec(required=["REFUND"], min_required_hits=1)
        quality, _ = DeterministicGrader.weighted_keyword_score("refund issued", spec)
        assert quality > 0.5

    def test_stem_prefix_matching(self) -> None:
        """Keyword stems match inflected forms via token-prefix."""
        spec = KeywordSpec(required=["apolog"], min_required_hits=1)
        quality, _ = DeterministicGrader.weighted_keyword_score(
            "We sincerely apologize", spec
        )
        assert quality > 0.5

    def test_punctuation_stripping(self) -> None:
        """Punctuation in text must not block keyword matches."""
        spec = KeywordSpec(required=["refund", "apolog"], min_required_hits=2)
        quality, _ = DeterministicGrader.weighted_keyword_score(
            "We apologize! Your refund: $29.99.", spec
        )
        assert quality > 0.5

    def test_token_boundary_prevents_false_match(self) -> None:
        """Single-word keyword must not match mid-word (e.g. 'fix' in 'prefix')."""
        spec = KeywordSpec(required=["fix"], min_required_hits=1)
        q_false, _ = DeterministicGrader.weighted_keyword_score("This is a prefix", spec)
        q_true, _ = DeterministicGrader.weighted_keyword_score("We will fix the bug", spec)
        assert q_false < q_true
        assert q_false < 0.5  # "fix" should NOT match "prefix"
        assert q_true > 0.5

    def test_multi_word_phrase_matching(self) -> None:
        """Multi-word forbidden keywords match as contiguous phrases."""
        spec = KeywordSpec(
            required=["refund"],
            forbidden=["not a bug"],
            min_required_hits=1,
        )
        _, pen_match = DeterministicGrader.weighted_keyword_score(
            "This is not a bug, please refund me", spec
        )
        _, pen_no_match = DeterministicGrader.weighted_keyword_score(
            "This is not related; a bug was found. Refund issued.", spec
        )
        assert pen_match == pytest.approx(-0.03)
        assert pen_no_match == 0.0  # words present but not contiguous

    def test_diversity_penalty_on_repetition(self) -> None:
        """Repeating a keyword should reduce quality via diversity factor."""
        spec = KeywordSpec(required=["refund"], min_required_hits=1)
        q_diverse, _ = DeterministicGrader.weighted_keyword_score(
            "We will process your refund for the duplicate charge", spec
        )
        q_spam, _ = DeterministicGrader.weighted_keyword_score(
            "refund refund refund refund refund refund", spec
        )
        assert q_spam < q_diverse

    def test_normal_text_no_diversity_penalty(self) -> None:
        """Well-written sentences should not incur diversity penalty."""
        spec = KeywordSpec(required=["investigat", "crash"], min_required_hits=2)
        quality, _ = DeterministicGrader.weighted_keyword_score(
            "We are investigating the crash you reported when uploading files.",
            spec,
        )
        assert quality == pytest.approx(1.0)


# -- SLA penalty ---------------------------------------------------------


class TestSLAPenalty:
    def test_within_sla(self) -> None:
        assert DeterministicGrader.sla_penalty(0, 3) == 0.0
        assert DeterministicGrader.sla_penalty(2, 3) == 0.0

    def test_at_deadline(self) -> None:
        assert DeterministicGrader.sla_penalty(3, 3) == pytest.approx(-0.02)

    def test_increasing_overage(self) -> None:
        assert DeterministicGrader.sla_penalty(4, 3) == pytest.approx(-0.04)
        assert DeterministicGrader.sla_penalty(5, 3) == pytest.approx(-0.06)
        assert DeterministicGrader.sla_penalty(7, 3) == pytest.approx(-0.10)


# -- compensation -------------------------------------------------------


class TestCompensationAccuracy:
    def test_in_range(self) -> None:
        assert DeterministicGrader.compensation_accuracy(30.0, (20.0, 50.0)) == 1.0

    def test_at_boundaries(self) -> None:
        assert DeterministicGrader.compensation_accuracy(20.0, (20.0, 50.0)) == 1.0
        assert DeterministicGrader.compensation_accuracy(50.0, (20.0, 50.0)) == 1.0

    def test_out_of_range(self) -> None:
        assert DeterministicGrader.compensation_accuracy(100.0, (20.0, 50.0)) == 0.3

    def test_both_none(self) -> None:
        assert DeterministicGrader.compensation_accuracy(None, None) == 1.0

    def test_offered_but_not_expected(self) -> None:
        assert DeterministicGrader.compensation_accuracy(10.0, None) == 0.5

    def test_expected_but_not_offered(self) -> None:
        assert DeterministicGrader.compensation_accuracy(None, (20.0, 50.0)) == 0.0


# -- refund constraint ---------------------------------------------------


class TestCheckRefundConstraint:
    def test_violation(self) -> None:
        assert DeterministicGrader.check_refund_constraint("do not offer refund > $50", 75.0) is True

    def test_within_limit(self) -> None:
        assert DeterministicGrader.check_refund_constraint("do not offer refund > $50", 30.0) is False

    def test_none_offered(self) -> None:
        assert DeterministicGrader.check_refund_constraint("do not offer refund > $50", None) is False


# -- multi-objective episode grading -------------------------------------


class TestGradeEpisode:
    def test_perfect_episode(self) -> None:
        score = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=3,
            max_steps=8,
            sla_steps=4,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.9

    def test_worst_episode(self) -> None:
        score = DeterministicGrader.grade_episode(
            classification_correct=False,
            routing_correct=False,
            response_quality=0.0,
            resolution_quality=0.0,
            escalation_score=0.0,
            urgency_handled=False,
            steps_taken=8,
            max_steps=8,
            sla_steps=3,
        )
        assert score == 0.0

    def test_sla_compliance_matters(self) -> None:
        within = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=3,
            max_steps=10,
            sla_steps=4,
        )
        over = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=7,
            max_steps=10,
            sla_steps=4,
        )
        assert within > over

    def test_urgency_dimension(self) -> None:
        with_urgency = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=3,
            max_steps=8,
            sla_steps=4,
        )
        without_urgency = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=False,
            steps_taken=3,
            max_steps=8,
            sla_steps=4,
        )
        assert with_urgency > without_urgency
        assert with_urgency - without_urgency == pytest.approx(0.10)

    def test_constraint_penalty(self) -> None:
        base = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=4,
            max_steps=8,
            sla_steps=5,
        )
        penalized = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=4,
            max_steps=8,
            sla_steps=5,
            constraints_violated=2,
        )
        assert penalized < base

    def test_score_clamped_at_zero(self) -> None:
        score = DeterministicGrader.grade_episode(
            classification_correct=True,
            routing_correct=True,
            response_quality=1.0,
            resolution_quality=1.0,
            escalation_score=1.0,
            urgency_handled=True,
            steps_taken=0,
            max_steps=8,
            sla_steps=4,
            constraints_violated=100,
        )
        assert score == 0.0
