"""Checks on the scoring tools."""

import numpy as np
import pandas as pd
import pytest

from hidden_partner.evaluation import (
    baseline_log_likelihood,
    information_criteria,
    paired_bootstrap,
    win_shares,
)
from hidden_partner.fitting import log_likelihood, round_probabilities
from hidden_partner.models import M3, Params, action_of


def test_baseline_uses_smoothed_training_frequencies():
    # Training: 3 choices in the top action, so counts are 1, 1, 1, 4 out of 7.
    assert baseline_log_likelihood([10, 10, 10], [10]) == pytest.approx(np.log(4 / 7))
    assert baseline_log_likelihood([10, 10, 10], [0]) == pytest.approx(np.log(1 / 7))


def test_baseline_stays_finite_for_a_never_chosen_action():
    assert np.isfinite(baseline_log_likelihood([5] * 15, [0, 0, 0, 0, 0]))


def test_bootstrap_interval_contains_the_mean():
    result = paired_bootstrap(np.random.default_rng(0).normal(1, 1, 300))
    assert result["low"] < result["mean"] < result["high"]
    assert result["low"] > 0


def test_win_shares_split_ties_and_sum_to_one():
    scores = pd.DataFrame({"a": [1.0, 1.0, 0.0], "b": [0.0, 1.0, 1.0]})
    shares = win_shares(scores)
    assert shares["a"] == pytest.approx(0.5)
    assert shares.sum() == pytest.approx(1.0)


def test_information_criteria():
    result = information_criteria(-10.0, 4, 15)
    assert result["aic"] == pytest.approx(28.0)
    assert result["bic"] == pytest.approx(4 * np.log(15) + 20)


def test_round_probabilities_agree_with_the_likelihood():
    sends = np.array([10, 7, 0, 3])
    returns = np.array([14, 0, 0, 2])
    params = Params(2.0, 0.05, 0.5, 2.0)
    probabilities = round_probabilities(M3, params, 1.3, sends, returns)
    picked = [probabilities[i, action_of(s)] for i, s in enumerate(sends)]
    assert np.log(picked).sum() == pytest.approx(log_likelihood(M3, params, 1.3, sends, returns))


def test_baseline_lapse_matches_the_blend():
    plain = 4 / 7
    assert baseline_log_likelihood([10, 10, 10], [10], lapse=0.05) == pytest.approx(
        np.log(0.95 * plain + 0.05 / 4)
    )
