"""Scoring tools: the baseline, paired bootstrap intervals, and win shares with split ties."""

import numpy as np
import pandas as pd

from hidden_partner.models import N_ACTIONS, action_of

TIE_TOLERANCE = 0.01


def baseline_log_likelihood(train_sends, test_sends, lapse=0.0) -> float:
    """Predict each choice from how often it was chosen in training.

    Add-one smoothing gives unseen actions a small probability, because a raw frequency of
    zero would make a single new choice infinitely surprising.
    """
    counts = np.ones(N_ACTIONS)
    for send in train_sends:
        counts[action_of(send)] += 1
    probabilities = (1 - lapse) * counts / counts.sum() + lapse / N_ACTIONS
    return float(sum(np.log(probabilities[action_of(send)]) for send in test_sends))


def paired_bootstrap(differences, n_draws=5000, seed=20261105) -> dict:
    """Mean per-person difference with a 95% interval, resampling people."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(n_draws, len(values)))
    means = values[picks].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def win_shares(scores: pd.DataFrame, tolerance=TIE_TOLERANCE) -> pd.Series:
    """Share of people each column wins. Near-ties split the win equally."""
    winners = scores.ge(scores.max(axis=1) - tolerance, axis=0).astype(float)
    return winners.div(winners.sum(axis=1), axis=0).mean()


def information_criteria(log_likelihood, n_parameters, n_choices) -> dict:
    return {
        "aic": 2 * n_parameters - 2 * log_likelihood,
        "bic": n_parameters * np.log(n_choices) - 2 * log_likelihood,
    }
