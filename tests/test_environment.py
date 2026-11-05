"""Checks on the loader and the rebuilt partner."""

import dataclasses

import numpy as np
import pytest

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import (
    TrustGame,
    Visible,
    estimate_rules,
    return_cap,
    return_distribution,
)


@pytest.fixture(scope="module")
def rounds():
    return load_trustor_rounds()


@pytest.fixture(scope="module")
def rules(rounds):
    return estimate_rules(rounds)


def make_game(rules, seed=0, ai_endowment=0):
    return TrustGame(rules, PROFILES[0], 10, ai_endowment, np.random.default_rng(seed))


def test_loader_shape(rounds):
    assert len(rounds) == 302 * 20 * 2
    assert rounds["id"].nunique() == 302
    assert not rounds.duplicated(["id", "task", "round"]).any()


def test_payoff_identity_holds_in_data(rounds):
    assert (rounds["endowment"] - rounds["send"] + rounds["returned"] == rounds["payoff"]).all()


def test_returns_stay_within_cap_in_data(rounds):
    assert (rounds["returned"] <= 3 * rounds["send"] + rounds["ai_endowment"]).all()


def test_no_defection_in_round_one_in_data(rounds):
    assert not rounds.loc[rounds["round"] == 1, "defected"].any()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("send", [0, 1, 5, 10])
@pytest.mark.parametrize("ai_endowment", [0, 10])
def test_return_distribution_sums_to_one(rules, profile, send, ai_endowment):
    probs = return_distribution(rules, profile, send, ai_endowment)
    assert probs.sum() == pytest.approx(1.0)
    assert len(probs) == return_cap(send, ai_endowment) + 1


def test_simulator_matches_exact_distribution(rules):
    rng = np.random.default_rng(1)
    send, draws = 7, 60_000
    counts = np.zeros(return_cap(send, 0) + 1)
    for _ in range(draws):
        game = TrustGame(rules, PROFILES[2], 10, 0, rng, round=2)
        counts[game.play(send).returned] += 1
    exact = return_distribution(rules, PROFILES[2], send, 0)
    assert 0.5 * np.abs(counts / draws - exact).sum() < 0.02


def test_never_defects_in_round_one(rules):
    for seed in range(500):
        game = make_game(rules, seed)
        game.play(10)
        assert not game.hidden_log[0].defected


def test_sending_nothing_returns_nothing_in_task_one(rules):
    game = make_game(rules)
    assert all(game.play(0).returned == 0 for _ in range(20))


def test_payoff_identity_holds_in_simulator(rules):
    game = make_game(rules, ai_endowment=10)
    for send in [0, 3, 10, 7]:
        outcome = game.play(send)
        assert outcome.payoff == 10 - send + outcome.returned


def test_visible_outcome_carries_no_hidden_fields(rules):
    outcome = make_game(rules).play(5)
    assert isinstance(outcome, Visible)
    names = {f.name for f in dataclasses.fields(outcome)}
    assert names == {"send", "returned", "payoff"}


def test_rejects_impossible_send(rules):
    with pytest.raises(ValueError):
        make_game(rules).play(11)


def test_low_sends_follow_the_lookup_not_base_plus_noise(rules):
    # Social, send 1, strategy 1: noise +1 gives 2 but noise +2 gives 1, so no additive rule fits.
    assert rules.response_table[("Social_AI", 1, 1, 1)][0].tolist() == [2]
    assert rules.response_table[("Social_AI", 1, 1, 2)][0].tolist() == [1]


def test_unseen_cell_falls_back_to_coarser_table(rules):
    trimmed = dataclasses.replace(rules, response_table={})
    probs = return_distribution(trimmed, PROFILES[0], 5, 0)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1:].sum() > 0
