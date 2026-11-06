"""Checks on the four learners, including against closed-form answers."""

import numpy as np
import pytest

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import Visible, estimate_rules
from hidden_partner.fitting import fit_person, log_likelihood
from hidden_partner.models import (
    ACTION_SENDS,
    M1,
    M2,
    M3,
    M4,
    MODELS,
    Learner,
    Params,
    action_of,
    per_send_features,
    pooled_features,
)
from hidden_partner.simulate import send_sampler, simulate_agent

G0 = 1.3


def params_for(model, beta=2.0, rho=0.05, tau=0.5):
    return Params(beta, rho, tau, 2.0 if model.bayesian else 0.3)


def outcome(send, returned):
    return Visible(send, returned, 10 - send + returned)


HISTORY = [(10, 14), (7, 0), (3, 2), (0, 0), (10, 20), (5, 7), (2, 3), (9, 12)]


@pytest.mark.parametrize("send,expected", [(0, 0), (1, 1), (4, 1), (5, 2), (9, 2), (10, 3)])
def test_action_bins(send, expected):
    assert action_of(send) == expected


@pytest.mark.parametrize("model", MODELS)
def test_probabilities_sum_to_one(model):
    learner = Learner(model, params_for(model), G0)
    for send, returned in HISTORY:
        assert learner.choice_probabilities().sum() == pytest.approx(1.0)
        learner.update(outcome(send, returned))


def test_all_models_open_identically():
    opening = [
        Learner(m, Params(2.0, 0.05, 0.5, 2.0 if m.bayesian else 0.3), G0).choice_probabilities()
        for m in MODELS
    ]
    for probs in opening[1:]:
        assert probs == pytest.approx(opening[0])


@pytest.mark.parametrize("model", MODELS)
def test_value_of_sending_nothing_is_zero(model):
    assert Learner(model, params_for(model), G0).action_values()[0] == 0


@pytest.mark.parametrize("model", MODELS)
def test_sending_nothing_changes_no_belief(model):
    learner = Learner(model, params_for(model), G0)
    learner.update(outcome(8, 11))
    mean = learner.mean.copy()
    cov = None if learner.cov is None else learner.cov.copy()
    learner.update(outcome(0, 3))
    assert learner.mean == pytest.approx(mean)
    if cov is not None:
        assert learner.cov == pytest.approx(cov)
    assert learner.last_action == 0


@pytest.mark.parametrize("model", MODELS)
def test_stickiness_needs_a_previous_action(model):
    with_tau = Learner(model, params_for(model, tau=3.0), G0)
    without = Learner(model, params_for(model, tau=0.0), G0)
    assert with_tau.choice_probabilities() == pytest.approx(without.choice_probabilities())
    with_tau.update(outcome(7, 9))
    without.update(outcome(7, 9))
    assert with_tau.choice_probabilities()[2] > without.choice_probabilities()[2]


def test_m2_is_q_learning_at_the_bin_value_point():
    # At send 10 the value of the action is linear in the belief, so a delta rule on the
    # belief is a delta rule on the value, with the reward as its target.
    alpha, rho = 0.3, 0.05
    learner = Learner(M2, Params(1.0, rho, 0.0, alpha), G0)
    value = learner.action_values()[3]
    for returned in (14, 0, 20, 11):
        reward = returned - 10 - 0.5 * rho * 100
        value += alpha * (reward - value)
        learner.update(outcome(10, returned))
        assert learner.action_values()[3] == pytest.approx(value)


def test_m2_leaves_other_bins_alone():
    learner = Learner(M2, params_for(M2), G0)
    learner.update(outcome(10, 0))
    assert learner.predicted_ratio(7) == pytest.approx(G0)
    assert learner.predicted_ratio(10) != pytest.approx(G0)


def test_m1_moves_the_prediction_by_alpha_of_the_error():
    alpha = 0.3
    learner = Learner(M1, Params(1.0, 0.05, 0.0, alpha), G0)
    for send, returned in HISTORY:
        if send == 0:
            continue
        before = learner.predicted_ratio(send)
        learner.update(outcome(send, returned))
        assert learner.predicted_ratio(send) == pytest.approx(
            before + alpha * (returned / send - before)
        )


def test_pooled_learning_transfers_across_sends():
    learner = Learner(M1, params_for(M1), G0)
    learner.update(outcome(10, 0))
    assert learner.predicted_ratio(7) < G0


def test_m4_is_the_one_over_kappa_plus_n_rule():
    kappa = 2.5
    learner = Learner(M4, Params(1.0, 0.05, 0.0, kappa), G0)
    belief = G0
    for count, returned in enumerate((14, 0, 20, 11), start=1):
        belief += (returned / 10 - belief) / (kappa + count)
        learner.update(outcome(10, returned))
        assert learner.predicted_ratio(10) == pytest.approx(belief)


@pytest.mark.parametrize("model,features", [(M3, pooled_features), (M4, per_send_features)])
def test_bayesian_update_matches_the_batch_posterior(model, features):
    # The recursion must land exactly on the textbook posterior of Bayesian linear regression.
    kappa = 1.7
    learner = Learner(model, Params(1.0, 0.05, 0.0, kappa), G0)
    prior_mean = learner.mean.copy()
    for send, returned in HISTORY:
        learner.update(outcome(send, returned))
    sent = [(s, r) for s, r in HISTORY if s > 0]
    design = np.array([features(s) for s, _ in sent])
    ratios = np.array([r / s for s, r in sent])
    precision = kappa * np.eye(design.shape[1]) + design.T @ design
    batch_cov = np.linalg.inv(precision)
    batch_mean = batch_cov @ (kappa * prior_mean + design.T @ ratios)
    assert learner.mean == pytest.approx(batch_mean)
    assert learner.cov == pytest.approx(batch_cov)


@pytest.mark.parametrize("model", [M3, M4])
def test_bayesian_order_does_not_matter(model):
    forward = Learner(model, params_for(model), G0)
    backward = Learner(model, params_for(model), G0)
    for send, returned in HISTORY:
        forward.update(outcome(send, returned))
    for send, returned in reversed(HISTORY):
        backward.update(outcome(send, returned))
    assert forward.mean == pytest.approx(backward.mean)


def test_constant_gain_order_does_matter():
    forward = Learner(M2, params_for(M2), G0)
    backward = Learner(M2, params_for(M2), G0)
    for send, returned in HISTORY:
        forward.update(outcome(send, returned))
    for send, returned in reversed(HISTORY):
        backward.update(outcome(send, returned))
    assert forward.mean != pytest.approx(backward.mean)


def test_models_collapse_without_learning():
    probabilities = []
    for model in MODELS:
        gain = 1e7 if model.bayesian else 1e-7
        learner = Learner(model, Params(2.0, 0.05, 0.5, gain), G0)
        for send, returned in HISTORY:
            learner.update(outcome(send, returned))
        probabilities.append(learner.choice_probabilities())
    for probs in probabilities[1:]:
        assert probs == pytest.approx(probabilities[0], abs=1e-4)


def test_scored_rounds_exclude_choices_but_keep_learning():
    sends = np.array([10, 7, 5, 10])
    returns = np.array([14, 0, 6, 12])
    params = params_for(M3)
    everything = log_likelihood(M3, params, G0, sends, returns)
    first = log_likelihood(M3, params, G0, sends, returns, [True, True, False, False])
    last = log_likelihood(M3, params, G0, sends, returns, [False, False, True, True])
    assert first + last == pytest.approx(everything)


@pytest.fixture(scope="module")
def world():
    rounds = load_trustor_rounds()
    return estimate_rules(rounds), send_sampler(rounds[rounds["task"] == 1])


def test_sampler_stays_inside_each_action(world):
    _, sampler = world
    for action, (levels, _) in enumerate(sampler):
        assert all(action_of(level) == action for level in levels)


@pytest.mark.parametrize("model", MODELS)
def test_simulated_agent_plays_a_legal_game(world, model):
    rules, sampler = world
    sends, returns = simulate_agent(
        model, params_for(model), G0, rules, PROFILES[0], 20, sampler, np.random.default_rng(3)
    )
    assert len(sends) == 20
    assert ((sends >= 0) & (sends <= 10)).all()
    assert (returns <= 3 * sends).all()


def test_fitting_beats_a_poor_guess_and_finds_beta_on_long_data(world):
    rules, sampler = world
    rng = np.random.default_rng(5)
    truth = Params(3.0, 0.05, 0.5, 0.2)
    sends, returns = simulate_agent(M2, truth, G0, rules, PROFILES[0], 400, sampler, rng)
    fit = fit_person(M2, G0, sends, returns, rng)
    assert fit.log_likelihood >= log_likelihood(M2, truth, G0, sends, returns) - 1e-6
    assert fit.params.beta == pytest.approx(3.0, rel=0.5)


@pytest.mark.parametrize("model", MODELS)
def test_lapse_sets_a_floor_and_keeps_probabilities_valid(model):
    learner = Learner(model, Params(20.0, 0.05, 6.0, 2.0 if model.bayesian else 0.3), G0, lapse=0.05)
    for send, returned in HISTORY:
        probabilities = learner.choice_probabilities()
        assert probabilities.sum() == pytest.approx(1.0)
        assert probabilities.min() >= 0.05 / 4 - 1e-12
        learner.update(outcome(send, returned))


def test_zero_lapse_changes_nothing():
    sends, returns = np.array([10, 7, 5]), np.array([14, 0, 6])
    params = params_for(M3)
    assert log_likelihood(M3, params, G0, sends, returns, lapse=0.0) == pytest.approx(
        log_likelihood(M3, params, G0, sends, returns)
    )


def test_lapse_blends_with_a_uniform_choice():
    plain = Learner(M2, params_for(M2), G0).choice_probabilities()
    blended = Learner(M2, params_for(M2), G0, lapse=0.1).choice_probabilities()
    assert blended == pytest.approx(0.9 * plain + 0.1 / 4)


def test_lapse_does_not_change_learning():
    plain = Learner(M3, params_for(M3), G0)
    lapsing = Learner(M3, params_for(M3), G0, lapse=0.05)
    for send, returned in HISTORY:
        plain.update(outcome(send, returned))
        lapsing.update(outcome(send, returned))
    assert lapsing.mean == pytest.approx(plain.mean)


def test_lapse_caps_the_cost_of_one_surprise():
    sends, returns = np.array([0]), np.array([0])
    certain = Params(20.0, 0.001, 0.0, 0.3)  # strongly favours sending, so sending 0 is a shock
    assert log_likelihood(M2, certain, 2.0, sends, returns, lapse=0.05) >= np.log(0.05 / 4)
