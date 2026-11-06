"""Likelihood and maximum-likelihood fitting. Uses the same `Learner` as the simulator."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from hidden_partner.environment import Visible
from hidden_partner.models import Learner, Model, Params, action_of

# Search limits, on the natural scale of each parameter.
LIMITS = {
    "beta": (0.01, 20.0),
    "rho": (0.001, 1.0),
    "tau": (-3.0, 6.0),
    "alpha": (0.005, 0.995),
    "kappa": (0.05, 200.0),
}
ENDOWMENT = 10


def log_likelihood(model, params, g0, sends, returns, scored=None, lapse=0.0) -> float:
    """Sum of log P(choice) over scored rounds. Beliefs update on every round regardless."""
    learner = Learner(model, params, g0, lapse)
    total = 0.0
    for index, (send, returned) in enumerate(zip(sends, returns)):
        if scored is None or scored[index]:
            total += np.log(learner.choice_probabilities()[action_of(send)] + 1e-12)
        learner.update(Visible(int(send), int(returned), ENDOWMENT - int(send) + int(returned)))
    return total


def _gain_limits(model: Model):
    return LIMITS[model.gain_name]


def _to_search(model: Model, params: Params) -> np.ndarray:
    gain = params.gain
    search_gain = np.log(gain) if model.bayesian else np.log(gain / (1 - gain))
    return np.array([np.log(params.beta), np.log(params.rho), params.tau, search_gain])


def _from_search(model: Model, point: np.ndarray) -> Params:
    """Unconstrained search values back to parameters: logs and a logit keep them in range."""
    gain = np.exp(point[3]) if model.bayesian else 1 / (1 + np.exp(-point[3]))
    return Params(float(np.exp(point[0])), float(np.exp(point[1])), float(point[2]), float(gain))


def _search_bounds(model: Model):
    low, high = _gain_limits(model)
    gain = (np.log(low), np.log(high)) if model.bayesian else (
        np.log(low / (1 - low)),
        np.log(high / (1 - high)),
    )
    return [tuple(np.log(LIMITS["beta"])), tuple(np.log(LIMITS["rho"])), LIMITS["tau"], gain]


@dataclass(frozen=True)
class Fit:
    params: Params
    log_likelihood: float


def fit_person(model, g0, sends, returns, rng, scored=None, n_starts=4, lapse=0.0) -> Fit:
    """Maximise the likelihood from several random starts and keep the best."""
    bounds = _search_bounds(model)

    def negative(point):
        return -log_likelihood(model, _from_search(model, point), g0, sends, returns, scored, lapse)

    best = None
    for _ in range(n_starts):
        start = np.array([rng.uniform(low, high) for low, high in bounds])
        result = minimize(negative, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    return Fit(_from_search(model, best.x), -float(best.fun))


def at_bound(model: Model, params: Params, tolerance: float = 1e-3) -> dict:
    """Which parameters ended up pressed against a search limit."""
    limits = {
        "beta": LIMITS["beta"],
        "rho": LIMITS["rho"],
        "tau": LIMITS["tau"],
        "gain": _gain_limits(model),
    }
    flags = {}
    for name, (low, high) in limits.items():
        value = getattr(params, name)
        flags[name] = bool(value <= low * (1 + tolerance) or value >= high * (1 - tolerance))
    return flags


def round_probabilities(model, params, g0, sends, returns, lapse=0.0) -> np.ndarray:
    """The model's choice probabilities before each round, given the real history so far."""
    learner = Learner(model, params, g0, lapse)
    rows = []
    for send, returned in zip(sends, returns):
        rows.append(learner.choice_probabilities())
        learner.update(Visible(int(send), int(returned), ENDOWMENT - int(send) + int(returned)))
    return np.array(rows)
