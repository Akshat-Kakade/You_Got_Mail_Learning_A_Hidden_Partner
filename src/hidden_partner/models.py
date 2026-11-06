"""The four learners from docs/model_spec.md.

All four share one prediction, one valuation and one choice rule. They differ in two ways:

  * `pooled` picks the features psi: shared partner traits, or one belief per send bin.
  * `bayesian` picks the update: exact posterior, or a constant gain.

                 pooled    per-send
  constant gain    M1         M2
  Bayesian         M3         M4
"""

from dataclasses import dataclass

import numpy as np

from hidden_partner.environment import Visible

ACTION_SENDS = np.array([0.0, 2.5, 7.0, 10.0])  # the send each action is valued at
N_ACTIONS = len(ACTION_SENDS)
SLOPE_SCALE = np.sqrt(0.5)  # s in the spec: how far pooled beliefs can differ across sends


def action_of(send: float) -> int:
    """Map a send to its action: 0, 1-4, 5-9 or 10."""
    if send == 0:
        return 0
    if send <= 4:
        return 1
    return 2 if send <= 9 else 3


def pooled_features(send: float) -> np.ndarray:
    return np.array([1.0, SLOPE_SCALE * (send - 5.5) / 4.5])


def per_send_features(send: float) -> np.ndarray:
    features = np.zeros(N_ACTIONS - 1)
    features[action_of(send) - 1] = 1.0
    return features


@dataclass(frozen=True)
class Model:
    name: str
    pooled: bool
    bayesian: bool

    @property
    def features(self):
        return pooled_features if self.pooled else per_send_features

    @property
    def gain_name(self) -> str:
        return "kappa" if self.bayesian else "alpha"


M1 = Model("M1", pooled=True, bayesian=False)
M2 = Model("M2", pooled=False, bayesian=False)
M3 = Model("M3", pooled=True, bayesian=True)
M4 = Model("M4", pooled=False, bayesian=True)
MODELS = (M1, M2, M3, M4)


@dataclass(frozen=True)
class Params:
    """Per-person parameters. `gain` is alpha for M1/M2 and kappa for M3/M4."""

    beta: float
    rho: float
    tau: float
    gain: float


class Learner:
    """One simulated mind: holds beliefs, values actions, and updates on what it can see."""

    def __init__(self, model: Model, params: Params, g0: float, lapse: float = 0.0):
        self.model = model
        self.params = params
        self.lapse = lapse  # chance of a random choice, whatever the values say
        self.action_features = np.array([model.features(x) for x in ACTION_SENDS[1:]])
        n_features = self.action_features.shape[1]
        self.mean = np.array([g0, 0.0]) if model.pooled else np.full(n_features, g0)
        self.cov = np.eye(n_features) / params.gain if model.bayesian else None
        self.last_action = None  # round 1 has no previous action, so no stickiness term

    def predicted_ratio(self, send: float) -> float:
        """mu_t(x): expected return per point sent."""
        return float(self.model.features(send) @ self.mean)

    def action_values(self) -> np.ndarray:
        """U(a) = x (mu(x) - 1) - (rho / 2) x^2, with U(0) = 0."""
        sends = ACTION_SENDS[1:]
        ratios = self.action_features @ self.mean
        values = sends * (ratios - 1) - 0.5 * self.params.rho * sends**2
        return np.concatenate(([0.0], values))

    def choice_probabilities(self) -> np.ndarray:
        logits = self.params.beta * self.action_values()
        if self.last_action is not None:
            logits[self.last_action] += self.params.tau
        logits -= logits.max()
        weights = np.exp(logits)
        return (1 - self.lapse) * weights / weights.sum() + self.lapse / N_ACTIONS

    def update(self, outcome: Visible) -> None:
        """Learn from a finished round. Sending 0 gives no y, so no belief changes."""
        self.last_action = action_of(outcome.send)
        if outcome.send == 0:
            return
        ratio = outcome.returned / outcome.send
        features = self.model.features(outcome.send)
        error = ratio - features @ self.mean
        if self.model.bayesian:
            self._bayesian_update(features, error)
        else:
            self.mean = self.mean + self.params.gain * features * error / (features @ features)

    def _bayesian_update(self, features: np.ndarray, error: float) -> None:
        """Exact posterior for y ~ N(psi'w, sigma^2); `cov` is in units of sigma^2."""
        spread = self.cov @ features
        gain = spread / (features @ spread + 1)
        self.mean = self.mean + gain * error
        self.cov = self.cov - np.outer(gain, spread)
