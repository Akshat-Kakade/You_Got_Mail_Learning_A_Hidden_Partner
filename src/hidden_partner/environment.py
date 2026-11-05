"""A simulator of the AI partner, rebuilt from the observed rows.

Three kinds of information are kept apart on purpose:

  * `PartnerRules`: the true generating mechanism, estimated from the AI's own columns.
  * `Visible`: what a participant can observe after a round.
  * `Hidden`: the AI's internal draw, logged only for validating this simulator.

`TrustGame.play` returns only `Visible`, so a simulated learner never sees `Hidden`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hidden_partner.data import MULTIPLIER

SEND_LEVELS = tuple(range(11))
STRATEGIES = (1, 2, 3, 4)
FIRST_ROUND_WITH_DEFECTION = 2  # the AI never defected in round 1 (0 of 302, both tasks)


@dataclass(frozen=True)
class Visible:
    """What a participant can know about a finished round."""

    send: int
    returned: int
    payoff: int


@dataclass(frozen=True)
class Hidden:
    """The AI's internal draw. Never pass this to a model of a participant."""

    defected: bool
    strategy: int | None
    noise: int | None


@dataclass(frozen=True)
class PartnerRules:
    """The estimated generating mechanism of the partner."""

    defect_prob: float
    strategy_probs: np.ndarray
    noise_values: np.ndarray
    noise_probs: np.ndarray
    response_table: dict  # (profile, strategy, send, noise) -> (returns, probabilities)
    fallback_table: dict  # (profile, strategy, send) -> (returns, probabilities)


def return_cap(send: int, ai_endowment: int) -> int:
    """The most the AI could return: what it received plus what it started with."""
    return MULTIPLIER * send + ai_endowment


def _empirical(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = values.value_counts().sort_index()
    return counts.index.to_numpy(), (counts / counts.sum()).to_numpy()


def estimate_rules(rounds: pd.DataFrame) -> PartnerRules:
    """Estimate the partner's rules from observed rows, including its hidden columns.

    The return is stored as a lookup on (profile, strategy, send, noise), not as a
    formula. "Base plus noise" holds at high sends but breaks at low ones, where noise
    can be ignored or reversed, so a lookup avoids assuming a rule the data contradicts.
    Rows that can only return 0 (Task 1, send 0) say nothing, so they are left out.
    """
    after_first = rounds[rounds["round"] >= FIRST_ROUND_WITH_DEFECTION]
    paid = rounds[~rounds["defected"]]
    informative = paid[return_cap_of(paid) > 0]

    strategy_counts = paid["strategy"].value_counts().reindex(STRATEGIES, fill_value=0)
    noise_counts = paid["noise"].value_counts().sort_index()

    fine = informative.groupby(["profile", "strategy", "send", "noise"])["returned"]
    coarse = informative.groupby(["profile", "strategy", "send"])["returned"]
    return PartnerRules(
        defect_prob=float(after_first["defected"].mean()),
        strategy_probs=(strategy_counts / strategy_counts.sum()).to_numpy(),
        noise_values=noise_counts.index.to_numpy(),
        noise_probs=(noise_counts / noise_counts.sum()).to_numpy(),
        response_table={key: _empirical(values) for key, values in fine},
        fallback_table={key: _empirical(values) for key, values in coarse},
    )


def return_cap_of(rows: pd.DataFrame) -> pd.Series:
    return MULTIPLIER * rows["send"] + rows["ai_endowment"]


def _response_cell(rules: PartnerRules, profile: str, strategy: int, send: int, noise: int):
    """Possible returns for this draw, using the coarser table if the cell was never seen."""
    cell = rules.response_table.get((profile, strategy, send, noise))
    if cell is None:
        cell = rules.fallback_table.get((profile, strategy, send))
    if cell is None:
        return np.array([0]), np.array([1.0])
    return cell


def return_distribution(
    rules: PartnerRules, profile: str, send: int, ai_endowment: int, can_defect: bool = True
) -> np.ndarray:
    """Exact probability of every possible return, indexed by return amount."""
    cap = return_cap(send, ai_endowment)
    probs = np.zeros(cap + 1)
    if cap == 0:
        probs[0] = 1.0
        return probs
    defect_prob = rules.defect_prob if can_defect else 0.0
    probs[0] += defect_prob
    for strategy, strategy_prob in zip(STRATEGIES, rules.strategy_probs):
        for noise, noise_prob in zip(rules.noise_values, rules.noise_probs):
            values, value_probs = _response_cell(rules, profile, strategy, send, int(noise))
            for value, value_prob in zip(values, value_probs):
                returned = min(int(value), cap)
                probs[returned] += (1 - defect_prob) * strategy_prob * noise_prob * value_prob
    return probs


@dataclass
class TrustGame:
    """One participant's repeated game against one partner."""

    rules: PartnerRules
    profile: str
    endowment: int
    ai_endowment: int
    rng: np.random.Generator
    round: int = 1
    hidden_log: list = field(default_factory=list)

    def play(self, send: int) -> Visible:
        """Play one round and return only what the participant may observe."""
        if not 0 <= send <= self.endowment:
            raise ValueError(f"send must be between 0 and {self.endowment}, got {send}")
        returned, hidden = self._partner_response(send)
        self.hidden_log.append(hidden)
        self.round += 1
        return Visible(send, returned, self.endowment - send + returned)

    def _partner_response(self, send: int) -> tuple[int, Hidden]:
        can_defect = self.round >= FIRST_ROUND_WITH_DEFECTION
        if can_defect and self.rng.random() < self.rules.defect_prob:
            return 0, Hidden(True, None, None)
        strategy = int(self.rng.choice(STRATEGIES, p=self.rules.strategy_probs))
        noise = int(self.rng.choice(self.rules.noise_values, p=self.rules.noise_probs))
        cap = return_cap(send, self.ai_endowment)
        if cap == 0:
            return 0, Hidden(False, strategy, noise)
        values, value_probs = _response_cell(self.rules, self.profile, strategy, send, noise)
        return min(int(self.rng.choice(values, p=value_probs)), cap), Hidden(False, strategy, noise)
