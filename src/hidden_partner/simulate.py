"""Play a learner against the rebuilt partner. The learner only ever gets `Visible` outcomes."""

import numpy as np
import pandas as pd

from hidden_partner.environment import TrustGame
from hidden_partner.models import Learner, action_of, N_ACTIONS


def send_sampler(rounds: pd.DataFrame) -> list:
    """Observed sends within each action, because actions are bins and the game takes integers."""
    actions = rounds["send"].map(action_of)
    sampler = []
    for action in range(N_ACTIONS):
        counts = rounds.loc[actions == action, "send"].value_counts().sort_index()
        sampler.append((counts.index.to_numpy(), (counts / counts.sum()).to_numpy()))
    return sampler


def simulate_agent(model, params, g0, rules, profile, n_rounds, sampler, rng, ai_endowment=0, lapse=0.0):
    """Return the sends and returns of one simulated participant."""
    learner = Learner(model, params, g0, lapse)
    game = TrustGame(rules, profile, 10, ai_endowment, rng)
    sends, returns = [], []
    for _ in range(n_rounds):
        action = rng.choice(N_ACTIONS, p=learner.choice_probabilities())
        levels, probabilities = sampler[action]
        outcome = game.play(int(rng.choice(levels, p=probabilities)))
        learner.update(outcome)
        sends.append(outcome.send)
        returns.append(outcome.returned)
    return np.array(sends), np.array(returns)
