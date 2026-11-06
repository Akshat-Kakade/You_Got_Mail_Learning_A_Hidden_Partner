"""Small sanity check that the lapse implementation hasn't broken anything.

A few representative settings per model, not a full study. Agents play with a 5% lapse and
are fitted with the same lapse, so parameters should land near the truth on longer games,
and the fit should not get worse when the true model is the one being fitted.

Run:  .venv/bin/python scripts/08_lapse_recovery_check.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import estimate_rules
from hidden_partner.fitting import fit_person
from hidden_partner.models import MODELS, Params
from hidden_partner.simulate import send_sampler, simulate_agent

LAPSE = 0.05
G0 = 1.4
AGENTS = 20
GAME_LENGTHS = (15, 100)
WORKERS = 6  # leave half the machine free
RESULTS_FILE = "results/08_lapse_recovery_check.txt"
# Slow and fast learners, at a moderate and a high consistency.
SETTINGS = (
    ("slow", 2.0, 0.08, 0.5, 0.1, 20.0),
    ("fast", 2.0, 0.08, 0.5, 0.4, 1.0),
    ("fast, consistent", 6.0, 0.08, 1.0, 0.4, 1.0),
)

rounds = load_trustor_rounds()
RULES = estimate_rules(rounds)
SAMPLER = send_sampler(rounds[rounds["task"] == 1])


def job(task):
    model_index, setting_index, length, agent = task
    model = MODELS[model_index]
    label, beta, rho, tau, alpha, kappa = SETTINGS[setting_index]
    truth = Params(beta, rho, tau, kappa if model.bayesian else alpha)
    rng = np.random.default_rng([8, model_index, setting_index, length, agent])
    profile = PROFILES[agent % len(PROFILES)]
    sends, returns = simulate_agent(
        model, truth, G0, RULES, profile, length, SAMPLER, rng, lapse=LAPSE
    )
    fit = fit_person(model, G0, sends, returns, rng, lapse=LAPSE)
    return {
        "model": model.name, "setting": label, "rounds": length,
        "true_beta": beta, "beta": fit.params.beta,
        "true_rho": rho, "rho": fit.params.rho,
        "true_tau": tau, "tau": fit.params.tau,
        "true_gain": truth.gain, "gain": fit.params.gain,
    }


def main():
    tasks = [
        (m, s, length, agent)
        for m in range(len(MODELS))
        for s in range(len(SETTINGS))
        for length in GAME_LENGTHS
        for agent in range(AGENTS)
    ]
    with Pool(WORKERS) as pool:
        data = pd.DataFrame(pool.imap_unordered(job, tasks, chunksize=4))
    table = data.groupby(["rounds", "model", "setting"]).agg(
        true_beta=("true_beta", "first"), beta=("beta", "median"),
        true_rho=("true_rho", "first"), rho=("rho", "median"),
        true_tau=("true_tau", "first"), tau=("tau", "median"),
        true_gain=("true_gain", "first"), gain=("gain", "median"),
    ).round(3)
    text = table.to_string()
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
