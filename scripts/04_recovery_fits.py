"""Simulate agents from each model, then fit all four models to every agent.

This writes the raw results. `05_recovery_report.py` summarises them. Agents play the
validated partner for the 15 training rounds. `g0` is held at its true value here, and
a separate check tests whether a shared `g0` can be found from the group.

Run:  .venv/bin/python scripts/04_recovery_fits.py
Set RECOVERY_ROUNDS to test another game length: it runs one repeat and skips the g0 check.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny matrices: threads only slow them down
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import estimate_rules
from hidden_partner.fitting import at_bound, fit_person
from hidden_partner.models import MODELS, Params
from hidden_partner.simulate import send_sampler, simulate_agent

DEFAULT_ROUNDS = 15
N_ROUNDS = int(os.environ.get("RECOVERY_ROUNDS", DEFAULT_ROUNDS))
N_REPEATS = 3 if N_ROUNDS == DEFAULT_ROUNDS else 1
AGENTS_PER_MODEL = 100
TRUE_G0 = 1.3
G0_GRID = np.round(np.arange(1.0, 1.61, 0.1), 1)
G0_CHECK_VALUES = (1.15, 1.3, 1.45)
G0_CHECK_AGENTS = 40

rounds = load_trustor_rounds()
RULES = estimate_rules(rounds)
SAMPLER = send_sampler(rounds[rounds["task"] == 1])


def draw_params(model, rng):
    """Plausible values, wide enough to include people who barely learn."""
    gain = np.exp(rng.uniform(np.log(0.5), np.log(20))) if model.bayesian else rng.uniform(0.05, 0.6)
    return Params(
        beta=float(np.exp(rng.uniform(np.log(0.5), np.log(6)))),
        rho=float(rng.uniform(0.02, 0.12)),
        tau=float(rng.uniform(0, 1.5)),
        gain=float(gain),
    )


def one_agent(job):
    repeat, generator_index, agent = job
    rng = np.random.default_rng([20261105, repeat, generator_index, agent])
    generator = MODELS[generator_index]
    truth = draw_params(generator, rng)
    profile = PROFILES[agent % len(PROFILES)]
    sends, returns = simulate_agent(generator, truth, TRUE_G0, RULES, profile, N_ROUNDS, SAMPLER, rng)
    rows = []
    for model in MODELS:
        fit = fit_person(model, TRUE_G0, sends, returns, rng)
        flags = at_bound(model, fit.params)
        rows.append(
            {
                "repeat": repeat,
                "generator": generator.name,
                "agent": agent,
                "fitted": model.name,
                "true_beta": truth.beta,
                "true_rho": truth.rho,
                "true_tau": truth.tau,
                "true_gain": truth.gain,
                "beta": fit.params.beta,
                "rho": fit.params.rho,
                "tau": fit.params.tau,
                "gain": fit.params.gain,
                "ll": fit.log_likelihood,
                "any_bound": any(flags.values()),
                **{f"bound_{name}": flag for name, flag in flags.items()},
                "n_actions_used": len({0 if s == 0 else 1 if s <= 4 else 2 if s <= 9 else 3 for s in sends}),
                "mean_send": sends.mean(),
                "repeat_rate": float(np.mean(
                    [(a == b) for a, b in zip(np.digitize(sends, [1, 5, 10]), np.digitize(sends, [1, 5, 10])[1:])]
                )),
            }
        )
    return rows


def g0_check(job):
    generator_index, true_g0, agent = job
    rng = np.random.default_rng([777, generator_index, int(true_g0 * 100), agent])
    model = MODELS[generator_index]
    truth = draw_params(model, rng)
    profile = PROFILES[agent % len(PROFILES)]
    sends, returns = simulate_agent(model, truth, true_g0, RULES, profile, N_ROUNDS, SAMPLER, rng)
    return [
        {
            "generator": model.name,
            "true_g0": true_g0,
            "agent": agent,
            "g0": float(g0),
            "ll": fit_person(model, float(g0), sends, returns, rng, n_starts=2).log_likelihood,
        }
        for g0 in G0_GRID
    ]


def main():
    jobs = [
        (repeat, index, agent)
        for repeat in range(N_REPEATS)
        for index in range(len(MODELS))
        for agent in range(AGENTS_PER_MODEL)
    ]
    checks = [
        (index, value, agent)
        for index in (1, 2)  # M2 and M3, the two diagonal corners
        for value in G0_CHECK_VALUES
        for agent in range(G0_CHECK_AGENTS)
    ]
    with Pool() as pool:
        fits = [row for rows in pool.imap_unordered(one_agent, jobs, chunksize=4) for row in rows]
        if N_ROUNDS != DEFAULT_ROUNDS:
            pd.DataFrame(fits).to_csv(f"results/recovery_fits_{N_ROUNDS}.csv", index=False)
            return
        grid = [row for rows in pool.imap_unordered(g0_check, checks, chunksize=2) for row in rows]
    pd.DataFrame(fits).to_csv("results/recovery_fits.csv", index=False)
    pd.DataFrame(grid).to_csv("results/recovery_g0.csv", index=False)


if __name__ == "__main__":
    main()
