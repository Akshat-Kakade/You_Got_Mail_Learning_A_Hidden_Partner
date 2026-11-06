"""Model recovery for the main analysis: agents built from the real lapse fits.

Each agent borrows one real person's fitted parameters from Task 1, plays 15 rounds with
a 5% lapse, and is fitted by all four models with the same lapse. So these agents look much
more like people than the first recovery's. `g0` is each generating model's fitted value.

Run:  .venv/bin/python scripts/09_recovery_at_fitted_values.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import estimate_rules
from hidden_partner.evaluation import TIE_TOLERANCE
from hidden_partner.fitting import fit_person
from hidden_partner.models import MODELS, Params
from hidden_partner.simulate import send_sampler, simulate_agent

LAPSE = 0.05
N_ROUNDS = 15
AGENTS_PER_MODEL = 200
WORKERS = 6
RESULTS_FILE = "results/09_recovery_at_fitted_values.txt"

rounds = load_trustor_rounds()
RULES = estimate_rules(rounds)
SAMPLER = send_sampler(rounds[rounds["task"] == 1])
FITS = pd.read_csv("results/real_lapse_fits.csv")
FITS = FITS[(FITS["task"] == 1) & (FITS["model"] != "baseline")]


def agent_job(job):
    generator_index, agent = job
    generator = MODELS[generator_index]
    rng = np.random.default_rng([9, generator_index, agent])
    pool = FITS[FITS["model"] == generator.name]
    row = pool.iloc[rng.integers(len(pool))]
    truth = Params(row.beta, row.rho, row.tau, row.gain)
    sends, returns = simulate_agent(
        generator, truth, row.g0, RULES, PROFILES[agent % 3], N_ROUNDS, SAMPLER, rng, lapse=LAPSE
    )
    scores = {m.name: fit_person(m, row.g0, sends, returns, rng, lapse=LAPSE).log_likelihood for m in MODELS}
    return {
        "generator": generator.name,
        "mean_send": sends.mean(),
        "repeat_rate": float(np.mean(np.digitize(sends[1:], [1, 5, 10]) == np.digitize(sends[:-1], [1, 5, 10]))),
        **scores,
    }


def main():
    names = [m.name for m in MODELS]
    jobs = [(g, a) for g in range(len(MODELS)) for a in range(AGENTS_PER_MODEL)]
    with Pool(WORKERS) as pool:
        data = pd.DataFrame(pool.imap_unordered(agent_job, jobs, chunksize=4))
    scores = data[names]
    wins = scores.ge(scores.max(axis=1) - TIE_TOLERANCE, axis=0).astype(float)
    wins = wins.div(wins.sum(axis=1), axis=0)
    matrix = wins.groupby(data["generator"]).mean().reindex(names)
    pooled, bayes = {"M1", "M3"}, {"M3", "M4"}
    generators = data["generator"]
    same_pooling = np.where(generators.isin(pooled), wins[["M1", "M3"]].sum(axis=1), wins[["M2", "M4"]].sum(axis=1))
    same_dynamics = np.where(generators.isin(bayes), wins[["M3", "M4"]].sum(axis=1), wins[["M1", "M2"]].sum(axis=1))
    text = "\n".join(
        [
            "Rows true, columns best fit",
            matrix.round(3).to_string(),
            f"\nGeneralisation accuracy {same_pooling.mean():.3f}, dynamics accuracy {same_dynamics.mean():.3f}",
            f"Near-tied agents {(wins.max(axis=1) < 1).mean():.3f}",
            "\nDo agents look like people? (real Task 1: mean send 5.5, repeat 0.72)",
            data.groupby("generator")[["mean_send", "repeat_rate"]].mean().round(2).to_string(),
        ]
    )
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
