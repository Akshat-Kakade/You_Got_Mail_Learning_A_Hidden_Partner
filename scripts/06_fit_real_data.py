"""Fit all four models to the real data, each task separately.

Stage 1 picks the shared g0 per task and model from a grid, on training rounds only.
Stage 2 fits each person at that g0, then scores held-out rounds, an early-rounds block,
and transfer to the other task. Beliefs keep updating through every scored round; only
the parameters are frozen.

Run:  .venv/bin/python scripts/06_fit_real_data.py
Set FIT_LAPSE=0.05 to fit with a fixed lapse. Those results get a `real_lapse_` prefix, so
the raw fits are never overwritten.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd

from hidden_partner.data import TRUSTOR_TASKS, load_trustor_rounds
from hidden_partner.evaluation import baseline_log_likelihood
from hidden_partner.fitting import at_bound, fit_person, log_likelihood, round_probabilities
from hidden_partner.models import MODELS

TRAIN_ROUNDS = 15
EARLY_ROUNDS = 5
N_ROUNDS = 20
G0_GRID = np.round(np.arange(1.0, 2.01, 0.1), 1)
GRID_STARTS = 2
FINAL_STARTS = 6
LAPSE = float(os.environ.get("FIT_LAPSE", 0.0))
PREFIX = "results/real_lapse_" if LAPSE > 0 else "results/real_"

rounds = load_trustor_rounds()
SEQUENCES = {
    key: (group["send"].to_numpy(), group["returned"].to_numpy())
    for key, group in rounds.groupby(["id", "task"])
}
PROFILE = rounds.groupby("id")["profile"].first().to_dict()
PEOPLE = sorted(PROFILE)


def grid_job(job):
    task, model_index, g0, person = job
    rng = np.random.default_rng([11, task, model_index, int(g0 * 10), PEOPLE.index(person)])
    sends, returns = SEQUENCES[(person, task)]
    fit = fit_person(
        MODELS[model_index], g0, sends[:TRAIN_ROUNDS], returns[:TRAIN_ROUNDS], rng, n_starts=GRID_STARTS, lapse=LAPSE
    )
    return task, MODELS[model_index].name, g0, fit.log_likelihood


def person_job(job):
    task, model_index, g0, person = job
    model = MODELS[model_index]
    rng = np.random.default_rng([22, task, model_index, PEOPLE.index(person)])
    sends, returns = SEQUENCES[(person, task)]
    other = 3 if task == 1 else 1
    other_sends, other_returns = SEQUENCES[(person, other)]

    fit = fit_person(
        model, g0, sends[:TRAIN_ROUNDS], returns[:TRAIN_ROUNDS], rng, n_starts=FINAL_STARTS, lapse=LAPSE
    )
    held_out = np.arange(N_ROUNDS) >= TRAIN_ROUNDS
    early = np.arange(N_ROUNDS) < EARLY_ROUNDS
    early_fit = fit_person(model, g0, sends, returns, rng, scored=~early, n_starts=FINAL_STARTS, lapse=LAPSE)
    flags = at_bound(model, fit.params)
    probabilities = round_probabilities(model, fit.params, g0, sends, returns, LAPSE)
    row = {
        "id": person,
        "profile": PROFILE[person],
        "task": task,
        "model": model.name,
        "g0": g0,
        "beta": fit.params.beta,
        "rho": fit.params.rho,
        "tau": fit.params.tau,
        "gain": fit.params.gain,
        "train_ll": fit.log_likelihood,
        "held_out_ll": log_likelihood(model, fit.params, g0, sends, returns, held_out, LAPSE),
        "early_ll": log_likelihood(model, early_fit.params, g0, sends, returns, early, LAPSE),
        "transfer_ll": log_likelihood(model, fit.params, g0, other_sends, other_returns, lapse=LAPSE),
        **{f"bound_{name}": flag for name, flag in flags.items()},
    }
    return row, [
        {"id": person, "task": task, "model": model.name, "round": index + 1,
         "p0": p[0], "p1": p[1], "p2": p[2], "p3": p[3]}
        for index, p in enumerate(probabilities)
    ]


def baselines():
    rows = []
    for (person, task), (sends, _) in SEQUENCES.items():
        other_sends = SEQUENCES[(person, 3 if task == 1 else 1)][0]
        rows.append(
            {
                "id": person,
                "profile": PROFILE[person],
                "task": task,
                "model": "baseline",
                "train_ll": baseline_log_likelihood(sends[:TRAIN_ROUNDS], sends[:TRAIN_ROUNDS], LAPSE),
                "held_out_ll": baseline_log_likelihood(sends[:TRAIN_ROUNDS], sends[TRAIN_ROUNDS:], LAPSE),
                "early_ll": baseline_log_likelihood(sends[EARLY_ROUNDS:], sends[:EARLY_ROUNDS], LAPSE),
                "transfer_ll": baseline_log_likelihood(sends[:TRAIN_ROUNDS], other_sends, LAPSE),
            }
        )
    return pd.DataFrame(rows)


def main():
    grid_jobs = [
        (task, index, float(g0), person)
        for task in TRUSTOR_TASKS
        for index in range(len(MODELS))
        for g0 in G0_GRID
        for person in PEOPLE
    ]
    with Pool() as pool:
        grid = pd.DataFrame(
            pool.imap_unordered(grid_job, grid_jobs, chunksize=16),
            columns=["task", "model", "g0", "ll"],
        )
        totals = grid.groupby(["task", "model", "g0"])["ll"].sum().reset_index()
        totals.to_csv(f"{PREFIX}g0_grid.csv", index=False)
        best = totals.loc[totals.groupby(["task", "model"])["ll"].idxmax()]
        chosen = {(row.task, row.model): row.g0 for row in best.itertuples()}
        print(best.to_string(index=False), flush=True)

        jobs = [
            (task, index, chosen[(task, model.name)], person)
            for task in TRUSTOR_TASKS
            for index, model in enumerate(MODELS)
            for person in PEOPLE
        ]
        results = list(pool.imap_unordered(person_job, jobs, chunksize=4))
    fits = pd.concat([pd.DataFrame([row for row, _ in results]), baselines()], ignore_index=True)
    fits.to_csv(f"{PREFIX}fits.csv", index=False)
    pd.DataFrame([p for _, rows in results for p in rows]).to_csv(f"{PREFIX}predictions.csv", index=False)
    print(f"{len(results)} person fits done")


if __name__ == "__main__":
    main()
