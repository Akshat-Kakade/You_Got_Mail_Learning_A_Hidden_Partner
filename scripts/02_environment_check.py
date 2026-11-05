"""Does the rebuilt partner reproduce the real return distributions, not just the means?

For every task, profile and send, compare the observed returns with the simulator's
exact distribution. Rules built on one task are also scored on the other, so the check
isn't only grading the simulator on the data it was built from. Also reports how often
people chose the payoff-maximising send, and who never changed their send.

Run:  .venv/bin/python scripts/02_environment_check.py
"""

import numpy as np
import pandas as pd

from hidden_partner.data import PROFILES, constant_senders, load_trustor_rounds
from hidden_partner.environment import (
    FIRST_ROUND_WITH_DEFECTION,
    SEND_LEVELS,
    estimate_rules,
    return_distribution,
)

RESULTS_FILE = "results/02_environment_check.txt"
N_BOOTSTRAP = 4000


def total_variation(observed_probs, model_probs):
    """Half the summed absolute gaps: 0 means identical, 1 means no overlap."""
    return 0.5 * np.abs(observed_probs - model_probs).sum(axis=-1)


def score_distributions(rules, rounds, min_send, rng):
    """Sum total variation over cells, and compare with what sampling alone would give."""
    observed_total, null_totals, rows = 0.0, np.zeros(N_BOOTSTRAP), []
    data = rounds[rounds["round"] >= FIRST_ROUND_WITH_DEFECTION]
    for (task, profile, send), cell in data.groupby(["task", "profile", "send"]):
        if send < min_send:
            continue
        model = return_distribution(rules, profile, send, int(cell["ai_endowment"].iloc[0]))
        model = model / model.sum()
        counts = np.bincount(cell["returned"], minlength=len(model))
        if len(counts) > len(model):
            raise ValueError(f"return above cap in cell {(task, profile, send)}")
        n = len(cell)
        distance = total_variation(counts / n, model)
        null = total_variation(rng.multinomial(n, model, size=N_BOOTSTRAP) / n, model)
        observed_total += distance * n
        null_totals += null * n
        rows.append(
            {
                "task": task,
                "profile": profile,
                "send": send,
                "n": n,
                "tv": distance,
                "p_cell": (null >= distance).mean(),
                "mean_obs": cell["returned"].mean(),
                "mean_model": model @ np.arange(len(model)),
            }
        )
    cells = pd.DataFrame(rows)
    return {
        "weighted_tv": observed_total / cells["n"].sum(),
        "null_mean": null_totals.mean() / cells["n"].sum(),
        "p": (null_totals >= observed_total).mean(),
        "cells": len(cells),
        "cells_p_below_05": int((cells["p_cell"] < 0.05).sum()),
        "max_mean_gap": (cells["mean_obs"] - cells["mean_model"]).abs().max(),
    }, cells


def lookup_summary(rules):
    """How much of the return is fixed once profile, strategy, send and noise are known."""
    sizes = pd.Series({key: len(values) for key, (values, _) in rules.response_table.items()})
    possible = 3 * 4 * 11 * 5
    random_cells = sizes[sizes > 1]
    by_strategy = random_cells.groupby(level=[0, 1]).size().to_dict()
    return {
        "cells_seen": len(sizes),
        "cells_possible": possible,
        "share_single_valued": round(float((sizes == 1).mean()), 3),
        "multi_valued_by_profile_strategy": by_strategy,
    }


def expected_net_gain(rules, ai_endowment):
    """Expected return minus send, for each profile and send, under the true rules."""
    table = {}
    for profile in PROFILES:
        gains = []
        for send in SEND_LEVELS:
            probs = return_distribution(rules, profile, send, ai_endowment)
            gains.append(probs @ np.arange(len(probs)) - send)
        table[profile] = gains
    return pd.DataFrame(table, index=list(SEND_LEVELS)).round(2)


def main():
    rng = np.random.default_rng(20261105)
    rounds = load_trustor_rounds()
    task1, task3 = rounds[rounds["task"] == 1], rounds[rounds["task"] == 3]
    pooled = estimate_rules(rounds)
    lines = []

    def report(title, body):
        lines.extend([f"\n=== {title}", str(body)])

    report(
        "Estimated rules (pooled)",
        f"defect_prob {pooled.defect_prob:.4f}\n"
        f"strategy_probs {np.round(pooled.strategy_probs, 3)}\n"
        f"noise {dict(zip(pooled.noise_values.tolist(), np.round(pooled.noise_probs, 3).tolist()))}",
    )
    report("Lookup summary", lookup_summary(pooled))
    report("All defections return 0", bool((rounds.loc[rounds["defected"], "returned"] == 0).all()))

    checks = (
        ("In-sample: pooled rules, both tasks", pooled, rounds, 0),
        ("Out-of-sample: Task 1 rules on Task 3, sends 1-10", estimate_rules(task1), task3, 1),
        ("Out-of-sample: Task 3 rules on Task 1, sends 1-10", estimate_rules(task3), task1, 1),
    )
    worst = None
    for title, rules, data, min_send in checks:
        summary, cells = score_distributions(rules, data, min_send, rng)
        report(title, {k: round(float(v), 4) for k, v in summary.items()})
        if worst is None:
            worst = cells.sort_values("p_cell").head(8).round(3)
    report("Lowest-p cells, in-sample", worst.to_string(index=False))

    for task, ai_endowment in ((1, 0), (3, 10)):
        report(f"Task {task}: expected net gain (return - send) by send", expected_net_gain(pooled, ai_endowment))

    best_send = 10
    blocks = pd.cut(rounds["round"], [0, 5, 10, 15, 20], labels=["1-5", "6-10", "11-15", "16-20"])
    table = rounds.assign(block=blocks, chose_best=rounds["send"] == best_send)
    report(
        "Share sending 10, by block",
        table.groupby(["task", "profile", "block"], observed=True)["chose_best"].mean().unstack().round(3),
    )
    report(
        "Mean send, by block",
        table.groupby(["task", "profile", "block"], observed=True)["send"].mean().unstack().round(2),
    )

    constant = constant_senders(rounds).reset_index()
    constant = constant.merge(
        rounds.groupby(["id", "task"])[["send"]].first().reset_index(), on=["id", "task"]
    )
    flagged = constant[constant["constant"]]
    report("Constant senders by task", flagged.groupby("task").size().to_dict())
    report("Their fixed send", flagged.groupby("task")["send"].value_counts().sort_index().to_dict())
    report("Constant in both tasks", int((flagged.groupby("id").size() == 2).sum()))

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
