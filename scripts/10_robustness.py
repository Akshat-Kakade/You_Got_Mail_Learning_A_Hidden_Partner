"""Two small leftovers: do fitted parameters differ by profile, and the Task 1 strategy anomaly.

Run:  .venv/bin/python scripts/10_robustness.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from hidden_partner.data import PROFILES, load_trustor_rounds

RESULTS_FILE = "results/10_robustness.txt"
NAMES = ["M1", "M2", "M3", "M4"]
PARAMETERS = ["beta", "rho", "tau", "gain"]
TESTS_RUN_IN_DATA_CHECK = 30  # rough count of tests in the first partner audit


def profile_tests(fits):
    rows = []
    for (task, model), data in fits[fits["model"].isin(NAMES)].groupby(["task", "model"]):
        for parameter in PARAMETERS:
            groups = [data.loc[data["profile"] == p, parameter] for p in PROFILES]
            rows.append({"task": task, "model": model, "parameter": parameter, "p": stats.kruskal(*groups).pvalue})
    table = pd.DataFrame(rows)
    table["p_adjusted"] = (table["p"] * len(table)).clip(upper=1)
    return table


def send_by_profile(rounds):
    people = rounds.groupby(["task", "profile", "id"])["send"].mean().reset_index()
    means = people.groupby(["task", "profile"])["send"].mean().unstack().round(2)
    tests = {
        task: round(stats.kruskal(*[g["send"] for _, g in data.groupby("profile")]).pvalue, 4)
        for task, data in people.groupby("task")
    }
    return means.to_string(), tests


def strategy_anomaly(rounds):
    data = rounds[(rounds["task"] == 1) & (rounds["round"] >= 2) & (rounds["strategy"] > 0)]
    table = pd.crosstab(data["round"], data["strategy"])
    chi2, p, _, expected = stats.chi2_contingency(table)
    residuals = (table - expected) / np.sqrt(expected)
    shares = table.div(table.sum(axis=1), axis=0)
    per_strategy = {}
    for strategy in table.columns:
        one = pd.crosstab(data["round"], data["strategy"] == strategy)
        per_strategy[int(strategy)] = round(stats.chi2_contingency(one)[1], 4)

    # Does the wobble change what a person receives? Return given profile and send, across rounds.
    paid = rounds[(rounds["task"] == 1) & (rounds["round"] >= 2) & (~rounds["defected"]) & (rounds["send"] > 0)]
    surprise = paid["returned"] - paid.groupby(["profile", "send"])["returned"].transform("mean")
    trend = stats.linregress(paid["round"], surprise)
    by_round = surprise.groupby(paid["round"]).mean()
    return {
        "chi2 p": round(float(p), 4),
        "after correcting for about 30 tests": round(min(1.0, float(p) * TESTS_RUN_IN_DATA_CHECK), 3),
        "p per strategy": per_strategy,
        "largest standardised residual": round(float(residuals.abs().to_numpy().max()), 2),
        "share range across rounds": {int(s): (round(float(shares[s].min()), 3), round(float(shares[s].max()), 3)) for s in table.columns},
        "return drift per round": round(float(trend.slope), 4),
        "drift p": round(float(trend.pvalue), 3),
        "mean return shift by round, min and max": (round(float(by_round.min()), 2), round(float(by_round.max()), 2)),
    }


def main():
    rounds = load_trustor_rounds()
    fits = pd.read_csv("results/real_lapse_fits.csv")
    tests = profile_tests(fits)
    lines = [
        "=== Parameters by profile (Kruskal-Wallis, lapse fits)",
        f"{len(tests)} tests, smallest p {tests['p'].min():.4f}, below 0.05: {(tests['p'] < 0.05).sum()}, "
        f"below 0.05 after correction: {(tests['p_adjusted'] < 0.05).sum()}",
        tests.sort_values("p").head(5).round(4).to_string(index=False),
    ]
    means, send_tests = send_by_profile(rounds)
    lines += ["\n=== Mean send by profile", means, f"Kruskal-Wallis p by task: {send_tests}"]
    lines += ["\n=== Task 1 strategy mix across rounds 2-20"]
    lines += [f"{key}: {value}" for key, value in strategy_anomaly(rounds).items()]
    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
