"""Summarise the real-data fits. Each task is reported on its own, never pooled.

Primary: held-out log-likelihood, as paired per-person differences with bootstrap intervals.
Secondary: win shares, AIC/BIC, parameters, and predicted against observed behaviour.

Run:  .venv/bin/python scripts/07_results_report.py
Set REPORT_LAPSE=1 to report the lapse fits, with a side-by-side against the raw fits.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidden_partner.data import constant_senders, load_trustor_rounds
from hidden_partner.environment import Visible
from hidden_partner.evaluation import information_criteria, paired_bootstrap, win_shares
from hidden_partner.models import ACTION_SENDS, MODELS, Learner, Params, action_of

USE_LAPSE = os.environ.get("REPORT_LAPSE") == "1"
PREFIX = "results/real_lapse_" if USE_LAPSE else "results/real_"
RESULTS_FILE = "results/07_results_lapse.txt" if USE_LAPSE else "results/07_results_report.txt"
CHART_FILE = "results/real_lapse_contrasts.png" if USE_LAPSE else "results/real_contrasts.png"
NAMES = ["M1", "M2", "M3", "M4"]
TASKS = (1, 3)
TRAIN_ROUNDS = 15
N_PARAMETERS = 4
SCORES = {
    "held_out_ll": ("Held-out, rounds 16-20", 5),
    "early_ll": ("Blocked, rounds 1-5", 5),
    "transfer_ll": ("Transfer to the other task, 20 rounds", 20),
}
CONTRASTS = {
    "Pooled - per-send, constant gain (M1 - M2)": (["M1"], ["M2"]),
    "Pooled - per-send, Bayesian (M3 - M4)": (["M3"], ["M4"]),
    "Pooled - per-send, average": (["M1", "M3"], ["M2", "M4"]),
    "Bayesian - constant, pooled (M3 - M1)": (["M3"], ["M1"]),
    "Bayesian - constant, per-send (M4 - M2)": (["M4"], ["M2"]),
    "Bayesian - constant, average": (["M3", "M4"], ["M1", "M2"]),
}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
TASK_COLOURS = {1: "#2a78d6", 3: "#eb6834"}


def wide_scores(fits, task, column):
    data = fits[fits["task"] == task]
    return data.pivot(index="id", columns="model", values=column)


def score_table(fits, column, n_choices):
    rows = []
    for task in TASKS:
        scores = wide_scores(fits, task, column)
        for name in [*NAMES, "baseline"]:
            gap = paired_bootstrap(scores[name] - scores["baseline"])
            rows.append(
                {
                    "task": task,
                    "model": name,
                    "mean_ll": scores[name].mean(),
                    "median_ll": scores[name].median(),
                    "prob_per_choice": np.exp(scores[name].mean() / n_choices),
                    "vs_baseline": gap["mean"],
                    "low": gap["low"],
                    "high": gap["high"],
                    "very_surprised": (scores[name] < -20).mean(),
                }
            )
    return pd.DataFrame(rows).round(3).to_string(index=False)


def contrast_rows(fits, column, keep=None):
    rows = []
    for task in TASKS:
        scores = wide_scores(fits, task, column)
        if keep is not None:
            scores = scores[scores.index.isin(keep[task])]
        for label, (first, second) in CONTRASTS.items():
            difference = scores[first].mean(axis=1) - scores[second].mean(axis=1)
            rows.append(
                {"task": task, "contrast": label, **paired_bootstrap(difference),
                 "median": difference.median(), "people": len(difference)}
            )
    return pd.DataFrame(rows)


def criteria_table(fits):
    rows = []
    for task in TASKS:
        for name in NAMES:
            data = fits[(fits["task"] == task) & (fits["model"] == name)]
            person = [information_criteria(ll, N_PARAMETERS, TRAIN_ROUNDS) for ll in data["train_ll"]]
            rows.append(
                {
                    "task": task,
                    "model": name,
                    "train_ll": data["train_ll"].sum(),
                    "aic": sum(p["aic"] for p in person) + 2,  # + 2 for the shared g0
                    "bic": sum(p["bic"] for p in person) + np.log(TRAIN_ROUNDS * len(data)),
                }
            )
    return pd.DataFrame(rows).round(1).to_string(index=False)


def parameter_table(fits):
    data = fits[fits["model"] != "baseline"]
    rows = []
    for (task, model), group in data.groupby(["task", "model"]):
        row = {"task": task, "model": model, "g0": group["g0"].iloc[0]}
        for name in ("beta", "rho", "tau", "gain"):
            low, mid, high = group[name].quantile([0.25, 0.5, 0.75])
            row[name] = f"{mid:.3g} ({low:.3g}-{high:.3g})"
            row[f"{name}_bound"] = round(float(group[f"bound_{name}"].astype(bool).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


def choice_shares(predictions, rounds):
    """Average predicted share of each action on held-out rounds, against what people did."""
    held_out = predictions[predictions["round"] > TRAIN_ROUNDS]
    table = held_out.groupby(["task", "model"])[["p0", "p1", "p2", "p3"]].mean()
    actual = rounds[rounds["round"] > TRAIN_ROUNDS]
    shares = actual.groupby("task")["send"].apply(
        lambda s: s.map(action_of).value_counts(normalize=True).reindex(range(4), fill_value=0)
    ).unstack()
    shares.columns = table.columns
    shares.index = pd.MultiIndex.from_product([shares.index, ["observed"]], names=["task", "model"])
    return pd.concat([table, shares]).sort_index().round(3).to_string()


def belief_check(fits, rounds):
    """Each model's belief about return per point after training, against real returns."""
    sequences = {key: group for key, group in rounds.groupby(["id", "task"])}
    models = {model.name: model for model in MODELS}
    rows = []
    for row in fits[fits["model"] != "baseline"].itertuples():
        model = models[row.model]
        learner = Learner(model, Params(row.beta, row.rho, row.tau, row.gain), row.g0)
        history = sequences[(row.id, row.task)]
        for send, returned in zip(history["send"][:TRAIN_ROUNDS], history["returned"][:TRAIN_ROUNDS]):
            learner.update(Visible(int(send), int(returned), 10 - int(send) + int(returned)))
        beliefs = {f"send_{x:g}": learner.predicted_ratio(x) for x in ACTION_SENDS[1:]}
        rows.append({"task": row.task, "model": row.model, **beliefs})
    table = pd.DataFrame(rows).groupby(["task", "model"]).mean()
    sent = rounds[rounds["send"] > 0].assign(
        ratio=lambda d: d["returned"] / d["send"], action=lambda d: d["send"].map(action_of)
    )
    actual = sent.groupby(["task", "action"])["ratio"].mean().unstack()
    actual.columns = table.columns
    actual.index = pd.MultiIndex.from_product([actual.index, ["observed"]], names=["task", "model"])
    return pd.concat([table, actual]).sort_index().round(2).to_string()


def lapse_check(fits, predictions, rounds, lapse):
    """Rescore held-out rounds after blending in a small chance of a random choice.

    A post-hoc check with no refitting. Applied to every model and the baseline alike, it
    limits how badly one overconfident prediction can score.
    """
    actions = rounds.assign(action=rounds["send"].map(action_of))[["id", "task", "round", "action"]]
    held_out = predictions[predictions["round"] > TRAIN_ROUNDS].merge(actions, on=["id", "task", "round"])
    picked = held_out[["p0", "p1", "p2", "p3"]].to_numpy()[np.arange(len(held_out)), held_out["action"]]
    held_out = held_out.assign(ll=np.log((1 - lapse) * picked + lapse / 4))
    scores = held_out.groupby(["task", "id", "model"])["ll"].sum().unstack()

    train = rounds[rounds["round"] <= TRAIN_ROUNDS].assign(action=lambda d: d["send"].map(action_of))
    counts = train.groupby(["task", "id"])["action"].value_counts().unstack(fill_value=0).reindex(columns=range(4), fill_value=0) + 1
    shares = counts.div(counts.sum(axis=1), axis=0)
    test = actions[actions["round"] > TRAIN_ROUNDS]
    probabilities = [shares.loc[(row.task, row.id), row.action] for row in test.itertuples()]
    baseline = test.assign(ll=np.log((1 - lapse) * np.array(probabilities) + lapse / 4))
    scores["baseline"] = baseline.groupby(["task", "id"])["ll"].sum()

    rows = []
    for task in TASKS:
        data = scores.loc[task]
        row = {"lapse": lapse, "task": task}
        for name in NAMES:
            gap = paired_bootstrap(data[name] - data["baseline"])
            row[name] = f"{gap['mean']:+.2f} ({gap['low']:+.2f}, {gap['high']:+.2f})"
        for label, (first, second) in CONTRASTS.items():
            if label.endswith("average"):
                gap = paired_bootstrap(data[first].mean(axis=1) - data[second].mean(axis=1))
                row[label.split(",")[0]] = f"{gap['mean']:+.2f} ({gap['low']:+.2f}, {gap['high']:+.2f})"
        rows.append(row)
    return rows


def raw_comparison(lapse_fits):
    """What changed: fitted settings and scores, raw fits against lapse fits."""
    raw_fits = pd.read_csv("results/real_fits.csv")
    rows = []
    for label, fits in (("raw", raw_fits), ("lapse", lapse_fits)):
        models = fits[fits["model"] != "baseline"]
        for (task, model), group in models.groupby(["task", "model"]):
            no_learning = group["gain"] > 199 if model in ("M3", "M4") else group["gain"] < 0.0051
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "fit": label,
                    "g0": group["g0"].iloc[0],
                    "median_beta": group["beta"].median(),
                    "beta_at_limit": group["bound_beta"].astype(bool).mean(),
                    "no_learning": no_learning.mean(),
                    "held_out_mean": group["held_out_ll"].mean(),
                    "held_out_median": group["held_out_ll"].median(),
                    "worst": group["held_out_ll"].min(),
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "model", "fit"]).round(3).to_string(index=False)


def plot_contrasts(table):
    averages = table[table["contrast"].str.endswith("average")]
    labels = {"Pooled - per-send, average": "Pooled minus per-send", "Bayesian - constant, average": "Bayesian minus constant gain"}
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    for axis, (column, (title, _)) in zip(axes, SCORES.items()):
        data = averages[averages["score"] == column]
        axis.set_facecolor(SURFACE)
        axis.axvline(0, color=INK, linewidth=1)
        for position, label in enumerate(labels):
            for offset, task in ((0.12, 1), (-0.12, 3)):
                row = data[(data["contrast"] == label) & (data["task"] == task)].iloc[0]
                y = position + offset
                axis.plot([row["low"], row["high"]], [y, y], color=TASK_COLOURS[task], linewidth=2)
                axis.scatter(row["mean"], y, s=46, color=TASK_COLOURS[task], zorder=3,
                             edgecolors=SURFACE, linewidths=1.5, label=f"Task {task}" if position == 0 else None)
        axis.set_yticks(range(2), list(labels.values()))
        axis.set_ylim(-0.6, 1.6)
        axis.set_title(title, fontsize=10, color=INK, loc="left")
        axis.set_xlabel("Log-likelihood per person (right favours the first)", fontsize=8, color=MUTED)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
        axis.grid(axis="x", color="#e6e5e1", linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#c3c2b7")
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, frameon=False, fontsize=9, loc="upper right", ncols=2, labelcolor=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(CHART_FILE, dpi=150, facecolor=SURFACE)


def main():
    fits = pd.read_csv(f"{PREFIX}fits.csv")
    predictions = pd.read_csv(f"{PREFIX}predictions.csv")
    grid = pd.read_csv(f"{PREFIX}g0_grid.csv")
    rounds = load_trustor_rounds()
    flags = constant_senders(rounds).reset_index()
    varied = {task: set(flags[(flags["task"] == task) & ~flags["constant"]]["id"]) for task in TASKS}
    lines = []

    def report(title, body):
        lines.extend([f"\n=== {title}", str(body)])

    best = grid.loc[grid.groupby(["task", "model"])["ll"].idxmax()]
    report("Shared g0 (grid runs 1.0 to 2.0)", best.round(1).to_string(index=False))

    tables = []
    for column, (title, n_choices) in SCORES.items():
        report(f"{title}: scores", score_table(fits, column, n_choices))
        contrasts = contrast_rows(fits, column).assign(score=column)
        tables.append(contrasts)
        report(f"{title}: paired contrasts", contrasts.drop(columns="score").round(3).to_string(index=False))

    report(
        "Held-out contrasts without constant senders",
        contrast_rows(fits, "held_out_ll", varied).round(3).to_string(index=False),
    )
    constant = {task: set(flags[(flags["task"] == task) & flags["constant"]]["id"]) for task in TASKS}
    report(
        "Constant senders only: mean held-out LL",
        pd.concat({t: wide_scores(fits, t, "held_out_ll").loc[sorted(constant[t])].mean() for t in TASKS}, axis=1).round(3).to_string(),
    )
    report(
        "Winner shares, held-out, ties split (secondary)",
        pd.concat({t: win_shares(wide_scores(fits, t, "held_out_ll")[NAMES]) for t in TASKS}, axis=1).round(3).to_string(),
    )
    report("Training fit: LL, AIC, BIC", criteria_table(fits))
    report("Parameters: median (IQR), share at a limit. See recovery for what's readable", parameter_table(fits))
    report(
        "rho by profile (median)",
        fits[fits["model"] != "baseline"].groupby(["task", "model", "profile"])["rho"].median().unstack().round(3).to_string(),
    )
    report("Choice shares on held-out rounds", choice_shares(predictions, rounds))
    if USE_LAPSE:
        report("Raw fits against lapse fits", raw_comparison(fits))
    else:
        report(
            "Post-hoc lapse check: held-out gap to baseline, and the two average contrasts",
            pd.DataFrame(
                [row for lapse in (0.02, 0.05, 0.1) for row in lapse_check(fits, predictions, rounds, lapse)]
            ).to_string(index=False),
        )
    report(
        "Fitted learning at its lower limit, or kappa at its upper limit",
        fits[fits["model"] != "baseline"].assign(
            no_learning=lambda d: np.where(d["model"].isin(["M3", "M4"]), d["gain"] > 199, d["gain"] < 0.0051)
        ).groupby(["task", "model"])["no_learning"].mean().unstack().round(3).to_string(),
    )
    report("Belief about return per point after training", belief_check(fits, rounds))

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")
    plot_contrasts(pd.concat(tables))


if __name__ == "__main__":
    main()
