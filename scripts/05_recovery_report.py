"""Summarise recovery: can the fitting find the true parameters and the true model?

Run:  .venv/bin/python scripts/05_recovery_report.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS_FILE = "results/05_recovery_report.txt"
NAMES = ["M1", "M2", "M3", "M4"]
PARAMETERS = ["beta", "rho", "tau", "gain"]
LOG_SCALE = {"beta": True, "rho": False, "tau": False}
SURFACE, INK, MUTED, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
TIE_TOLERANCE = 0.01
LONGER_GAMES = (20, 40)
BOTTOM_LABELS = {"M1": "alpha", "M2": "alpha", "M3": "kappa", "M4": "kappa"}


def is_log(parameter, model):
    return model in ("M3", "M4") if parameter == "gain" else LOG_SCALE[parameter]


def parameter_table(own):
    rows = []
    for model in NAMES:
        data = own[own["generator"] == model]
        for parameter in PARAMETERS:
            true, found = data[f"true_{parameter}"], data[parameter]
            if is_log(parameter, model):
                true, found = np.log(true), np.log(found)
            per_repeat = [
                spearmanr(true[data["repeat"] == r], found[data["repeat"] == r]).statistic
                for r in sorted(data["repeat"].unique())
            ]
            rows.append(
                {
                    "model": model,
                    "parameter": BOTTOM_LABELS[model] if parameter == "gain" else parameter,
                    "scale": "log" if is_log(parameter, model) else "raw",
                    "spearman": spearmanr(true, found).statistic,
                    "low": min(per_repeat),
                    "high": max(per_repeat),
                    "bias": (found - true).mean(),
                    "rmse": np.sqrt(((found - true) ** 2).mean()),
                    "at_bound": data[f"bound_{parameter}"].mean(),
                }
            )
    return pd.DataFrame(rows).round(3)


def credit(fits):
    """Share of each agent's win given to each model, splitting near-ties equally.

    Learning can vanish in a fit, and then the models tie. Without splitting, ties would
    all go to whichever model happens to be listed first.
    """
    likelihoods = fits.pivot_table(index=["repeat", "generator", "agent"], columns="fitted", values="ll")
    winners = likelihoods.ge(likelihoods.max(axis=1) - TIE_TOLERANCE, axis=0).astype(float)
    return winners.div(winners.sum(axis=1), axis=0)[NAMES]


def confusion(shares):
    return shares.groupby(level="generator").mean().reindex(NAMES)


def dimension_accuracy(shares):
    """Credit given to the right level of each dimension. Generators are balanced."""
    levels = {"generalisation": (["M1", "M3"], ["M2", "M4"]), "dynamics": (["M1", "M2"], ["M3", "M4"])}
    generators = shares.index.get_level_values("generator")
    rows = {}
    for label, (first, second) in levels.items():
        right = np.where(generators.isin(first), shares[first].sum(axis=1), shares[second].sum(axis=1))
        by_repeat = pd.Series(right, index=shares.index).groupby(level="repeat").mean()
        rows[label] = {
            "accuracy": round(float(right.mean()), 3),
            "low": round(float(by_repeat.min()), 3),
            "high": round(float(by_repeat.max()), 3),
        }
    return rows


def own_model_credit(shares):
    generators = shares.index.get_level_values("generator")
    own = pd.Series(
        [shares.iloc[i][name] for i, name in enumerate(generators)], index=shares.index
    )
    return own.groupby(level="generator").mean().round(3).to_dict()


def by_learning_strength(fits, shares):
    """Credit to the true model for fast and slow learners, split at each gain's median."""
    keys = ["repeat", "generator", "agent"]
    gains = fits.drop_duplicates(keys).set_index(keys)["true_gain"].reindex(shares.index)
    generators = gains.index.get_level_values("generator")
    median = gains.groupby(level="generator").transform("median")
    # A small kappa and a large alpha both mean fast learning.
    fast = np.where(generators.isin(["M3", "M4"]), gains < median, gains > median)
    own = pd.Series([shares.iloc[i][name] for i, name in enumerate(generators)], index=shares.index)
    return own.groupby(np.where(fast, "fast half", "slow half")).mean().round(3).to_dict()


def by_game_length(base):
    """Does more data help? Same recovery with longer games."""
    rows = []
    for rounds, fits in [(15, base), *[(n, pd.read_csv(f"results/recovery_fits_{n}.csv")) for n in LONGER_GAMES]]:
        shares = credit(fits)
        own = fits[fits["generator"] == fits["fitted"]]
        dimensions = dimension_accuracy(shares)
        matrix = confusion(shares).to_numpy()
        rows.append(
            {
                "rounds": rounds,
                "four_way": round(float(np.diag(matrix).mean()), 3),
                "generalisation": dimensions["generalisation"]["accuracy"],
                "dynamics": dimensions["dynamics"]["accuracy"],
                "bayes_to_own": round(float(np.diag(matrix)[2:].mean()), 3),
                "constant_to_own": round(float(np.diag(matrix)[:2].mean()), 3),
                "gain_spearman": round(float(np.mean([
                    spearmanr(own.loc[own["generator"] == m, "true_gain"], own.loc[own["generator"] == m, "gain"]).statistic
                    for m in NAMES
                ])), 3),
                "tau_spearman": round(float(spearmanr(own["true_tau"], own["tau"]).statistic), 3),
                "beta_at_bound": round(float(own["bound_beta"].mean()), 3),
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


def g0_summary(grid):
    totals = grid.groupby(["generator", "true_g0", "g0"])["ll"].sum().reset_index()
    best = totals.loc[totals.groupby(["generator", "true_g0"])["ll"].idxmax()]
    return best.rename(columns={"g0": "best_g0", "ll": "total_ll"}).round(2).to_string(index=False)


def style(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.grid(color="#e6e5e1", linewidth=0.6)
    axis.set_axisbelow(True)


def plot_parameters(own, table):
    figure, axes = plt.subplots(4, 4, figsize=(12, 11.5), facecolor=SURFACE)
    for row, model in enumerate(NAMES):
        data = own[own["generator"] == model]
        for column, parameter in enumerate(PARAMETERS):
            axis = axes[row, column]
            style(axis)
            log = is_log(parameter, model)
            true, found = data[f"true_{parameter}"], data[parameter]
            axis.scatter(true, found, s=11, color=BLUE, alpha=0.45, linewidths=0)
            low, high = true.min(), true.max()
            axis.plot([low, high], [low, high], color=INK, linewidth=1, linestyle=(0, (4, 3)))
            if log:
                axis.set_xscale("log")
                axis.set_yscale("log")
                ticks = [0.5, 1, 2, 5] if parameter == "beta" else [1, 3, 10]
                axis.set_xticks(ticks, [str(t) for t in ticks])
                axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            label = BOTTOM_LABELS[model] if parameter == "gain" else parameter
            score = table[(table["model"] == model) & (table["parameter"] == label)]["spearman"].iloc[0]
            axis.set_title(f"{model}: {label}, Spearman {score:.2f}", fontsize=10, color=INK, loc="left")
            if row == 3:
                axis.set_xlabel("True value", fontsize=9, color=MUTED)
            if column == 0:
                axis.set_ylabel("Recovered value", fontsize=9, color=MUTED)
    figure.suptitle(
        "Parameter recovery from 15 simulated rounds. Dashed line is perfect recovery.",
        fontsize=13, color=INK, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig("results/recovery_parameters.png", dpi=150, facecolor=SURFACE)


def plot_confusion(matrix):
    figure, axis = plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    colours = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#9ec5f4", "#3987e5", "#104281"]
    )
    axis.imshow(matrix.to_numpy(), cmap=colours, vmin=0, vmax=1)
    axis.set_xticks(range(4), NAMES)
    axis.set_yticks(range(4), NAMES)
    axis.tick_params(colors=INK, length=0, labelsize=11)
    for side in axis.spines.values():
        side.set_visible(False)
    for i in range(4):
        for j in range(4):
            value = matrix.iloc[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=12,
                      color="#fcfcfb" if value > 0.5 else INK)
    axis.set_xlabel("Best-fitting model", color=MUTED, fontsize=10)
    axis.set_ylabel("True model", color=MUTED, fontsize=10)
    axis.set_title("Which model wins for each true model", color=INK, fontsize=12, loc="left")
    figure.tight_layout()
    figure.savefig("results/recovery_confusion.png", dpi=150, facecolor=SURFACE)


def main():
    fits = pd.read_csv("results/recovery_fits.csv")
    grid = pd.read_csv("results/recovery_g0.csv")
    own = fits[fits["generator"] == fits["fitted"]]
    table = parameter_table(own)
    shares = credit(fits)
    matrix = confusion(shares)
    lines = []

    def report(title, body):
        lines.extend([f"\n=== {title}", str(body)])

    report("Do simulated agents look like people? (real Task 1: mean send 5.5, repeat rate 0.72)",
           own.groupby("generator")[["mean_send", "repeat_rate"]].mean().round(2).to_string())
    report("Parameter recovery, own model", table.to_string(index=False))
    report("Confusion: rows true, columns best fit", matrix.round(3).to_string())
    report("Accuracy on each dimension (chance 0.5)", dimension_accuracy(shares))
    report("Credit to the true model by learning speed", by_learning_strength(fits, shares))
    report("Near-tied agents", round(float((shares.max(axis=1) < 1).mean()), 3))
    report("Same recovery with longer games", by_game_length(fits))
    report("Shared g0: best grid value by true value", g0_summary(grid))

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")
    plot_parameters(own, table)
    plot_confusion(matrix)


if __name__ == "__main__":
    main()
