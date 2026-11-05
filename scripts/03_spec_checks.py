"""Facts the model specification depends on. Nothing here fits a learning model.

Checks the shape of return noise, whether a two-number curve can describe the partner's
response, what choice patterns need explaining, and how much evidence 15 rounds contain.

Run:  .venv/bin/python scripts/03_spec_checks.py
"""

import numpy as np
import pandas as pd

from hidden_partner.data import PROFILES, load_trustor_rounds
from hidden_partner.environment import estimate_rules, return_distribution

RESULTS_FILE = "results/03_spec_checks.txt"
BIN_EDGES = [-1, 0, 4, 9, 10]
BIN_LABELS = ["0", "1-4", "5-9", "10"]
TRAIN_ROUNDS = 15
N_SHUFFLES = 2000


def true_ratio_curve(rules, profile):
    """Expected return per point sent, for sends 1 to 10, under the rebuilt partner."""
    sends = np.arange(1, 11)
    means = []
    for send in sends:
        probs = return_distribution(rules, profile, int(send), 0)
        means.append(probs @ np.arange(len(probs)) / send)
    return sends, np.array(means)


def r_squared(target, fitted):
    return 1 - ((target - fitted) ** 2).sum() / ((target - target.mean()) ** 2).sum()


def response_shape_fits(rules):
    """How well a flat line, a sloped line and a three-step function describe each curve."""
    rows = []
    for profile in PROFILES:
        sends, ratio = true_ratio_curve(rules, profile)
        centred = (sends - 5.5) / 4.5
        line = np.polyval(np.polyfit(centred, ratio, 1), centred)
        bins = pd.cut(sends, [0, 4, 9, 10], labels=False)
        steps = pd.Series(ratio).groupby(bins).transform("mean").to_numpy()
        rows.append(
            {
                "profile": profile,
                "ratio_1": ratio[0],
                "ratio_3": ratio[2],
                "ratio_5": ratio[4],
                "ratio_10": ratio[9],
                "r2_sloped_line": r_squared(ratio, line),
                "r2_three_steps": r_squared(ratio, steps),
            }
        )
    return pd.DataFrame(rows).round(3).to_string(index=False)


def noise_scale(rounds):
    """Spread of the raw return and of the return per point sent, by send."""
    sent = rounds[rounds["send"] > 0].assign(ratio=lambda d: d["returned"] / d["send"])
    cells = sent.groupby(["profile", "send"])
    table = pd.DataFrame({"sd_return": cells["returned"].std(), "sd_ratio": cells["ratio"].std()})
    return table.groupby("send").mean().round(2).T.to_string()


def repeat_rate(bins, rng):
    """Share of rounds repeating last round's bin, against the same choices shuffled."""
    real = (bins[:, 1:] == bins[:, :-1]).mean()
    shuffled = []
    for _ in range(N_SHUFFLES):
        mixed = rng.permuted(bins, axis=1)
        shuffled.append((mixed[:, 1:] == mixed[:, :-1]).mean())
    return real, float(np.mean(shuffled)), float(np.quantile(shuffled, 0.975))


def gain_paths(train, kappa, slope_scale):
    """Share of a surprise absorbed each round, on people's real send sequences.

    Pure arithmetic on the proposed Bayesian recursion: uses sends only, no returns, no
    fitted values. `gain` is the fraction of the error taken up at the chosen send.
    """
    pooled, separate = {}, {}
    for _, person in train.groupby(["id", "task"]):
        cov = np.eye(2) / kappa
        visits = np.zeros(3)
        for round_number, send in zip(person["round"], person["send"]):
            if send == 0:
                continue
            feature = np.array([1.0, slope_scale * (send - 5.5) / 4.5])
            spread = feature @ cov @ feature
            pooled.setdefault(round_number, []).append(spread / (spread + 1))
            cov = cov - np.outer(cov @ feature, cov @ feature) / (spread + 1)
            bin_index = 0 if send <= 4 else 1 if send <= 9 else 2
            visits[bin_index] += 1
            separate.setdefault(round_number, []).append(1 / (kappa + visits[bin_index]))
    picks = (2, 5, 10, 15)
    return {
        "pooled": [round(float(np.mean(pooled[r])), 3) for r in picks],
        "per_send": [round(float(np.mean(separate[r])), 3) for r in picks],
    }


def main():
    rng = np.random.default_rng(20261106)
    rounds = load_trustor_rounds()
    rules = estimate_rules(rounds)
    rounds = rounds.assign(bin=pd.cut(rounds["send"], BIN_EDGES, labels=False))
    lines = []

    def report(title, body):
        lines.extend([f"\n=== {title}", str(body)])

    report("Spread by send (averaged over profiles)", noise_scale(rounds))
    report("Partner's response curve: return per point sent", response_shape_fits(rules))

    shares = rounds.groupby("task")["bin"].value_counts(normalize=True).unstack().round(3)
    shares.columns = BIN_LABELS
    report("Choice share by bin", shares.to_string())

    for task in (1, 3):
        data = rounds[rounds["task"] == task]
        counts = data.pivot_table(index="id", columns="bin", values="round", aggfunc="count", fill_value=0)
        modal = counts.to_numpy().argmax(axis=1)
        report(
            f"Task {task}: favourite bin across people",
            dict(zip(BIN_LABELS, np.round(np.bincount(modal, minlength=4) / len(modal), 3).tolist())),
        )
        sequences = data.pivot(index="id", columns="round", values="bin").to_numpy()
        real, shuffled, upper = repeat_rate(sequences, rng)
        report(
            f"Task {task}: repeat last round's bin",
            f"real {real:.3f}, shuffled within person {shuffled:.3f} (97.5% {upper:.3f})",
        )

    train = rounds[rounds["round"] <= TRAIN_ROUNDS]
    per_person = train.groupby(["task", "id"])["bin"].agg(lambda b: b[b > 0].nunique())
    report(
        "Rounds 1-15: number of non-zero bins used, share of people",
        per_person.groupby("task").value_counts(normalize=True).unstack().round(3).to_string(),
    )
    visits = train.groupby(["task", "id", "bin"]).size().unstack(fill_value=0)
    report("Rounds 1-15: mean visits per bin", visits.groupby("task").mean().round(2).to_string())
    observed = train[train["send"] > 0].groupby(["task", "id"]).size()
    report("Rounds 1-15: informative observations per person (median)", observed.groupby("task").median().to_dict())

    for kappa in (1, 5):
        for slope_scale in (np.sqrt(0.5),):
            report(f"Bayesian gain at rounds 2, 5, 10, 15; kappa={kappa}", gain_paths(train, kappa, slope_scale))

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
