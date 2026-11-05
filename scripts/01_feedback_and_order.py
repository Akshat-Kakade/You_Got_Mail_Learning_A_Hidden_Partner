"""Did participants see each outcome before their next decision, and in what task order?

The deposit doesn't document feedback or task order, so this uses the AI's own
randomness as a natural experiment. Defections are drawn independently of everything
the participant does, so any later change in sending that follows one is caused by it.

Run:  .venv/bin/python scripts/01_feedback_and_order.py
"""

import numpy as np
import pandas as pd

from hidden_partner.data import load_trustor_rounds

RESULTS_FILE = "results/01_feedback_and_order.txt"


def within_person_ols(frame, outcome, regressors, cluster):
    """OLS with person fixed effects and person-clustered standard errors.

    Returns a table of coefficient, standard error and t per regressor.
    """
    columns = [outcome, *regressors]
    data = frame.dropna(subset=columns)
    demeaned = data[columns] - data.groupby(cluster)[columns].transform("mean")
    y = demeaned[outcome].to_numpy(float)
    X = demeaned[regressors].to_numpy(float)

    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta

    scores = pd.DataFrame(X * residual[:, None]).groupby(data[cluster].to_numpy()).sum()
    meat = scores.to_numpy().T @ scores.to_numpy()
    n_clusters = len(scores)
    correction = n_clusters / (n_clusters - 1)
    se = np.sqrt(np.diag(bread @ meat @ bread) * correction)
    return pd.DataFrame(
        {"coef": beta, "se": se, "t": beta / se}, index=regressors
    ).round(3), len(data), n_clusters


def add_timing_columns(rounds):
    """Add each round's past, present and future defections, per person and task.

    Uses the raw defection draw, which is random. A flag like "defected and the
    person sent something" would not be, because sending predicts later sending.
    """
    frame = rounds.copy()
    frame["defect_now"] = frame["defected"].astype(float)  # placebo: comes after the choice
    grouped = frame.groupby(["id", "task"])
    for lag in (1, 2, 3):
        frame[f"defect_lag{lag}"] = grouped["defect_now"].shift(lag)
    frame["defect_next"] = grouped["defect_now"].shift(-1)  # placebo
    frame["send_lag1"] = grouped["send"].shift(1)
    frame["strategy_lag1"] = grouped["strategy"].shift(1)
    return frame


def return_surprise(frame):
    """Actual return minus the average return for that profile and send.

    Computed on non-defection rounds, so it isolates variation in the amount, not
    whether the partner defected. Tests whether people saw the amount itself.
    """
    paid = frame[frame["strategy"] > 0]
    expected = paid.groupby(["task", "profile", "send"])["returned"].transform("mean")
    surprise = (paid["returned"] - expected).rename("surprise")
    out = frame.join(surprise)
    out["surprise_lag1"] = out.groupby(["id", "task"])["surprise"].shift(1)
    return out


def add_news(frame):
    """Add `news`: this round's return minus the average for that task, profile and send.

    Defined on every row. Given the send, it depends only on the AI's random draw, so
    it needs no person controls. Keeping rows with send = 0 matters: dropping them would
    discard people who stopped sending after bad news, hiding the reaction being measured.
    """
    expected = frame.groupby(["task", "profile", "send"])["returned"].transform("mean")
    out = frame.assign(news=frame["returned"] - expected)
    grouped = out.groupby(["id", "task"])
    for lag in (1, 2, 3):
        out[f"news_lag{lag}"] = grouped["news"].shift(lag)
    out["news_next"] = grouped["news"].shift(-1)  # placebo: not yet happened
    out["change"] = out["send"] - out["send_lag1"]
    return out


def clustered_ols(frame, outcome, regressors, cluster):
    """OLS with an intercept and person-clustered standard errors, no fixed effects."""
    data = frame.dropna(subset=[outcome, *regressors])
    y = data[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(data)), data[regressors].to_numpy(float)])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    scores = pd.DataFrame(X * (y - X @ beta)[:, None]).groupby(data[cluster].to_numpy()).sum()
    meat = scores.to_numpy().T @ scores.to_numpy()
    n_clusters = len(scores)
    se = np.sqrt(np.diag(bread @ meat @ bread) * n_clusters / (n_clusters - 1))
    table = pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=["const", *regressors])
    return table.drop(index="const").round(4), len(data), n_clusters


def order_test(rounds):
    """Compare Task 1 experience -> Task 3 start, against Task 3 experience -> Task 1 start.

    Random defections can only change an opening choice if they happened earlier.
    Round 1 has no defections, so round-1 sends in the other task are pre-experience.
    """
    wide = rounds[rounds["round"].between(2, 20)].pivot_table(
        index="id", columns="task", values="defected", aggfunc="mean"
    )
    wide.columns = [f"defect_rate_t{task}" for task in wide.columns]
    openers = (
        rounds[rounds["round"] <= 3]
        .pivot_table(index="id", columns="task", values="send", aggfunc="mean")
        .rename(columns={1: "open_t1", 3: "open_t3"})
    )
    profile = rounds.groupby("id")["profile"].first()
    data = wide.join(openers).join(profile)
    rows = []
    for name, outcome, predictor in (
        ("Task 1 defections -> Task 3 opening", "open_t3", "defect_rate_t1"),
        ("Task 3 defections -> Task 1 opening", "open_t1", "defect_rate_t3"),
    ):
        design = pd.get_dummies(data["profile"], dtype=float)
        design[predictor] = data[predictor]
        X = design.to_numpy()
        y = data[outcome].to_numpy(float)
        bread = np.linalg.inv(X.T @ X)
        beta = bread @ X.T @ y
        residual = y - X @ beta
        leverage_free = (X * residual[:, None]).T @ (X * residual[:, None])
        se = np.sqrt(np.diag(bread @ leverage_free @ bread))
        rows.append({"test": name, "coef": beta[-1], "se": se[-1], "t": beta[-1] / se[-1]})
    return pd.DataFrame(rows).round(3)


def interleaving_test(frame):
    """If tasks alternated within a round, a defection in one would move the other's next send."""
    wide = frame.pivot_table(index=["id", "round"], columns="task", values=["send", "defect_lag1"])
    wide.columns = [f"{name}_t{task}" for name, task in wide.columns]
    wide = wide.reset_index()
    wide = wide[wide["round"] >= 3]
    results = {}
    for outcome, own, other in (
        ("send_t1", "defect_lag1_t1", "defect_lag1_t3"),
        ("send_t3", "defect_lag1_t3", "defect_lag1_t1"),
    ):
        results[outcome] = within_person_ols(wide, outcome, [own, other], "id")[0]
    return results


def main():
    rounds = load_trustor_rounds()
    frame = add_news(return_surprise(add_timing_columns(rounds)))
    lines = []

    def report(title, body):
        lines.extend([f"\n=== {title}", str(body)])

    for task in (1, 3):
        # Round 5 onward, so all three lags fall in rounds where defection is possible.
        data = frame[(frame["task"] == task) & (frame["round"] >= 5)]
        sent_before = data[data["send_lag1"] > 0]
        sent_nothing = data[data["send_lag1"] == 0]

        regressors = ["defect_lag1", "defect_lag2", "defect_lag3", "defect_now"]
        table, n, people = within_person_ols(sent_before, "send", regressors, "id")
        report(f"Task {task}: send after defections, sent>0 last round (n={n}, people={people})", table)

        table, n, _ = within_person_ols(
            sent_before[sent_before["round"] < 20], "send", ["defect_lag1", "defect_next"], "id"
        )
        report(f"Task {task}: placebo, a defection that has not happened yet (n={n})", table)

        table, n, people = within_person_ols(sent_nothing, "send", ["defect_lag1"], "id")
        report(f"Task {task}: sent 0 last round, so a defection was unnoticeable (n={n}, people={people})", table)

        paid = sent_before[sent_before["strategy_lag1"] > 0]
        table, n, _ = within_person_ols(paid, "send", ["surprise_lag1"], "id")
        report(f"Task {task}: size of last return, no defection (n={n})", table)

        change = (
            sent_before.assign(change=lambda d: d["send"] - d["send_lag1"])
            .groupby("defect_lag1")["change"]
            .agg(["mean", "count"])
            .round(3)
        )
        report(f"Task {task}: mean change in send, after no defection (0) vs defection (1)", change)

    news_terms = ["news_lag1", "news_lag2", "news_lag3", "news", "news_next"]
    late = frame[frame["round"] >= 5]
    for label, data in (("Task 1", late[late["task"] == 1]), ("Task 3", late[late["task"] == 3]), ("Pooled", late)):
        table, n, people = clustered_ols(data, "change", news_terms, "id")
        report(f"{label}: change in send on past news; `news` and `news_next` are placebos (n={n}, people={people})", table)

    report("Task order: does one task's randomness reach the other's opening?", order_test(rounds))
    for outcome, table in interleaving_test(frame).items():
        report(f"Interleaving check, outcome {outcome}", table)

    text = "\n".join(lines)
    print(text)
    with open(RESULTS_FILE, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
