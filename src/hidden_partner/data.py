"""Load the trust-game deposit into one tidy table of trustor rounds.

One row is one participant in one round of one trustor task (Task 1 or Task 3).

Columns are split by who could see them. `VISIBLE_COLUMNS` is what a participant
could know. `HIDDEN_COLUMNS` describes the AI's internal draw and exists only to
reconstruct and validate the partner: it must never reach a model of a participant.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "Exploring trust in Artificial Intelligence (AI) sy"
    / "Data_cleaned.dta"
)

TRUSTOR_TASKS = (1, 3)
PROFILES = ("Social_AI", "Exploratory_AI", "Competitive_AI")
MULTIPLIER = 3
DEFECT_CODE = -1  # value of ai_strategy_task* on rounds where the AI returned nothing

VISIBLE_COLUMNS = ("id", "task", "round", "endowment", "send", "returned", "payoff")
HIDDEN_COLUMNS = ("strategy", "noise", "defected")
LABEL_COLUMNS = ("profile",)  # known to us for analysis, never a model input


def load_trustor_rounds(path: Path = DATA_FILE) -> pd.DataFrame:
    """Return Task 1 and Task 3 stacked, sorted by participant, task and round."""
    raw = pd.read_stata(path, convert_categoricals=False)
    frames = [_one_task(raw, task) for task in TRUSTOR_TASKS]
    rounds = pd.concat(frames, ignore_index=True)
    return rounds.sort_values(["id", "task", "round"], ignore_index=True)


def _one_task(raw: pd.DataFrame, task: int) -> pd.DataFrame:
    strategy = pd.to_numeric(raw[f"ai_strategy_task{task}"]).astype(int)
    return pd.DataFrame(
        {
            "id": raw["participant_code"].astype(str),
            "profile": raw["treatment"].astype(str),
            "task": task,
            "round": raw["round_number"].astype(int),
            "endowment": raw[f"player_a_endowment_task{task}"].astype(int),
            "ai_endowment": raw[f"ai_endowment_task{task}"].astype(int),
            "send": raw[f"player_a_sent_amount_task{task}"].astype(int),
            "returned": raw[f"ai_sent_back_task{task}"].astype(int),
            "payoff": raw[f"player_a_payoff_task{task}"].astype(int),
            "strategy": strategy,
            "noise": raw[f"ai_variance_task{task}"].astype(int),
            "defected": strategy == DEFECT_CODE,
        }
    )


def constant_senders(rounds: pd.DataFrame) -> pd.Series:
    """Flag (id, task) pairs where the send never changed across rounds.

    These sequences say nothing about adaptation within a task, but they do carry
    information about preferences, so they get reported separately, not dropped.
    """
    return rounds.groupby(["id", "task"])["send"].nunique().eq(1).rename("constant")
