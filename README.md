# Learning a Hidden Partner

How do people learn about a partner they can't see? This project fits four computational models to a public
repeated trust game, where 302 people played 20 rounds against an AI partner whose type was hidden from them.

The models separate two questions: does a learner share evidence across different actions, and does its
learning slow down as evidence builds? I built this to learn computational modelling and RL. It's a learning
project, not a peer-reviewed study.

**Start here:** [`docs/report.md`](docs/report.md) has the findings. [`docs/learn.md`](docs/learn.md) explains
every concept used, with intuitions, metaphors and the maths.

## The models

| | Pooled across sends | Separate per send |
|---|---|---|
| **Constant gain** | M1 | M2 (Q-learning) |
| **Bayesian** | M3 (Bayesian linear regression) | M4 |

All four share one value function, one softmax choice rule with stickiness, and a fixed 5% lapse. Full equations
are in [`docs/model_spec.md`](docs/model_spec.md).

## Layout

```
src/hidden_partner/   data.py, environment.py, models.py, fitting.py, simulate.py, evaluation.py
scripts/              numbered analysis steps, and run_all.sh
tests/                unit tests, including checks against closed-form answers
results/              reports, fits and charts
docs/                 report, concept notes, model spec, frozen design, original plan
```

## Running it

Python 3.11 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
scripts/run_all.sh     # rebuilds everything; uses all cores for about an hour
```

## Data

The dataset isn't included. Download the [Mendeley deposit](https://data.mendeley.com/datasets/58mxtr44gm) (CC BY 4.0)
and put its files in `data/Exploring trust in Artificial Intelligence (AI) sy/`.

Jayasekara, D. N., Deng, R., Ho, J. Q., & Prissé, B. (2026). Exploring trust in Artificial Intelligence (AI) systems:
Insights from a repeated trust game. *Journal of Behavioral and Experimental Economics, 125*, 102658.
