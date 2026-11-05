# Frozen design

Frozen 5 Nov 2026, before any learning model was written or fitted. No further extensions until
the four basic models survive parameter and model recovery. This updates the draft plan in
`Hidden_Partner_Project_Plan.pdf`, which stays as written. Section 7 of that plan calls the partner
scripted: the data show it reacts to the current send and has no memory across rounds.

## Research question

When people interact repeatedly with an uncertain agent, do they learn about the agent or learn
action-specific reward associations? Does how they generalise across actions explain behaviour
better than the rate at which they learn?

H1 and H2 from the draft plan stand. H1's "mostly in early rounds" is tested by blocked
cross-validation.

## Models

|                | Pooled across sends | Separate per send |
|----------------|---------------------|-------------------|
| Constant rate  | M1                  | M2                |
| Shrinking rate | M3                  | M4                |

- **Pooled** means shared beliefs about the partner with a conditional response model `P(r | x, theta)`.
  It does not mean one average return rate applied to every send.
- **Per-send** means independent beliefs about the response at each send level, `P(r | x)`.
- Constant-rate and shrinking-rate models use the same response likelihood wherever possible.
- Update equations and free-parameter counts: to be specified and reviewed before implementation.

## Evaluation

1. **Primary:** Task 1 within task. Fit rounds 1-15, score rounds 16-20.
2. **Secondary:** Task 3 within task, same procedure, run independently.
3. **Exploratory:** fit on Task 1, score on Task 3 without refitting. This is cross-task transfer,
   not prospective prediction, because task order is unconfirmed.
4. **Blocked cross-validation:** early against late predictive performance.

Models are compared with AIC/BIC, held-out log-likelihood, parameter recovery, model recovery and
sensitivity analysis. Tasks 1 and 3 are never concatenated into one 40-round trajectory. Analyses
stay paired: 302 people in two conditions, not 604 people. If four-model recovery is poor, simplify
the comparison and don't force a conclusion.

## Rules

- Strategy and noise columns are for rebuilding and validating the partner. They are never inputs
  to a model of a participant.
- The profile label is for analysis only, never a model input.
- People who never changed their send are reported separately, never silently excluded.
- The Task 1 strategy-mix anomaly is a robustness analysis, not grounds for a new model.

## Feedback prerequisite

Status: supported by behaviour in the data, not yet confirmed from the instructions.
Evidence is in `results/01_feedback_and_order.txt` and `docs/learn.md`, entry 2.

| Question | Status |
|---|---|
| Feedback after every round? | Supported: last round's surprise moves the next send |
| Saw the return, or only the payoff? | Equivalent: `payoff = E - x + r`, and `E` and `x` are known |
| Immediate or delayed? | Supported as immediate: the effect appears at one round's lag |
| Past outcomes stayed visible? | Unknown |
| Task order | Unknown from the data; a related deposit lists trustor, trustee, changed endowments |

## Schedule

| Date | Deliverable |
|---|---|
| Nov 6 | Feedback and task-order investigation; reconstruct environment |
| Nov 7 | Q-learning and constant-rate pooled model |
| Nov 8 | Bayesian pooled and per-send models; recovery |
| Nov 9 | Complete 2x2 recovery and model recovery |
| Nov 10 | Fit Task 1 and Task 3; parameter estimation |
| Nov 11 | Held-out prediction and blocked cross-validation |
| Nov 12 | Cross-task transfer, individual differences, robustness |
| Nov 13 | Report, reproducibility, CV and application |
