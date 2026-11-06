# Frozen design

This is a learning project and a CaCTüS application piece, not a research submission, so checks stay
light. Frozen 5 Nov 2026, before any learning model was written or fitted. No further extensions until
the four basic models have been through parameter and model recovery. This updates the draft plan in
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
- Update equations and parameter counts: drafted in `docs/model_spec.md`, awaiting review.

## Evaluation

1. **Primary:** Task 1 within task. Fit rounds 1-15, score rounds 16-20.
2. **Secondary:** Task 3 within task, same procedure, run independently.
3. **Exploratory:** fit on Task 1, score on Task 3 without refitting. This is cross-task transfer,
   not prospective prediction, because task order is unconfirmed.
4. **Blocked cross-validation:** early against late predictive performance.

Models are compared with AIC/BIC, held-out log-likelihood, parameter recovery, model recovery and
sensitivity analysis. Tasks 1 and 3 are never concatenated into one 40-round trajectory. Analyses
stay paired: 302 people in two conditions, not 604 people. The four models are a structured comparison,
not a clean factorial, because the two dimensions interact. If recovery is poor, find out why and report
what the data can't separate.

## Rules

- Strategy and noise columns are for rebuilding and validating the partner. They are never inputs
  to a model of a participant.
- The profile label is for analysis only, never a model input.
- People who never changed their send are reported separately, never silently excluded.
- The Task 1 strategy-mix anomaly is a robustness analysis, not grounds for a new model.

## Feedback prerequisite

Cleared on 5 Nov 2026 as sufficient to proceed. The methods claim, word for word:

> Behavioural analyses support the assumption that participants had access to information about
> the preceding round's outcome before making their next decision. The effect is consistent across
> both trustor tasks, while a future-outcome placebo shows no corresponding association.

Never write that the instructions confirmed feedback. This is behavioural evidence of outcome
exposure, not documentary evidence of what was displayed. Evidence is in
`results/01_feedback_and_order.txt` and `docs/learn.md`, entry 2.

| Question | Status |
|---|---|
| Information about the last outcome before the next choice? | Supported by behaviour |
| Saw the return, the payoff, or both? | Unknown; kept explicit in the observation model |
| Past outcomes stayed visible? | Unknown; a limitation, not a blocker. The models need sequential updating, not retrieval |
| Task order | Unknown from the data; a related deposit lists trustor, trustee, changed endowments |

## Main analysis

Frozen 6 Nov 2026 after both fits were seen: the fit with a fixed 5% lapse inside the likelihood
(`results/real_lapse_*`, `results/07_results_lapse.txt`) is the main analysis. The lapse was added after the raw
results, because of overconfidence, so it is a disclosed change, not part of the original plan. The raw fits
(`results/real_*`, `results/07_results_report.txt`) stay as the comparison and are reported alongside.

## Interpretation limits

- The simulator validation shows the AI's conditional response distribution is reproduced. It says
  nothing about human adaptation to that distribution.
- Task 3 rules on Task 1 scored p = 0.054: a borderline result, not a clean pass.
- A model winning on held-out likelihood is not evidence for its mechanism unless recovery supports it.

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
