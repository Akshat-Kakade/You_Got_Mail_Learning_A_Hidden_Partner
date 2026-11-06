# Model specification (draft for review)

Status: revised after review, not implemented, not fitted. This is a learning project, so checks are
kept light. Numbers quoted here come from `scripts/03_spec_checks.py`
and `results/03_spec_checks.txt`, which fit no learning model.

## 1. Shared foundation

These parts are identical in all four models.

**Observation model.** After a round with `x_t > 0` the learner has

    y_t = r_t / x_t        (return per point sent)

We don't know whether people saw the return, the payoff, or both. It's kept explicit as assumption
A1: they could recover `r_t`, directly or from `r_t = payoff_t - E + x_t`, using the known `E` and `x_t`.

**Actions.** Four send levels, as in the draft plan:

| Action a | Sends | Value evaluated at x̄_a |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1-4 | 2.5 |
| 2 | 5-9 | 7 |
| 3 | 10 | 10 |

The likelihood is over actions. Updates use the exact send `x_t`, not `x̄_a`.

**Assumed response model.** Every model predicts the return per point with

    mu_t(x) = psi(x)' m_t

and the implied conditional distribution is

    y | x, w ~ N( psi(x)' w, sigma^2 )    so    r | x, w ~ N( x psi(x)' w, x^2 sigma^2 )

This conditions on `x`. The Normal is the learner's assumed model, not the true partner, which has a
10% spike at 0. "Bayesian" below means exact under this assumed model. It does not mean ideal observer.

Checks behind it: the spread of `y` is about constant (0.51 to 0.58) for sends of 4 and up, while the
spread of `r` grows with the send. It's larger at sends 1 to 3 (1.0, 0.8, 0.66), which is a known
misfit of the assumption.

**Valuation.** Utility is exponential:

    u(z) = -exp(-lambda z),   lambda > 0,   defined for all real z

Power and log utility need `z > 0`, but the assumed Normal payoff can go negative, so they don't fit.
With payoff `z = E - x + x y` and `y ~ N(mu, sigma^2)`, the certainty equivalent is exact. Setting
`rho = lambda sigma^2` and dropping `E`, which cancels in the softmax:

    U_t(a) = x̄_a ( mu_t(x̄_a) - 1 ) - (rho / 2) x̄_a^2

`U_t(0) = 0` always. `rho` is a descriptive risk penalty: it mixes risk aversion with the noise the
learner assumes, and it would also absorb anything else that favours middle sends, such as fairness.
Don't read it as pure risk aversion.

All four models value actions at the current mean `mu_t`. For M3 and M4 that's the posterior mean, so
**uncertainty sets how fast they learn, not what they choose**. No model has an exploration bonus: all
are myopic. Option: full posterior predictive expected utility is a one-term change for M3 and M4,
replacing `x̄^2` with `x̄^2 (1 + psi' P psi)`. It's left out of the primary models because it would make
`kappa` also shift the opening choice, tangling it with `rho`.

**Choice rule.**

    P(a_t = a) proportional to exp( beta U_(t-1)(a) + tau 1[a = a_(t-1)] )

In the main analysis the final probabilities also include a fixed 5% lapse, `(1 - 0.05) P(a) + 0.05 / 4`. It was added
after the raw fits, as recorded in `docs/design.md`.

Round 1 has no previous action, so the `tau` term is simply left out. `tau` describes persistence in
choices. It doesn't prove a separate habit mechanism.

**Send = 0.** There is no `y_t`, so no update in any model, and nothing decays. Rounds still pass.
In Task 3 a send of 0 can return up to 3 (expected 0.5 to 0.7). All four ignore that, equally.

**Task boundaries.** State resets to the prior at the start of each task. Tasks are fitted separately.

## 2. The two dimensions

**Generalisation** sets `psi`, and nothing else.

    Pooled:    psi(x) = ( 1, s c(x) ),  c(x) = (x - 5.5) / 4.5,  s^2 = 0.5 (fixed)
    Per-send:  psi(x) = one-hot of the bin of x, over bins {1-4}, {5-9}, {10}

Pooled weights are `w = (generosity at a middling send, responsiveness to send size)`. Learning from one
send therefore changes predictions at every send. Per-send weights are independent. Per-send means
per-bin here, so sends inside a bin still share a belief.

**Learning dynamics** sets the update, and nothing else. With `e_t = y_t - psi_t' m_(t-1)`:

    Constant gain:   m_t = m_(t-1) + alpha  psi_t e_t / (psi_t' psi_t)

    Bayesian:        S_t = psi_t' P_(t-1) psi_t + 1
                     K_t = P_(t-1) psi_t / S_t
                     m_t = m_(t-1) + K_t e_t
                     P_t = P_(t-1) - K_t psi_t' P_(t-1)

Both are one equation, where `gamma_t` is the share of the error absorbed at the chosen send:

    m_t = m_(t-1) + gamma_t  [ P psi_t / (psi_t' P psi_t) ]  e_t

Constant gain has `gamma_t = alpha` and `P = I`, fixed. Bayesian has `gamma_t = psi' P psi / (psi' P psi + 1)`
with `P` updated.

The Bayesian recursion is the exact conjugate posterior for the likelihood above with prior
`w ~ N(m_0, (sigma^2 / kappa) I)`, so `P` is in units of `sigma^2`. `sigma` needs no separate estimate,
because only `kappa` affects the means. It is Bayesian because it follows from that prior and
likelihood, not because its step shrinks.

## 3. The four models

| | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| Name | Constant gain, pooled | Constant gain, per-send | Bayesian, pooled | Bayesian, per-send |
| Update | `m += alpha psi e / psi'psi` | `w_a += alpha (y - w_a)` | `S, K, m, P` recursion | `w_a += (y - w_a) / (kappa + n_a)` |
| Representation | Point estimate `(g, b)` | Point estimate per bin | Normal posterior over `(g, b)` | Independent Normal posterior per bin |
| Choice rule | Shared | Shared | Shared | Shared |
| Initial state | `m_0 = (g0, 0)` | `w_a = g0` | `m_0 = (g0, 0)`, `P_0 = I / kappa` | `w_a = g0`, `n_a = 0` |
| Prior | None; `m_0` is a starting estimate | None; same | `N(m_0, (sigma^2/kappa) I)` | `N(g0, sigma^2/kappa)` per bin |
| Send = 0 | No update | No update | No update | No update |
| Retained | 2 numbers + last action | 3 numbers + last action | 5 numbers + last action | 6 numbers + last action |
| Weight on old evidence | Decays with each observation | Decays with each visit to that bin | Equal | Equal |

All four predict `mu_0(x) = g0` at every send, so with equal parameters their round-1 choices are identical.

M2 is Q-learning: when `x_t = x̄_a`, it is exactly `V_a += alpha (reward - V_a)` up to a fixed offset.
M4 is the `1/(a_0 + n)` rule from the draft plan.

## 4. Parameters

| Parameter | Meaning | Models | Range | Level |
|---|---|---|---|---|
| `beta` | Choice consistency | All | > 0 | Person, per task |
| `rho` | Risk penalty | All | >= 0 | Person, per task |
| `tau` | Tendency to repeat | All | real | Person, per task |
| `alpha` | Constant gain | M1, M2 | (0, 1) | Person, per task |
| `kappa` | Prior strength, in pseudo-observations | M3, M4 | > 0 | Person, per task |
| `g0` | Starting belief about return per point | All | real | Shared across people, per task, per model |

All models use the same starting belief `g0` within a task. The abstract says people got no information
about the AI's objectives or incentives. Whether the profile itself was named isn't documented. If it
was, one shared `g0` would be too simple.

Count, for each model and task: 4 per person, plus 1 shared. That's 4 x 302 + 1 = 1,209 in total.

Nothing is shared between tasks in the primary fits. Cross-task transfer reuses Task 1 parameters,
including `g0`, on Task 3 with state reset.

The counts are equal, but flexibility isn't. Retained state runs from 2 to 6 numbers, so equal AIC/BIC
penalties don't level the comparison. Held-out likelihood and model recovery have to do that.

## 5. Structured comparison, not a clean factorial

Report the four models as a structured comparison. The two dimensions interact, so don't read an effect
of one as the same at both levels of the other.

Shared by construction: same likelihood family, prior mean, valuation, choice rule, send-0 handling and
parameter count. M1 to M2 and M3 to M4 change only `psi`. M1 to M3 and M2 to M4 change only the update.

Not fully clean:

1. **The dynamics factor does more in the pooled column.** In M3, `P` also rotates, which changes how
   evidence transfers across sends. M1's transfer is fixed. In M2 and M4 `P` stays diagonal, so only the
   step size differs. So the factors interact by construction. Report the dynamics effect per column.
2. **Shrinkage speed could differ between columns. It barely does.** On real send sequences with
   `kappa = 1`, pooled gain falls 0.384 to 0.102 between rounds 2 and 15, per-send 0.389 to 0.117. With
   `kappa = 5`: 0.169 to 0.069, and 0.151 to 0.074.
3. **The labels overreach.** M1 has a partner model, so it isn't model-free. Only M2 fits that label. A
   constant gain also matches the steady state of a Bayesian filter that assumes a drifting partner. So
   dynamics tests "treats the partner as stable" against "behaves as if it drifts", not rational against
   heuristic. The draft plan's Bayesian vs model-free contrast is the M3 vs M2 diagonal.

## 6. Identifiability risks

1. **All four collapse without learning.** As `alpha -> 0` or `kappa -> infinity`, the models become one
   static model. A person who barely learns can't be classified.
2. **`beta` and the gain multiply.** The shift in choice after a surprise scales with their product, so
   the 0.044 reaction constrains the product at best.
3. **`g0` and `rho` trade off.** The opening position is about `(g0 - 1) / rho`. Sharing `g0` is meant to
   address this.
4. **`tau` and slow learning look alike.** People repeat their bin 72% of the time, against 61% when
   shuffled within person. Shuffling also removes slow drift, so 11 points is an upper limit on stickiness.
5. **Dynamics needs the gain to change within 15 rounds.** At large `kappa` the Bayesian gain is nearly
   flat and M3/M4 look like a small constant gain.
6. **Generalisation needs people to change send level.** In rounds 1-15, 27% in Task 1 and 37% in Task 3
   used at most one non-zero bin, so they give little evidence on transfer. Bins average 3 to 6 visits.
7. **Constant senders (36 and 42) carry no information on learning.** They are reported separately.

## 7. Recovery checks

Simulate equal numbers of agents from each model against the validated partner, across a range of
parameter values. Actions are bins but the simulator takes integers, so sends are drawn from the observed
distribution within each bin. Repeat a few times to get rough intervals.

- **Parameters:** Spearman, bias, RMSE, how often estimates hit a bound, and a true-vs-recovered plot.
  As a guide: 0.7 or above is usable per person, 0.4 to 0.7 is rough, below 0.4 isn't interpreted.
- **Models:** 4x4 confusion matrix, plus accuracy on each dimension, with 0.7 as a practical target.
- **If something fails:** work out why, and write it up as what 15 to 20 rounds can and can't show. A
  held-out win isn't evidence for a mechanism unless recovery supports that reading.

## 8. Optional extras

Only if time allows and they're interesting: per-person `g0` (the one to try first), full predictive
valuation, a zero-inflated likelihood, and other bin schemes.

## 9. Known misfits

The Normal likelihood is the learner's assumption. The update is exact under it, but it is not a
description of the real AI.

- **Zero spike:** a Normal has no point mass at 0. About 10% of returns are exactly 0 at sends of 4 and
  up, all defections. At sends 1, 2 and 3 it's 35%, 28% and 24%, because some strategies return little or nothing there.
- **Noise at low sends:** the spread of `y` is 1.0, 0.8 and 0.66 at sends 1 to 3, against about 0.5 from
  send 4 up. The real noise does scale with the send above that, which is why `y` is the target.
- **Response shape:** the true return per point is U-shaped, for example Competitive at 1.07, 0.47, 0.99,
  1.25 for sends 1, 3, 5, 10. Across sends 1 to 10, a sloped line explains 40% to 74% of it, and three
  steps 42% to 85%. Neither model family contains the truth.

All three misfits sit at sends 1 to 3. After fitting, plot predicted against real outcomes by send.
