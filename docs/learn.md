# Learn: concepts behind the code

Each entry explains something as it gets built: the plain idea, an intuition, a metaphor,
the maths, where it lives, and how it can go wrong. Newest entries go at the bottom.

---

## 1. Checking the data: working out how the hidden partner behaves

**Where it lives:** throwaway checks run on 5 Nov 2026, not yet in the repo. They will be
rebuilt as a proper script during the Nov 6 work.

### 1.1 Why look at the partner before modelling the person

**Plain idea.** Your models describe what a person believes about the partner. Before you can
judge those beliefs, you need to know what the partner really does.

**Metaphor.** You want to grade how well someone learned to read a dealer at a card table.
The participant can't see the dealer's hands, but you have the security footage. The footage
is for grading only: you never give it to the model of the player.

**In this dataset.** The footage is `ai_strategy_task1` and `ai_variance_task1`. Participants
never saw these columns, so they must never be inputs to a model of a participant. They are
fair to use for building a simulator of the partner.

### 1.2 Finding the rule by subtraction

**Plain idea.** If the return is "a base amount plus some noise", and you know the noise,
then subtracting it reveals the base.

**Maths.** Suppose

    r = f(profile, strategy, x) + v

where `x` is the amount sent, `v` is the recorded noise, and `f` is the rule we want. Then
`r - v` should be the same number every time the same `(profile, strategy, x)` comes up. If
it is, `f` is a fixed lookup table there. If it isn't, some other randomness is involved.

**What we found.** Most cells gave a single value. A few strategies, such as Competitive
strategy 2, gave 11 to 14 at `x = 10`, so those carry a second layer of randomness.

**How it went wrong.** This entry first assumed `r = clip(base + v)` everywhere. That is true
at high sends and false at low ones. See entry 3 for how a stricter check caught it.

### 1.3 The chi-squared test: "are these two things related?"

**Plain idea.** Count how often each combination happens, then compare that to what you'd
expect if the two things had nothing to do with each other.

**Metaphor.** If a die doesn't care what day it is, sixes should show up about as often on
Mondays as on Fridays. The test measures how far the real counts stray from that.

**Maths.** For each cell of the table,

    expected = (row total x column total) / grand total

    chi^2 = sum over cells of (observed - expected)^2 / expected

Squaring treats too-many and too-few alike. Dividing by `expected` makes a miss of 5 count
for more when you expected 10 than when you expected 1000. A big `chi^2` means a small p-value.

**What we used it for.** Testing whether the AI's strategy depends on the round, the amount
sent, the previous amount sent, or the previous strategy.

**How it can go wrong.** A significant result says something is related, not what. Strategy
looked strongly tied to round (p = 0.00003). The real cause was one specific rule: the AI
never defects in round 1. Always look at the table, not only the p-value.

### 1.4 Reasoning from a count of zero

Round 1 had 0 defections out of 302 people. In later rounds the rate is about 10%, so if
round 1 were no different, the chance of seeing zero would be

    0.9^302 = about 1.5 x 10^-14

That is far too small to be luck, so round 1 is a special case built into the game. After
removing round 1, defection no longer depends on the round (p = 0.44 in Task 1, 0.87 in Task 3).

### 1.5 The permutation test: "do some people get a worse partner?"

**Plain idea.** Shuffle the data so that any real pattern is destroyed, and see whether your
real result looks unusual next to the shuffled versions.

**Metaphor.** Deal the same pile of bad cards to the same table 4,000 times at random. If one
real player has far more bad cards than anyone ever gets in those deals, that player isn't
just unlucky.

**What we did.** Within each round, shuffle the defection flags across people. This keeps the
number of defections per round but removes any link to a particular person. Then compare the
spread of defection counts across people with the spread in the shuffled data.

**Maths.** The p-value is the share of shuffles whose variance is at least the real variance:

    p = (number of shuffles with var >= observed var) / 4000

**What we found.** Task 1 observed 1.45 vs shuffled mean 1.66 (p = 0.96). Task 3 observed 1.93
vs 1.79 (p = 0.15). So there is no sign of some people facing a harsher partner.

### 1.6 Many tests, and one result left unexplained

**Plain idea.** Run enough tests and some will look significant by chance. At a 5% threshold,
about 1 test in 20 "succeeds" with nothing going on.

**In this check.** Dozens of tests were run. One leftover remains: in Task 1, the mix of
strategies 1 to 4 varies by round (p = 0.005). It doesn't appear in Task 3 (p = 0.48). The
sensible reading is probably chance, but it isn't proven, so it stays noted as unexplained.

### 1.7 What the partner turned out to be

Each round, independently of everything that happened before:

1. With about 10% probability, return 0. This never happens in round 1.
2. Otherwise pick one of 4 strategies, about equally often.
3. Draw a noise code from -2 to 2, then look up the return from profile, strategy, send and
   noise. At high sends this behaves like "base plus noise". At low sends it doesn't (entry 3).

**Why this matters for the models.** This partner has no memory and doesn't change over time,
so a Bayesian learner's assumption of a fixed hidden distribution is correct here. A shrinking
learning rate, `1/(a0 + n)`, is then the sensible way to learn. If people still behave as if
they have a constant learning rate, that says something about them and not about the partner.

**The catch.** The partner does respond to the current send: the return per unit sent is
lower for small sends than for large ones (Competitive: 0.81 vs 1.17). A single belief about
"how generous is this partner?" can't capture that on its own.

---

## 2. Did people see the outcome? A natural experiment

**Where it lives:** `scripts/01_feedback_and_order.py`, results in
`results/01_feedback_and_order.txt`. The tidy table comes from `load_trustor_rounds` in
`src/hidden_partner/data.py`, which also labels each column as visible or hidden.

### 2.1 Why the question matters

A learning model updates on outcomes. If people never saw the outcome before their next
choice, the model would be updating on information the person never had. The deposit doesn't
document feedback, so we need another way to find out.

### 2.2 The natural experiment

**Plain idea.** To prove a cause you normally randomise. Here the AI already did that for us:
its surprises are random draws, unrelated to who the person is or how they feel. If a random
surprise is followed by a change in behaviour, the surprise caused the change, and the only
route for that is the person having seen it.

**Metaphor.** You can't ask whether someone in a windowless office can see the weather. But
if they reach for an umbrella on the days it happens to rain, and only those days, they must
have some way of seeing out.

**Maths.** Define the news in a round as

    news_t = returned_t - average return for that task, profile and send

Then fit

    change in send_t = c + b1 news_(t-1) + b2 news_(t-2) + b3 news_(t-3)
                         + p1 news_t + p2 news_(t+1)

`b1` asks whether last round's news moved this round's choice. Each unit of news means one
point more than expected.

### 2.3 Placebo tests

**Plain idea.** A test that should find nothing, to prove your method isn't finding things
that aren't there.

**Metaphor.** Before trusting a metal detector on the beach, you walk it over bare sand.

**In this analysis.** The future can't cause the past. `news_t` arrives after the choice in
round `t`, and `news_(t+1)` later still, so `p1` and `p2` must be about 0. If they aren't, the
method is broken. A second placebo: in Task 1, a defection after sending 0 changes nothing the
person can see, because they'd get 0 anyway. So it shouldn't move the next send, and it didn't.

**Result.** `b1 = 0.044` in both tasks (t = 3.8 and 4.0, pooled t = 5.1). Both placebos are
near 0 (pooled t = -0.6 and -0.3). A return 10 points better than expected raises the next
send by about 0.44. So people did see outcomes, and promptly, but the average reaction is small.

**What it doesn't show.** It's an average, so it can't prove every person saw feedback. It also
can't say whether past outcomes stayed on screen. And seeing only your payoff would be the same
information as seeing the return, because `payoff = E - x + r` and you know `E` and `x`.

### 2.4 The mistake that nearly hid the result: selecting on the outcome

**Plain idea.** Never filter your data using the thing you're trying to explain.

**Metaphor.** To test whether a hard exam makes students quit, you study only the students who
sat the final. Everyone who quit is gone, so the exam looks harmless.

**What happened.** The first version only computed news when something was sent, then kept rows
with complete news. That quietly removed anyone who dropped to 0 after bad news, which is the
strongest reaction there is. That version also used fixed effects (2.5). Together they made `b1`
look about a third as big and pushed the peak to the wrong lag. Defining news on every row and
dropping the fixed effects fixed it. Both changes were made at once, so the split between them
wasn't measured.

### 2.5 Fixed effects, and when you don't need them

**Plain idea.** Fixed effects compare each person only with themselves, by subtracting their own
average. That protects you when your predictor might be tied to who the person is.

**Why we dropped them here.** Given the send, news is random, so it can't be tied to the person.
There was nothing to protect against, and there was a cost. Subtracting a person's average mixes
all their rounds together, so a past effect can leak into a slight negative bias on the future
placebos. That's the likely reason they looked mildly negative before the fix.

### 2.6 Clustered standard errors

**Plain idea.** Twenty rounds from one person aren't twenty independent facts. Treating them as
independent makes you overconfident.

**Metaphor.** Ask one person the same question twenty times and you've learned far less than by
asking twenty people once.

**Maths.** Add up each person's score contributions first, then measure how much those totals
vary. The effective sample size becomes closer to people (302) than to rows (about 9,000). This
is also why Task 1 and Task 3 are clustered by person: they are 302 people, not 604.

### 2.7 A test that came back empty: task order

Random events can only affect choices that come later. So if Task 1 ran first, Task 1 defections
could move Task 3's opening sends, but not the other way round. The result was t = 1.6 (and the
wrong sign for learning) one way and t = 0.03 the other. That's inconclusive, probably because
defection rates vary too little between people to detect this. "We couldn't detect it" isn't "it isn't there", so task
order stays unknown from the data.

---

## 3. Rebuilding the partner, and checking it properly

**Where it lives:** `src/hidden_partner/environment.py`, `scripts/02_environment_check.py`,
`tests/test_environment.py`.

### 3.1 Keeping three kinds of information apart

**Plain idea.** The simulator knows the partner's secrets. The learner must not.

**Metaphor.** A stage magician has the trick, the audience sees the show, and a critic writes
about what the audience could have worked out. Mix those up and the review is worthless.

**In the code.** `PartnerRules` is the trick. `Visible` is the show. `Hidden` is logged for
checking the simulator only. `TrustGame.play` returns just `Visible`, so a simulated learner
can't see a strategy or noise code even by accident.

### 3.2 Lookup table instead of a formula

**Plain idea.** If you can't state the rule, store what you saw.

**What happened.** `base + noise, clipped` looked right. But for Social at send 1, strategy 1,
noise +1 gives 2 and noise +2 gives 1, and for Social at send 0 the noise is ignored entirely. So
the simulator now stores, for each profile, strategy, send and noise, which returns occurred.

**The trade-off.** A lookup has many entries (660), so it can memorise quirks of its own data.
That's why it must be tested on data it didn't see (3.5).

### 3.3 Why matching the mean isn't enough

**Plain idea.** Two very different things can share an average.

**Metaphor.** A river with an average depth of one metre can still drown you. Knowing the average
tells you nothing about the deep channel.

**In this project.** A learner responds to what actually comes back: a mix of 0s, near misses and
good returns. The broken simulator had Social send 1 almost right on average (1.33 vs 1.38) but
badly wrong in shape, so any recovery test using it would have fed models a false world.

### 3.4 Total variation distance

**Plain idea.** How much probability would you have to move to turn one distribution into the other?

**Maths.**

    TV(P, Q) = 1/2 x sum over r of |P(r) - Q(r)|

0 means identical and 1 means no overlap. The 1/2 is there because each unit moved is counted
twice, once where it leaves and once where it arrives.

### 3.5 The parametric bootstrap, and testing on the other task

**Plain idea.** Even a perfect simulator won't match a finite sample exactly. So ask how big the
gap would be by luck alone, then compare.

**Metaphor.** A fair coin flipped 100 times rarely gives exactly 50 heads. To judge whether 58 is
strange, flip a known fair coin many times and see how often 58 turns up.

**How it works.** Draw 4,000 fake datasets from the simulator with the same cell sizes. Compute the
same total gap for each. The p-value is the share of fake datasets at least as far off as the real
data.

**Results.** Formula version: failed (p near 0), even when tested on the data used to build it.
Lookup version: in-sample p = 0.92, which is expected and proves little, since a lookup is built
from that data. The real test is rules from one task predicting the other: p = 0.19 (Task 1 rules
on Task 3) and p = 0.054 (Task 3 rules on Task 1). That second one is borderline. The test treats the
estimated rules as exact when they carry some error, which makes it slightly strict.

**Takeaway.** Task 1 and Task 3 appear to use the same partner rules for sends of 1 to 10.

---

## 4. Checking facts before writing the models

**Where it lives:** `scripts/03_spec_checks.py`, results in `results/03_spec_checks.txt`, feeding
`docs/model_spec.md`. No learning model exists yet. Full entries for each model come when they're built.

### 4.1 Why a straight-line value can never pick the middle

**Plain idea.** If each extra point sent adds the same amount of value, the best send is always an
extreme: all or nothing.

**Metaphor.** If every extra spoonful of sugar made tea exactly as much better, you'd pour in the whole
bag. People stop in the middle because value bends.

**Maths.** With value `U(x) = x (mu - 1)`, softmax log-odds are a straight line in `x`, so choice
probabilities can only rise or only fall across sends. But 71% of people in Task 1 and 58% in Task 3 favour
a middle bin. So value has to bend, and the risk penalty does that:

    U(x) = x (mu - 1) - (rho / 2) x^2,    best send = (mu - 1) / rho

The penalty grows with `x^2` because the amount at risk grows with `x`.

### 4.2 Checking how noise scales

**Plain idea.** Before you model noise, measure whether it's a fixed size or grows with the bet.

**What we found.** The spread of the return grows with the send (1.0 at send 1, 5.5 at send 10), while
the spread of return per point stays near 0.5 from send 4 up. So modelling return per point is
reasonable, and a 10-point send moves a person's belief no more than a 5-point send does.

### 4.3 Stickiness, and the shuffle baseline

**Plain idea.** People repeat themselves. If a model has no way to say so, it will blame something else.

**Metaphor.** Someone orders the same lunch every day out of habit. An observer who only thinks about taste
concludes they keep re-learning that it's delicious.

**What we did.** Shuffle each person's own choices across rounds, which keeps their favourites but breaks
the order. People repeat their bin 72% of the time, against 61% shuffled. Slow drift also vanishes when
shuffled, so the 11-point gap is an upper limit on habit alone.

### 4.4 Two learning rules as one equation

**Plain idea.** A constant-rate learner and a Bayesian learner both nudge a belief toward what they just
saw. They disagree only on how big the nudge should be.

**Metaphor.** A constant-rate learner is a new driver who corrects the wheel the same amount forever. A
Bayesian learner corrects less as they grow sure of the road, and assumes the road never changes.

**Maths.** For one belief, the Bayesian posterior mean with prior strength `kappa` moves by

    gain = 1 / (kappa + n)

while the constant-rate learner uses `alpha` every time. The nudge is `gain x error` in both, so the
comparison isolates one thing. The shrinking gain isn't what makes it Bayesian. It's Bayesian because that
gain is what a stated prior and likelihood imply. A rule that shrinks for no stated reason is only a
diminishing learning rate.

**Why the checks mattered.** I expected a pooled belief to shrink far faster, since every round feeds it.
On real send sequences the two gains were almost equal (0.102 and 0.117 by round 15), so that worry
didn't hold up.

### 4.5 Zero returns, and why utility has to be chosen carefully

**Zero returns.** Sending 4 or more, a zero means the AI defected, about 10% of the time. At sends 1 to 3,
zeros reach 24% to 35%, because some strategies return little or nothing at small sends. So a 0 after sending 2 is weak
evidence of betrayal, while a 0 after sending 8 is strong evidence. The models treat them alike, which is
a known simplification.

**Why exponential utility.** Power and log utility only work for positive amounts. The learner's assumed
Normal payoff has no lower limit, so they break. Exponential utility works on any number, and with a
Normal payoff the answer has a clean form:

    certainty equivalent = mean - (lambda / 2) x variance

**Metaphor.** The certainty equivalent is the sure amount you'd swap a gamble for. It's the average,
minus a discount for how wild the gamble is.

**Posterior mean vs posterior predictive.** Using only the mean is like asking "what's my best guess?"
Using the predictive is like asking "what's my best guess, and how unsure am I?" Our Bayesian models ask
the first when choosing, so uncertainty changes how fast they learn but not what they pick.

---

## 5. The four learners

**Where they live:** `src/hidden_partner/models.py`. The `Learner` class holds beliefs, values
actions, and updates. The same class drives both fitting and simulation, so recovery tests the real
code. Equations are in `docs/model_spec.md`, and `tests/test_models.py` checks them.

All four do the same loop each round: predict, value, choose, see the outcome, update. They only
differ in what a belief is about, and how far one outcome moves it.

### 5.1 M2: constant gain, per-send (Q-learning)

**Plain idea.** Keep a separate note for each send level, and every time you try one, nudge that note
a fixed fraction of the way toward what happened.

**Metaphor.** You rate three restaurants. Each visit moves that restaurant's score part of the way
toward tonight's meal, and tonight's meal says nothing about the other two.

**Maths.**

    w_a <- w_a + alpha (y - w_a)

The bracket is the prediction error, and `alpha` is how much of it you absorb. Unroll this and old
outcomes fade by `(1 - alpha)` each visit, so with `alpha = 0.3` an outcome three visits back keeps
about a third of its weight:

    w_n = (1 - alpha)^n w_0  +  sum over k of alpha (1 - alpha)^(n - k) y_k

Recent evidence counts more, which is the right habit if the world might be changing.

**Link to RL.** This is Q-learning for a one-step problem. There is no next state, so the usual
`gamma max Q(s', a')` term disappears, and the update is the bare delta rule. The test
`test_m2_is_q_learning_at_the_bin_value_point` proves the equivalence at send 10.

**Going wrong.** Untried options never improve. If you never go back to a restaurant after one bad
meal, its score stays bad forever.

### 5.2 M1: constant gain, pooled

**Plain idea.** Same fixed nudge, but now you're learning traits of the partner, not notes per send.

**Metaphor.** Rating a chef, not dishes. One great dish lifts your opinion of the whole menu.

**Maths.** The belief is two numbers: generosity `g` and responsiveness `b`.

    mu(x) = g + b s c(x)
    m <- m + alpha psi e / (psi' psi)

Dividing by `psi' psi` makes the prediction at the chosen send move by exactly `alpha` of the error,
however the features are scaled. So `alpha` means the same thing here as in M2. The rest of the change
spills onto other sends, in proportion to how alike their features are. This is called normalised LMS,
and it's the same idea as gradient descent on squared error with a tidy step size.

**Going wrong.** The line can't bend, and the real response is U-shaped. M1 will always get something
wrong at low sends.

### 5.3 M4: Bayesian, per-send

**Plain idea.** Keep a separate belief per send level, and let each one get harder to move the more
evidence it has.

**Metaphor.** A batting average. In a player's first game one hit changes it a lot. After a thousand
at-bats, one hit barely shows.

**Maths.** With a Normal prior and Normal likelihood, the posterior mean is a weighted average of the
prior and the data. Treat the prior as `kappa` imaginary observations:

    w_n = (kappa g0 + y_1 + ... + y_n) / (kappa + n)

Subtract `w_(n-1)` and it becomes the update itself:

    w_n = w_(n-1) + (y_n - w_(n-1)) / (kappa + n)

This is the delta rule again, with `alpha` replaced by `1 / (kappa + n)`. Every observation ends up
with equal weight, which is why the order of events doesn't matter: `test_bayesian_order_does_not_matter`
checks this, and `test_constant_gain_order_does_matter` checks the opposite for M2.

**Why this is Bayesian.** That gain falls out of a stated prior and likelihood. It isn't chosen. It's
also the right rule only if the partner never changes.

### 5.4 M3: Bayesian, pooled

**Plain idea.** Hold a whole cloud of belief over the partner's two traits, and shrink it as evidence
comes in, more in the directions you've tested.

**Metaphor.** You're working out a rectangle's width and height, but you can only measure its perimeter.
A measurement narrows a combination of the two, not each separately. Measuring in a different way
narrows a different combination.

**Maths.** This is recursive Bayesian linear regression, also known as a Kalman filter with no drift:

    K = P psi / (psi' P psi + 1)        how much to trust this observation
    m <- m + K e                        move the belief
    P <- P - K psi' P                   become less unsure, in this direction

`P` is uncertainty. If `P` is large, `K` is large, so a surprise moves you a lot. Each observation shrinks
`P`, so learning slows by itself. Because `P` carries correlations between traits, it also changes how one
send's outcome spreads to others. `test_bayesian_update_matches_the_batch_posterior` confirms that stepping
through rounds lands exactly on the formula that uses all the data at once.

### 5.5 What the four share, and when they're the same

Round 1 is identical in all four. And if learning is switched off, they collapse into one model, because
nothing ever moves. `test_models_collapse_without_learning` checks that. So someone who barely learns can't
be classified, by any method.

The value function is the same too:

    U(a) = x (mu(x) - 1) - (rho / 2) x^2

Then choice is a softmax, with `tau` added to whichever action was picked last time:

    P(a) proportional to exp( beta U(a) + tau 1[a = last action] )

`beta` is decisiveness and `tau` is the pull of repeating. Round 1 has no last action, so no `tau`.

---

## 6. Fitting and recovery

**Where they live:** `src/hidden_partner/fitting.py`, `src/hidden_partner/simulate.py`,
`scripts/04_recovery_fits.py`, `scripts/05_recovery_report.py`.

### 6.1 Maximum likelihood

**Plain idea.** Find the settings that make what the person did least surprising.

**Metaphor.** Tuning a radio. You turn the dials until the signal is clearest.

**Maths.** The model gives a probability to each real choice. Multiply them, and take the log so the
product becomes a sum and doesn't underflow:

    LL(theta) = sum over t of log P(a_t | history_t, theta)

Beliefs update on every round. `scored` only controls which choices count toward the total, which is
how held-out rounds work: they keep learning, but they don't get to tune the dials.

### 6.2 Parameter transforms and multiple starts

**Transforms.** `beta` must be positive and `alpha` must be between 0 and 1. So the optimiser searches over
`log(beta)` and `logit(alpha)`, which can be any number, and we convert back. It's like walking on a flat map
of a curved surface.

**Multiple starts.** An optimiser only walks downhill, so it stops at the nearest valley. Dropping it in 4
random places and keeping the best makes it less likely to miss the deepest one.

### 6.3 Parameter recovery and model recovery

**Plain idea.** Before trusting a measuring tool on people, test it on fakes where you know the answer.

**Metaphor.** Before weighing a mystery object, you put a known 1 kg weight on the scale.

**Parameter recovery.** Simulate from known parameters, refit, and compare. Spearman asks whether people
keep their order, bias asks whether estimates run high or low on average, and RMSE gives the typical miss.
Spearman alone isn't enough: a scale that always reads double keeps everyone's order perfectly.

**Model recovery.** Simulate from each model, fit all of them, and count which wins. The result is a
confusion matrix. Ideally the diagonal dominates. All four models have the same number of parameters here,
so the winner is just the highest likelihood.

**Why this matters more than the final fit.** If the fitting can't tell M2 from M4 on fake data where the
answer is known, it can't do so on people, however confident the numbers look.

### 6.4 Ties, and why "first in the list" is a hidden bias

**Plain idea.** When two options score the same, your tie-break rule is making the decision, even if you
never meant it to.

**What happened.** When learning vanishes in a fit, the models become the same model and score the same. A
plain "pick the highest" then always chose whichever was listed first, which quietly favoured M1 and M2.
About 17% of agents were near-tied. Now anything within 0.01 of the best shares the credit equally.

### 6.5 What recovery found, and how to read it

**Plain idea.** Fifteen rounds can see some things and not others. That's a real finding about the data.

| Rounds | All four (chance 0.25) | Generalisation (chance 0.5) | Dynamics (chance 0.5) |
|---|---|---|---|
| 15 | 0.50 | 0.77 | 0.62 |
| 20 | 0.55 | 0.78 | 0.67 |
| 40 | 0.61 | 0.83 | 0.71 |

**Generalisation is easy to see.** Pooled learners change their opinion of untried sends and per-send
learners don't, and that shows up in choices quickly.

**Dynamics is hard to see.** A Bayesian gain falling from 0.38 to 0.10 over 15 rounds has an average near
0.2, so a constant gain of 0.2 can stand in for it.

**Metaphor.** Picture a car slowing down gently over a short stretch of road. A car holding a steady
middling speed covers nearly the same ground, so from two photos you can't tell them apart. You need a
longer road.

**It isn't symmetric.** Bayesian agents go to the right dynamics less often than constant-gain agents do.
A constant gain can imitate a shrinking one, but a shrinking gain can't imitate a large constant gain, because
it must keep slowing. So M1 and M2 are more flexible, even with the same parameter count. In real data,
expect them to win somewhat more often than people truly behave that way.

**Parameters.** `rho` recovers best (0.71 to 0.84). `beta` is rough (0.62 to 0.68) and runs about twice too
high, because when a model predicts every choice, the fit pushes `beta` to its limit. `tau` (0.30 to 0.43) and
the gain (0.42 to 0.53) are too noisy to compare individuals.

**Going wrong.** `g0` was held at its true value in these runs, and simulated agents send less and repeat less
than real people, so this is a first look. It gets rerun at fitted values.
