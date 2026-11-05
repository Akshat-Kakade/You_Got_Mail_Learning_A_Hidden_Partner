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
