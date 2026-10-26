# LagPilot: the theory behind the project

This file explains every idea used in LagPilot **from zero**. You do not need
to know Kafka or reinforcement learning before reading it. Each part ends with
"**In LagPilot:**", which shows where the idea appears in the code.

For how to run the project and the results, see [README.md](README.md).
File paths in this document (like `lagpilot/env.py` or `results/...`) are
relative to the `lagpilot/` project folder.

---

## Contents

**Part A: Backend side**
1. [Message queues and Apache Kafka](#1-message-queues-and-apache-kafka)
2. [Topics, partitions and offsets](#2-topics-partitions-and-offsets)
3. [Consumer groups](#3-consumer-groups)
4. [Consumer lag](#4-consumer-lag)
5. [Rebalances](#5-rebalances)
6. [RPS, capacity and Little's Law](#6-rps-capacity-and-littles-law)
7. [SLA](#7-sla)
8. [Autoscaling and KEDA](#8-autoscaling-and-keda)

**Part B: Reinforcement learning side**

9. [What reinforcement learning is](#9-what-reinforcement-learning-is)
10. [Markov decision process (MDP)](#10-markov-decision-process-mdp)
11. [The Markov property](#11-the-markov-property)
12. [Return and discount γ](#12-return-and-discount-γ)
13. [Q-values and the Q-table](#13-q-values-and-the-q-table)
14. [The Q-learning update](#14-the-q-learning-update)
15. [Exploration: ε-greedy](#15-exploration-ε-greedy)
16. [Why RL here, and when not](#16-why-rl-here-and-when-not)

**Part C: How the experiment is designed**

17. [Simulator design](#17-simulator-design)
18. [Training vs evaluation, seeds, mean ± std](#18-training-vs-evaluation-seeds-mean--std)
19. [Glossary](#19-glossary)
20. [Further reading and videos](#20-further-reading-and-videos)

---

# Part A: Backend side

## 1. Message queues and Apache Kafka

Imagine an online shop. When a customer clicks "Buy", the web server must
charge the card, update the stock, send an email, and so on. If the web server
did all of this immediately, a sudden rush of customers would overload it.

So the web server only writes a short note, "order #123 was created", into a
**message queue** and answers the customer at once. Other programs read notes
from the queue and do the slow work at their own speed. The queue is a
**buffer** between "work arrives" and "work is done".

**Apache Kafka** is the most popular system of this kind. Words to know:

| Word | Meaning |
|---|---|
| **message** | one note, e.g. one order |
| **producer** | a program that writes messages (our web API) |
| **consumer** | a program that reads and processes messages (our worker) |
| **broker** | the Kafka server that stores messages |

Kafka does not delete a message when it is read. It keeps a long log of all
messages, and every consumer remembers **how far it has read**.

**In LagPilot:** `demo/producer_api.py` is a producer (`POST /orders` writes one
message). The simulator in `lagpilot/env.py` does not run Kafka; it simulates
the numbers.

## 2. Topics, partitions and offsets

- A **topic** is a named stream of messages, e.g. `orders`.
- A topic is split into **partitions**. Each partition is a separate,
  ordered list of messages. Partitions let many consumers work **in
  parallel**.
- Each message in a partition has a number called its **offset**: 0, 1, 2, …
- The **log end offset** is the offset of the next message that will be
  written, so it shows how many messages were written to that partition.
- The **committed offset** is how far the consumer group has finished
  reading in that partition.

```
partition 0:  [0][1][2][3][4][5][6][7][8][9]
                              ▲              ▲
                     committed offset = 5   log end offset = 10
                     (5 messages still waiting → lag = 5)
```

**In LagPilot:** the topic `orders` has **4 partitions** (`PARTITIONS = 4` in
`config.py`).

## 3. Consumer groups

Several consumers can join one **consumer group** to share the work. Kafka
gives each partition to **exactly one** consumer in the group:

```
4 partitions, 2 consumers:      4 partitions, 4 consumers:     4 partitions, 5 consumers:
 C1 ← P0, P1                     C1 ← P0                        C1 ← P0
 C2 ← P2, P3                     C2 ← P1                        C2 ← P1
                                 C3 ← P2                        C3 ← P2
                                 C4 ← P3                        C4 ← P3
                                                                C5 ← (nothing, idle!)
```

So **more consumers than partitions is useless**: the extra one waits.

**In LagPilot:** the agent can choose 1 to 4 consumers (`MAX_CONSUMERS = 4`).
In the demo, `consumer_manager.py` runs consumers as threads in the group
`lagpilot-group`.

## 4. Consumer lag

**Lag** = how many messages were written but not yet processed:

```
lag = Σ over partitions ( log end offset − committed offset )
```

Lag is the most important health signal of a consumer group:

- lag ≈ 0 → consumers keep up;
- lag growing → messages arrive faster than they are processed;
- lag shrinking → consumers are catching up.

**In LagPilot:** the simulator keeps lag as one number `L` and updates it every
step: `L ← L + arrivals − processed` (`env.py`, `step`, part 4). The demo reads
the real lag from Kafka offsets (`demo/agent.py`, `read_offsets`).

## 5. Rebalances

When a consumer **joins** or **leaves** the group, Kafka must re-divide the
partitions among the consumers. This is a **rebalance**.

In the classic **eager** ("stop-the-world") protocol:

1. every consumer gives back **all** its partitions,
2. the group agrees on a new assignment,
3. every consumer starts again with its new partitions.

**During steps 1–3 nobody in the group processes messages.** The lag grows. A
rebalance can take from a few seconds to tens of seconds in real systems.

Newer Kafka versions have **cooperative incremental rebalancing** (KIP-429):
only the partitions that really move are paused, so a rebalance is much
cheaper.

This is the heart of the project: **scaling is not free.** If you add a
consumer to reduce lag, the lag first gets *worse* during the rebalance.

**In LagPilot:** a scaling action starts a rebalance of `REBALANCE_STEPS = 2`
steps (20 s). During it `processed = 0`, and other scaling actions are
ignored. The new consumer count applies when the rebalance ends (`env.py`,
`step`, parts 1 and 3). We model the eager protocol, which is a limitation.

## 6. RPS, capacity and Little's Law

**RPS** = requests per second. Here: orders per second coming into the topic.

**Capacity** = how many messages per second the group can process. If one
consumer handles μ messages per second, `c` consumers handle `c × μ`.

**Capacity estimate** (the standard back-of-the-envelope check in backend
engineering):

```
consumers needed = ceil( RPS / μ )
night:  20 RPS / 50 = 0.4  → 1 consumer
peak:  120 RPS / 50 = 2.4  → 3 consumers
spike: 240 RPS / 50 = 4.8  → 5 consumers, but the max is 4 (= 200 msg/s)
```

During a big spike, `240 − 200 = 40` messages per second cannot be processed,
so lag **must** grow by about 400 messages every 10 seconds. No policy can
prevent it; a good policy can only limit the damage.

**Little's Law** (J. Little, 1961) is a simple, very general rule for queues:

```
L = λ × W
L = average number of items in the system
λ = average rate at which items leave (throughput)
W = average time an item spends in the system
```

Turn it around: `W = L / λ`. If there are 1000 orders waiting and the group
processes 100 orders per second, a new order waits about 10 seconds.

**In LagPilot:** the delay of a new order is

```
delay_s = lag / (consumers × μ)
```

Example: lag = 3000, 2 consumers, μ = 50 → delay = 3000 / 100 = 30 s → right at
the SLA limit. With 3 consumers, delay = 3000 / 150 = 20 s.

## 7. SLA

An **SLA** (service level agreement) is a promise about quality. Ours: **"an
order must be processed within 30 seconds."** A step where `delay_s > 30` is an
**SLA violation**.

**In LagPilot:** `SLA_SECONDS = 30`. The reward has an extra penalty of 5 for
each violating step, and results report the % of steps with a violation.

## 8. Autoscaling and KEDA

**Autoscaling** = automatically changing the number of workers based on load.

**KEDA** (Kubernetes Event-Driven Autoscaling) is the standard tool for this in
Kubernetes. Its Apache Kafka scaler roughly does:

```
desired consumers = ceil( lag / lagThreshold )
```

With `lagThreshold = 500`: lag 1200 → 3 consumers, lag 300 → 1 consumer.

It is simple and reacts quickly, but it has two blind spots:

1. **It ignores the rebalance cost.** When lag drops, it removes a consumer;
   the rebalance stops processing; lag grows; it adds the consumer back. This
   back-and-forth is called **flapping**. (Real KEDA/HPA has cooldown and
   stabilization windows to reduce it; our simplified rule does not.)
2. **It only reacts to lag**, not to incoming traffic, so it scales *after*
   the problem has started.

**In LagPilot:** `KedaRulePolicy` in `agents.py`. It moves one step per
decision toward `desired`. You can see the flapping in
`results/fig4_episode_trace.png`.

---

# Part B: Reinforcement learning side

## 9. What reinforcement learning is

In **reinforcement learning (RL)** an **agent** learns by trial and error. It
repeats a loop:

```
   ┌────────────── state s ───────────────┐
   │                                      │
   ▼                                      │
 AGENT ──── action a ────► ENVIRONMENT ───┤
   ▲                                      │
   └────────────── reward r ──────────────┘
```

1. The agent sees the **state** (the situation).
2. It picks an **action**.
3. The environment changes and gives back a **reward** (a number: how good
   that was) and the next state.

Nobody tells the agent the right action. It must discover which actions lead
to **high total reward over time**, not just a good reward right now.

| RL word | In LagPilot |
|---|---|
| agent | the autoscaler (`QLearningAgent`) |
| environment | the simulated consumer group (`KafkaLagEnv`) |
| state | delay level, lag trend, traffic level, consumers, rebalancing |
| action | keep / scale_up / scale_down |
| reward | minus the cost of this 10-second step |
| episode | one simulated day = 120 steps |
| policy | the rule "in state s, do action a" |

## 10. Markov decision process (MDP)

An **MDP** is the formal description of an RL problem. It has:

- **S, states.** Here 288 discrete states (see below).
- **A, actions.** Here 3 actions.
- **P, transitions.** How the next state follows from the state and action.
  Here it is random: orders arrive as a Poisson process and flash sales
  start at random.
- **R, reward.** Here `r = −(delay/10 + 2 × consumers + 5 × [delay > 30])`.
- **Start state.** 2 consumers, lag 0.
- **Horizon H.** Here 120 steps (the episode then ends).
- **Discount γ.** Here 0.95 (see section 12).

**How the state is built.** The agent cannot use exact numbers in a table, so
each measurement is put into a few **buckets** (discretization):

| Feature | Buckets |
|---|---|
| delay_level | low (< 5 s), medium (< 15 s), high (< 30 s), critical (≥ 30 s) |
| lag_trend | falling / stable / rising (changes smaller than max(5%, 100 msgs) count as stable) |
| rps_level | low (< 40), medium (< 90), high (≥ 90) orders per second |
| consumers | 1, 2, 3, 4 |
| rebalancing | no / yes |

`4 × 3 × 3 × 4 × 2 = 288` states. Each combination is turned into one integer
(`env.encode`), and back into names (`env.decode`).

**Why the reward looks like this:**

- `delay / 10`: waiting orders cost money (unhappy customers). It is capped at
  120 s so one terrible step does not dominate everything.
- `2 × consumers`: servers cost money.
- `5 × [delay > 30]`: extra pain for breaking the SLA.
- **There is no direct penalty for a rebalance.** Its cost shows up
  *indirectly*: no processing → lag grows → delay grows → bigger costs in the
  next steps. Learning this delayed effect is exactly the job of RL.

**One real step** (from `results/transition_example.json`):

```
state:   delay low, lag stable, traffic low, 2 consumers, no rebalance
action:  scale_down
what happened: 196 orders arrived, 0 processed (rebalance started), lag = 196
         delay = 196 / (2 × 50) = 1.96 s
reward:  −(1.96/10 + 2 × 2 + 0) = −4.196
next:    delay low, lag rising, traffic low, 2 consumers, rebalancing = yes
```

At night one consumer is enough, so the agent pays one short rebalance now to
save a server for the rest of the night.

## 11. The Markov property

A state is **Markov** if it contains everything needed to predict the future:
knowing the past adds nothing.

Our state is only **approximately** Markov. The agent does **not** see:

- the exact lag (only a delay bucket: 31 s and 300 s look the same);
- the exact traffic (only low / medium / high);
- whether a flash sale is happening and how long it will last;
- the time of day (it does not know that night is coming);
- how many rebalance steps are left.

So the same state can lead to very different futures. This makes learning
noisier. The experiment "with vs without `rps_level`" (README section 4.4)
shows that giving the agent more of this hidden information helps: it reacts
earlier and breaks the SLA less often.

## 12. Return and discount γ

The agent wants a large **return**, the sum of future rewards. Rewards far in
the future are weighted less:

```
G = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + …
```

With γ = 0.95, a reward 20 steps ahead counts about 0.95²⁰ ≈ 0.36 as much as a
reward now. The **effective horizon** is about `1 / (1 − γ) = 20 steps = 200 s`.
That is long enough to "see" the cost of a 20 s rebalance and the end of a
typical flash sale, and short enough to learn quickly.

## 13. Q-values and the Q-table

**Q(s, a)** = "how good is it to take action `a` in state `s`, and then act
well afterwards?", measured as expected return.

Because there are only 288 states and 3 actions, we can store all Q-values in
a **table** (a 288 × 3 array, `self.q` in `agents.py`). It starts at zeros.

**Using the table:** in state `s`, look at its row and pick the action with the
largest value:

```
row of state 26:   keep −81.71   scale_up −79.37   scale_down −76.63
                                                   ▲ largest → choose scale_down
```

(All values are negative because all rewards are costs. "Largest" means
"least bad".)

## 14. The Q-learning update

After every step `(s, a, r, s′)` the agent improves one cell of the table:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a′ Q(s′,a′) − Q(s,a) ]
```

Read it piece by piece:

| Piece | Meaning in simple words |
|---|---|
| `Q(s,a)` | my old guess |
| `r` | what I just paid in this step |
| `max_a′ Q(s′,a′)` | how good the next state is if I act well from there |
| `γ · max_a′ Q(s′,a′)` | that future value, discounted |
| `r + γ · max Q(s′,·)` | the **TD target**: a better guess, built from one real step |
| `target − Q(s,a)` | the **TD error**: how wrong my old guess was |
| `α` (0.1) | how far I move toward the target (small = average over noise) |

**Small numeric illustration** (made-up round numbers, just to show the
arithmetic):

```
old Q(s,a) = −80,   r = −4,   max Q(s′,·) = −78,   γ = 0.95,   α = 0.1
target   = −4 + 0.95 × (−78) = −4 − 74.1 = −78.1
TD error = −78.1 − (−80) = +1.9        (the action was a bit better than I thought)
new Q    = −80 + 0.1 × 1.9 = −79.81
```

**At the last step of the day** there is no future, so `target = r`. We treat
the time limit (step 120) as the end of the world for the agent.

**Why "off-policy":** the target uses the *best* next action (`max`), not the
action the agent will actually take while exploring. So Q-learning learns the
greedy policy even while it behaves randomly sometimes.

**In LagPilot:** `QLearningAgent.update` in `agents.py`; every term has a
comment.

## 15. Exploration: ε-greedy

If the agent always picks the best-looking action, it may never discover that
another action is better. So during training it uses **ε-greedy**:

- with probability **ε**: pick a random action (**explore**);
- otherwise: pick the action with the highest Q (**exploit**).

ε starts at 1.0 (fully random) and goes down linearly to 0.05 over the first
70% of training (2100 of 3000 episodes). Then it stays at 0.05.

During **evaluation** ε = 0: the agent only exploits and does not learn.

**In LagPilot:** `choose_action` (training) and `act` (evaluation) in
`agents.py`; the ε schedule is in `train.py`, `train_seed`.

## 16. Why RL here, and when not

**Why RL can help:**

- decisions are **sequential**: lag carries over from step to step;
- actions have **delayed effects**: a rebalance now costs throughput later;
- traffic is **random**, so the right timing must be learned from experience.

**When a simple rule is better** (and you should use KEDA or capacity planning):

- traffic is stable or predictable;
- rebalances are cheap (cooperative rebalancing);
- the team needs a rule that is easy to explain and audit.

Our own results show this: when rebalances take 30 s, simply running 4
consumers all the time beats the average Q-learning agent.

---

# Part C: How the experiment is designed

## 17. Simulator design

Each 10-second step in `KafkaLagEnv.step` does six things, in order:

1. **Apply the action.** If no rebalance is running and the action changes the
   count, start a rebalance of R steps. Invalid actions (scale_up at 4,
   scale_down at 1) and actions during a rebalance act as `keep`.
2. **Orders arrive:** `arrivals ~ Poisson(RPS × 10)`. RPS follows a `sin²` daily
   curve from 20 to 120 and is doubled during a flash sale.
3. **Process:** `processed = min(lag + arrivals, consumers × 50 × 10)`, or 0
   during a rebalance.
4. **Update lag:** `lag ← lag + arrivals − processed`.
5. **Delay (Little's Law):** `delay = lag / (consumers × 50)`.
6. **Reward:** minus the cost.

The traffic of the whole day is drawn at `reset`, because it does not depend
on the agent's actions. This makes the code simpler and faster, and it
guarantees that every policy sees **exactly the same day** for the same seed.

## 18. Training vs evaluation, seeds, mean ± std

- **Training and evaluation are separate.** We train on some days and test
  on **other days** (seeds 1000–1049) that the agent never saw. During testing
  the agent does not learn and does not explore.
- **Random seeds.** A seed fixes the random numbers, so the same seed gives
  the same day. One training run can be lucky or unlucky, so we train **10
  agents** with seeds 0–9 and report the average.
- **mean ± std.** The mean is the average result; the standard deviation
  (std) shows how spread out the results are. A large std means "results vary
  a lot". The median and the per-seed table show whether one bad run causes
  it (in our case: seed 2).
- **Baselines.** A result means nothing alone. We compare with KEDA (the
  real-world rule), static max (safe but expensive) and random (lower bound).
- **Parameter study.** We change one setting (the rebalance duration R) and
  retrain, to see how the answer depends on it.
- **No tuning for the win.** The simulator config was tuned once, only with
  baselines, before any Q-learning run, then frozen.

## 19. Glossary

| Term | Meaning |
|---|---|
| broker | Kafka server that stores messages |
| committed offset | how far a consumer group has finished reading a partition |
| consumer | program that reads and processes messages |
| consumer group | consumers that share the partitions of a topic |
| eager rebalance | all partitions are revoked and reassigned; the group stops |
| cooperative rebalance | only moved partitions pause (KIP-429) |
| flapping | scaling up and down again and again |
| KEDA | Kubernetes Event-Driven Autoscaling |
| lag | messages written but not yet processed |
| Little's Law | L = λ × W: items in system = rate × time in system |
| log end offset | offset of the next message to be written |
| partition | ordered sub-log of a topic; unit of parallelism |
| RPS | requests (orders) per second |
| SLA | promise about service quality (here: delay ≤ 30 s) |
| α (alpha) | learning rate, step size of the update |
| γ (gamma) | discount factor, how much the future matters |
| ε (epsilon) | probability of a random (exploration) action |
| episode | one run from start to end (one simulated day) |
| greedy | always choose the action with the highest Q |
| MDP | Markov decision process: states, actions, transitions, rewards |
| off-policy | learns one policy (greedy) while following another (ε-greedy) |
| policy | rule that maps a state to an action |
| Q-value | expected return of taking an action in a state |
| return | (discounted) sum of rewards |
| TD error | target minus old estimate |

## 20. Further reading and videos

**Kafka**
- Apache Kafka documentation (consumer groups, rebalance): https://kafka.apache.org/documentation/
- Confluent, "Kafka Rebalancing Explained": https://www.confluent.io/learn/kafka-rebalancing/
- KIP-429, incremental cooperative rebalancing: https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol
- Video, Kafka in 100 seconds: https://www.youtube.com/watch?v=YGoHJW_3fPs
- Video, consumer groups and offsets: https://www.youtube.com/watch?v=9o5LAbPNc28

**KEDA**
- KEDA Apache Kafka scaler: https://keda.sh/docs/latest/scalers/apache-kafka/
- Video, introduction to KEDA autoscaling: https://www.youtube.com/watch?v=fgwYp0zIIG0

**Little's Law**
- J. D. C. Little, "A proof for the queuing formula L = λW", *Operations Research* 9(3), 1961.
- Video series, Little's Law 1: https://www.youtube.com/watch?v=zQey2zav7jQ

**Reinforcement learning**
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018 (ch. 3 MDPs, ch. 6.5 Q-learning), free online: http://incompleteideas.net/book/the-book-2nd.html
- Video, David Silver RL course, lecture 1: https://www.youtube.com/watch?v=2pWv7GOvuf0
- Video, MDP in 5 minutes: https://www.youtube.com/watch?v=4Fqt2Nk2lhY
- Video, Q-learning explained: https://www.youtube.com/watch?v=RiSCh4H-GY4
