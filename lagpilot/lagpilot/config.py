"""All constants for LagPilot in one place.

These values were tuned once, only to make the problem non-trivial
(1 consumer is not enough at the peak, 4 consumers are enough but
expensive, spikes hurt, the KEDA rule is reasonable). Tuning used only
the baseline policies on seeds 5000-5039, before any Q-learning run.
They are frozen before the final experiments. Do not tune them to make
Q-learning win.
"""

# ---------- Time ----------
STEP_SECONDS = 10          # one step = 10 seconds
EPISODE_STEPS = 120        # 120 steps = 20 minutes of simulated traffic

# ---------- Traffic (requests per second, RPS) ----------
RPS_LOW = 20.0             # RPS at the start and at the end of the episode
RPS_PEAK = 120.0           # RPS in the middle of the episode
SPIKE_START_PROB = 0.04    # chance per step that a spike (flash sale) starts
SPIKE_MIN_STEPS = 5        # shortest spike
SPIKE_MAX_STEPS = 15       # longest spike
SPIKE_MULTIPLIER = 2.0     # RPS is multiplied by this during a spike

# ---------- Kafka consumer group ----------
PARTITIONS = 4             # topic "orders" has 4 partitions
MIN_CONSUMERS = 1
MAX_CONSUMERS = 4          # more consumers than partitions would sit idle
START_CONSUMERS = 2
MU = 50.0                  # messages per second one consumer can process
REBALANCE_STEPS = 2        # R: steps the whole group is stopped after a change

# ---------- SLA ----------
SLA_SECONDS = 30.0         # an order must be processed within 30 seconds

# ---------- Reward weights ----------
W_DELAY = 1.0              # cost per 10 seconds of delay
W_CONS = 2.0               # cost per running consumer per step
W_SLA = 5.0                # extra cost for a step that violates the SLA
DELAY_CAP = 120.0          # delay is capped at 120 s inside the reward

# ---------- State discretization ----------
DELAY_EDGES = (5.0, 15.0, 30.0)   # low < 5 <= medium < 15 <= high < 30 <= critical
RPS_EDGES = (40.0, 90.0)          # low < 40 <= medium < 90 <= high
TREND_REL_BAND = 0.05             # lag change within +-5% counts as "stable"
TREND_ABS_BAND = 100              # ... or within +-100 messages

# ---------- Baseline: KEDA-style rule ----------
KEDA_LAG_THRESHOLD = 500   # KEDA "lagThreshold": target lag per consumer
                           # (best of {250, 500, 1000, 2000} on tuning seeds)

# ---------- Q-learning ----------
ALPHA = 0.1                # learning rate
GAMMA = 0.95               # discount factor
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.7   # epsilon reaches EPS_END after 70% of training

# ---------- Experiments ----------
TRAIN_EPISODES = 3000
TRAIN_SEEDS = list(range(10))              # 0..9
EVAL_SEEDS = list(range(1000, 1050))       # 1000..1049, never used for training
EVAL_EVERY = 50                            # greedy check every 50 episodes
EVAL_EPISODES_DURING_TRAINING = 10


def make_config(**overrides):
    """Return all constants as a dict. Overrides are used by experiments
    (for example REBALANCE_STEPS=3)."""
    cfg = {k: v for k, v in globals().items() if k.isupper()}
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = value
    return cfg
