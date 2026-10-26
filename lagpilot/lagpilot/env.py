"""KafkaLagEnv: a small simulator of one Kafka consumer group.

Story: an online shop writes orders to the topic "orders" (4 partitions).
A consumer group reads them. The agent decides how many consumers to run.
Every change of the consumer count triggers a rebalance. During a
rebalance the whole group stops consuming (eager, stop-the-world rebalance).

Only NumPy is used. The API is similar to Gym:
    state = env.reset(seed)
    state, reward, done, info = env.step(action)
"""

import math

import numpy as np

# Actions
KEEP, SCALE_UP, SCALE_DOWN = 0, 1, 2
ACTION_NAMES = ["keep", "scale_up", "scale_down"]

DELAY_NAMES = ["low", "medium", "high", "critical"]
TREND_NAMES = ["falling", "stable", "rising"]
RPS_NAMES = ["low", "medium", "high"]


class KafkaLagEnv:
    def __init__(self, config, use_rps_feature=True):
        self.cfg = config
        self.use_rps_feature = use_rps_feature
        self.n_actions = 3
        # Sizes of the state features, in the order used by encode():
        # delay_level, lag_trend, [rps_level], consumers, rebalancing
        self.sizes = [4, 3]
        if use_rps_feature:
            self.sizes.append(3)
        self.sizes += [config["MAX_CONSUMERS"], 2]
        self.n_states = int(np.prod(self.sizes))   # 288 with RPS, 96 without

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------
    def _make_traffic(self, rng):
        """Build the RPS curve and the arrivals for the whole episode.

        Traffic does not depend on the agent's actions, so we can draw it
        once at reset. The agent never sees these arrays directly.
        """
        c = self.cfg
        H = c["EPISODE_STEPS"]
        t = np.arange(H)
        # Smooth "daily" curve: RPS_LOW at the ends, RPS_PEAK in the middle.
        # sin^2 gives a quiet night, a busy day and a clear peak.
        base = c["RPS_LOW"] + (c["RPS_PEAK"] - c["RPS_LOW"]) * np.sin(np.pi * t / (H - 1)) ** 2

        # Random spikes (flash sales).
        spike = np.ones(H)
        step = 0
        while step < H:
            if rng.random() < c["SPIKE_START_PROB"]:
                length = rng.integers(c["SPIKE_MIN_STEPS"], c["SPIKE_MAX_STEPS"] + 1)
                spike[step:step + length] = c["SPIKE_MULTIPLIER"]
                step += length
            else:
                step += 1

        rps = base * spike
        arrivals = rng.poisson(rps * c["STEP_SECONDS"])   # messages per step
        return rps, arrivals

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.rps, self.arrivals = self._make_traffic(rng)
        self.t = 0
        self.consumers = self.cfg["START_CONSUMERS"]
        self.lag = 0
        self.prev_lag = 0
        self.rebalance_left = 0          # steps of rebalance still to go
        self.pending_consumers = self.consumers
        # Before the first step we have not measured traffic yet,
        # so we use the curve value at t = 0.
        self.observed_rps = float(self.rps[0])
        return self._get_state()

    def step(self, action):
        c = self.cfg
        rebalance_started = False

        # 1) Apply the action. Actions during a rebalance are ignored (= keep).
        #    Invalid actions (scale_up at 4, scale_down at 1) also act as keep.
        if self.rebalance_left == 0:
            target = self.consumers
            if action == SCALE_UP:
                target = min(self.consumers + 1, c["MAX_CONSUMERS"])
            elif action == SCALE_DOWN:
                target = max(self.consumers - 1, c["MIN_CONSUMERS"])
            if target != self.consumers:
                self.pending_consumers = target
                self.rebalance_left = c["REBALANCE_STEPS"]
                rebalance_started = True

        # 2) New messages arrive.
        arrivals = int(self.arrivals[self.t])

        # 3) Consumers process messages (nothing during a rebalance).
        rebalancing_now = self.rebalance_left > 0
        if rebalancing_now:
            processed = 0
            self.rebalance_left -= 1
            if self.rebalance_left == 0:
                self.consumers = self.pending_consumers   # new count applies now
        else:
            capacity = int(self.consumers * c["MU"] * c["STEP_SECONDS"])
            processed = min(self.lag + arrivals, capacity)

        # 4) Update lag.
        self.prev_lag = self.lag
        self.lag = self.lag + arrivals - processed

        # 5) Delay by Little's Law: how long a new order waits
        #    with the current consumer capacity.
        delay_s = self.lag / (self.consumers * c["MU"])
        sla_violated = delay_s > c["SLA_SECONDS"]

        # 6) Reward = minus cost.
        capped_delay = min(delay_s, c["DELAY_CAP"])
        cost = (c["W_DELAY"] * capped_delay / 10.0
                + c["W_CONS"] * self.consumers
                + c["W_SLA"] * float(sla_violated))
        reward = -cost

        self.observed_rps = arrivals / c["STEP_SECONDS"]   # measured RPS
        self.t += 1
        done = self.t >= c["EPISODE_STEPS"]

        info = {
            "t": self.t - 1,
            "rps": float(self.rps[self.t - 1]),
            "arrivals": arrivals,
            "processed": int(processed),
            "lag": int(self.lag),
            "delay_s": float(delay_s),
            "consumers": self.consumers,
            "rebalancing": bool(rebalancing_now),
            "rebalance_started": rebalance_started,
            "sla_violated": bool(sla_violated),
        }
        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _features(self):
        c = self.cfg
        delay_s = self.lag / (self.consumers * c["MU"])
        delay_level = int(np.searchsorted(c["DELAY_EDGES"], delay_s, side="right"))

        # Lag trend with a dead-band: small changes count as "stable".
        band = max(c["TREND_REL_BAND"] * self.prev_lag, c["TREND_ABS_BAND"])
        diff = self.lag - self.prev_lag
        if diff > band:
            trend = 2          # rising
        elif diff < -band:
            trend = 0          # falling
        else:
            trend = 1          # stable

        rps_level = int(np.searchsorted(c["RPS_EDGES"], self.observed_rps, side="right"))
        rebalancing = 1 if self.rebalance_left > 0 else 0

        feats = [delay_level, trend]
        if self.use_rps_feature:
            feats.append(rps_level)
        feats += [self.consumers - 1, rebalancing]
        return feats

    def _get_state(self):
        return self.encode(self._features())

    def encode(self, feats):
        """Turn a list of feature values into one integer (mixed radix)."""
        s = 0
        for value, size in zip(feats, self.sizes):
            s = s * size + value
        return s

    def decode(self, state):
        """Turn a state integer back into readable names."""
        feats = []
        for size in reversed(self.sizes):
            feats.append(state % size)
            state //= size
        feats.reverse()
        out = {"delay_level": DELAY_NAMES[feats[0]], "lag_trend": TREND_NAMES[feats[1]]}
        i = 2
        if self.use_rps_feature:
            out["rps_level"] = RPS_NAMES[feats[i]]
            i += 1
        out["consumers"] = feats[i] + 1
        out["rebalancing"] = bool(feats[i + 1])
        return out


def min_consumers_for(rps, mu):
    """Capacity estimate: smallest consumer count that keeps up with rps."""
    return math.ceil(rps / mu)
