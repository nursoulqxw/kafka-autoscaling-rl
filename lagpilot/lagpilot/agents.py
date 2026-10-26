"""Policies: baselines (KEDA rule, static max, random) and the Q-learning agent.

Every policy has act(env, state) -> action. Baselines read the raw lag from
the env (like KEDA reads it from Kafka). The Q-learning agent only uses the
discrete state.
"""

import math

import numpy as np

from lagpilot.env import KEEP, SCALE_UP, SCALE_DOWN


class KedaRulePolicy:
    """Mimics the KEDA Apache Kafka scaler.

    desired = clip(ceil(lag / lag_threshold), 1, 4)
    Then move one step toward desired. Ignores rebalance cost and RPS.
    """

    def __init__(self, config):
        self.threshold = config["KEDA_LAG_THRESHOLD"]
        self.min_c = config["MIN_CONSUMERS"]
        self.max_c = config["MAX_CONSUMERS"]

    def act(self, env, state):
        desired = math.ceil(env.lag / self.threshold)
        desired = min(max(desired, self.min_c), self.max_c)
        if desired > env.consumers:
            return SCALE_UP
        if desired < env.consumers:
            return SCALE_DOWN
        return KEEP


class StaticMaxPolicy:
    """Scale up to the maximum number of consumers, then keep."""

    def __init__(self, config):
        self.max_c = config["MAX_CONSUMERS"]

    def act(self, env, state):
        return SCALE_UP if env.consumers < self.max_c else KEEP


class RandomPolicy:
    """Picks one of the 3 actions uniformly at random."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, state):
        return int(self.rng.integers(3))


class QLearningAgent:
    """Tabular Q-learning with an epsilon-greedy behaviour policy."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, seed=0):
        self.q = np.zeros((n_states, n_actions))   # Q-table starts at zeros
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.rng = np.random.default_rng(seed)

    def greedy_action(self, state):
        # Ties are broken by the lowest index, so unseen states pick "keep".
        return int(np.argmax(self.q[state]))

    def choose_action(self, state):
        """Epsilon-greedy: explore with probability epsilon, else greedy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def act(self, env, state):
        """Used during evaluation: greedy, no exploration, no learning."""
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        """Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]"""
        if done:
            # Last step of the episode: there is no future, so target = r.
            # We treat the time limit (H = 120) as a terminal state.
            target = reward
        else:
            # r                    : the cost we just paid (negative reward)
            # gamma                : how much we care about the future
            # max_a' Q(s', a')     : value of the best action in the next state
            target = reward + self.gamma * np.max(self.q[next_state])
        # target - Q(s,a)          : TD error, how wrong our old estimate was
        td_error = target - self.q[state, action]
        # alpha                    : step size, how far we move toward the target
        self.q[state, action] += self.alpha * td_error
