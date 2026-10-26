"""Helpers to run episodes, evaluate policies and train Q-learning."""

import numpy as np


def run_episode(env, policy, seed, log=False):
    """Run one episode with a fixed policy (no learning).

    Returns a dict with episode metrics. With log=True it also returns
    the per-step info list (used for the episode trace figure).
    """
    state = env.reset(seed)
    total_reward = 0.0
    delays, consumers, steps = [], [], []
    sla_steps = 0
    rebalances = 0
    done = False
    while not done:
        action = policy.act(env, state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        delays.append(info["delay_s"])
        consumers.append(info["consumers"])
        sla_steps += info["sla_violated"]
        rebalances += info["rebalance_started"]
        if log:
            info = dict(info, action=int(action), reward=reward,
                        state=int(state), next_state=int(next_state))
            steps.append(info)
        state = next_state
    result = {
        "return": total_reward,
        "avg_delay_s": float(np.mean(delays)),
        "sla_violation_pct": 100.0 * sla_steps / len(delays),
        "avg_consumers": float(np.mean(consumers)),
        "rebalances": rebalances,
    }
    if log:
        result["steps"] = steps
    return result


def evaluate(env, policy, seeds):
    """Run one episode per seed. Returns a list of metric dicts."""
    return [run_episode(env, policy, s) for s in seeds]


def train_seed(env, cfg, seed, check_seeds):
    """Train one Q-learning agent with one random seed.

    Training episodes use seeds 10_000_000 + seed * 100_000 + episode, so
    they never overlap with the check seeds or the final evaluation seeds.
    Every EVAL_EVERY episodes we run a greedy check (no learning) on
    check_seeds and save the mean return.

    Returns (agent, curve) where curve is a list of (episode, mean_return).
    """
    from lagpilot.agents import QLearningAgent

    agent = QLearningAgent(env.n_states, env.n_actions,
                           alpha=cfg["ALPHA"], gamma=cfg["GAMMA"], seed=seed)
    n_episodes = cfg["TRAIN_EPISODES"]
    decay_episodes = cfg["EPS_DECAY_FRACTION"] * n_episodes
    curve = []

    for episode in range(n_episodes):
        # Linear epsilon decay from EPS_START to EPS_END, then constant.
        frac = min(episode / decay_episodes, 1.0)
        agent.epsilon = cfg["EPS_START"] + frac * (cfg["EPS_END"] - cfg["EPS_START"])

        state = env.reset(10_000_000 + seed * 100_000 + episode)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if (episode + 1) % cfg["EVAL_EVERY"] == 0:
            res = evaluate(env, agent, check_seeds)
            curve.append((episode + 1, float(np.mean([r["return"] for r in res]))))

    return agent, curve


if __name__ == "__main__":
    # Sanity check: run every baseline for 20 episodes.
    from lagpilot.agents import KedaRulePolicy, RandomPolicy, StaticMaxPolicy
    from lagpilot.config import make_config
    from lagpilot.env import KafkaLagEnv

    cfg = make_config()
    env = KafkaLagEnv(cfg)
    seeds = range(1000, 1020)
    policies = {
        "keda": KedaRulePolicy(cfg),
        "static_max": StaticMaxPolicy(cfg),
        "random": RandomPolicy(seed=0),
    }
    for name, policy in policies.items():
        res = evaluate(env, policy, seeds)
        print(f"{name:11s} return={np.mean([r['return'] for r in res]):8.1f}"
              f"  delay={np.mean([r['avg_delay_s'] for r in res]):6.1f}s"
              f"  SLA viol={np.mean([r['sla_violation_pct'] for r in res]):5.1f}%"
              f"  consumers={np.mean([r['avg_consumers'] for r in res]):4.2f}"
              f"  rebalances={np.mean([r['rebalances'] for r in res]):5.1f}")
