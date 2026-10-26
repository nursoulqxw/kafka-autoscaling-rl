"""Run every experiment and save figures + CSVs to results/.

Usage:  python -m lagpilot.experiments

Every number in the report comes from the CSV files written here.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")   # no window, just save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lagpilot.agents import KedaRulePolicy, RandomPolicy, StaticMaxPolicy
from lagpilot.config import make_config
from lagpilot.env import ACTION_NAMES, KafkaLagEnv
from lagpilot.train import evaluate, run_episode, train_seed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CHECK_SEEDS = list(range(2000, 2010))   # greedy checks during training
METRICS = ["return", "avg_delay_s", "sla_violation_pct", "avg_consumers", "rebalances"]

# One fixed color per policy (same color in every figure).
COLORS = {
    "Q-learning": "#2a78d6",
    "KEDA rule": "#eb6834",
    "Static max": "#1baf7a",
    "Random": "#eda100",
    "Q-learning (no RPS)": "#6250d6",
}


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name), dpi=150)
    plt.close(fig)


def baselines(cfg):
    return {
        "KEDA rule": KedaRulePolicy(cfg),
        "Static max": StaticMaxPolicy(cfg),
        "Random": RandomPolicy(seed=0),
    }


def train_many(cfg, use_rps_feature=True):
    """Train one agent per training seed. Returns (agents, curves DataFrame)."""
    env = KafkaLagEnv(cfg, use_rps_feature=use_rps_feature)
    agents, rows = [], []
    for seed in cfg["TRAIN_SEEDS"]:
        agent, curve = train_seed(env, cfg, seed, CHECK_SEEDS)
        agents.append(agent)
        for episode, mean_return in curve:
            rows.append({"seed": seed, "episode": episode, "check_return": mean_return})
    return env, agents, pd.DataFrame(rows)


def eval_agents(env, agents, cfg):
    """Greedy evaluation of every trained agent on the evaluation seeds.
    Returns one row per (training seed, evaluation episode)."""
    rows = []
    for seed, agent in zip(cfg["TRAIN_SEEDS"], agents):
        for eval_seed, res in zip(cfg["EVAL_SEEDS"], evaluate(env, agent, cfg["EVAL_SEEDS"])):
            rows.append({"train_seed": seed, "eval_seed": eval_seed, **res})
    return pd.DataFrame(rows)


def eval_baseline(env, policy, cfg):
    rows = []
    for eval_seed, res in zip(cfg["EVAL_SEEDS"], evaluate(env, policy, cfg["EVAL_SEEDS"])):
        rows.append({"train_seed": -1, "eval_seed": eval_seed, **res})
    return pd.DataFrame(rows)


def summarize(df, policy):
    """Mean and std of every metric over all evaluation episodes.
    For Q-learning we also give the std of the per-seed mean return."""
    row = {"policy": policy, "episodes": len(df)}
    for m in METRICS:
        row[f"{m}_mean"] = df[m].mean()
        row[f"{m}_std"] = df[m].std()
    if df["train_seed"].max() >= 0:
        row["return_seed_std"] = df.groupby("train_seed")["return"].mean().std()
    else:
        row["return_seed_std"] = np.nan
    return row


def per_seed(df):
    """Mean of every metric for each training seed (one row per seed)."""
    return df.groupby("train_seed")[METRICS].mean().reset_index()


def plot_curves(ax, curves, label, color):
    stats = curves.groupby("episode")["check_return"].agg(["mean", "std"]).reset_index()
    ax.plot(stats["episode"], stats["mean"], color=color, lw=2, label=f"{label} (mean of 10 seeds)")
    ax.fill_between(stats["episode"], stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"], color=color, alpha=0.2, lw=0,
                    label=f"{label} ± 1 std")
    return stats


def main():
    start = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    cfg = make_config()

    # ------------------------------------------------------------------
    # Main run: default config, with the RPS feature (288 states)
    # ------------------------------------------------------------------
    print("Training Q-learning (default config, 288 states) ...")
    env, agents, curves = train_many(cfg)
    curves.to_csv(os.path.join(RESULTS, "fig1_training_curve.csv"), index=False)

    q_eval = eval_agents(env, agents, cfg)
    all_eval = [q_eval.assign(policy="Q-learning")]
    for name, policy in baselines(cfg).items():
        all_eval.append(eval_baseline(env, policy, cfg).assign(policy=name))
    all_eval = pd.concat(all_eval, ignore_index=True)
    all_eval.to_csv(os.path.join(RESULTS, "eval_episodes.csv"), index=False)
    order = ["Q-learning", "KEDA rule", "Static max", "Random"]
    table = pd.DataFrame([summarize(all_eval[all_eval.policy == p], p) for p in order])
    table.to_csv(os.path.join(RESULTS, "table_final.csv"), index=False)
    per_seed(q_eval).to_csv(os.path.join(RESULTS, "table_q_per_seed.csv"), index=False)
    print(table[["policy", "return_mean", "return_std", "avg_delay_s_mean",
                 "sla_violation_pct_mean", "avg_consumers_mean", "rebalances_mean"]]
          .round(2).to_string(index=False))

    # Figure 1: training curve with baselines as horizontal lines.
    # Baseline lines use the same check seeds as the curve.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_curves(ax, curves, "Q-learning", COLORS["Q-learning"])
    base_check = {}
    for name, policy in baselines(cfg).items():
        base_check[name] = np.mean([r["return"] for r in evaluate(env, policy, CHECK_SEEDS)])
        ax.axhline(base_check[name], color=COLORS[name], lw=1.5, ls="--", label=name)
    pd.DataFrame([{"policy": k, "check_return": v} for k, v in base_check.items()]).to_csv(
        os.path.join(RESULTS, "fig1_baselines.csv"), index=False)
    eps_end = cfg["EPS_DECAY_FRACTION"] * cfg["TRAIN_EPISODES"]
    ax.axvline(eps_end, color="#888888", lw=1, ls=":")
    ax.text(eps_end, ax.get_ylim()[1], " ε reaches 0.05", color="#555555", va="top", fontsize=8)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy return per episode (higher is better)")
    ax.set_title("Fig 1. Q-learning training curve (greedy check on 10 seeds every 50 episodes)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    save_fig(fig, "fig1_training_curve.png")

    # Figure 2: final comparison, 5 metrics.
    labels = {"return": "Return", "avg_delay_s": "Average delay (s)",
              "sla_violation_pct": "SLA violation (% of steps)",
              "avg_consumers": "Average consumers", "rebalances": "Rebalances per episode"}
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, m in zip(axes, METRICS):
        ax.bar(range(len(order)), table[f"{m}_mean"], yerr=table[f"{m}_std"], capsize=4,
               color=[COLORS[p] for p in order], edgecolor="white", linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(labels[m], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_title("Average delay (s), log scale", fontsize=10)
    fig.suptitle("Fig 2. Final greedy evaluation on seeds 1000-1049 (mean ± std over episodes)")
    save_fig(fig, "fig2_final_comparison.png")

    # ------------------------------------------------------------------
    # Save Q-table of seed 0 and one real transition example
    # ------------------------------------------------------------------
    np.save(os.path.join(RESULTS, "q_table.npy"), agents[0].q)

    # Figure 4 + transition example: one evaluation episode.
    # We use the first evaluation seed that contains a spike, so the trace
    # shows how each policy reacts to a flash sale.
    trace_seed = None
    for s in cfg["EVAL_SEEDS"]:
        env.reset(s)
        if env.rps.max() > cfg["RPS_PEAK"] * 1.2:
            trace_seed = s
            break
    traces = {}
    for name, policy in [("Q-learning", agents[0]), ("KEDA rule", KedaRulePolicy(cfg))]:
        res = run_episode(env, policy, trace_seed, log=True)
        traces[name] = pd.DataFrame(res["steps"]).assign(policy=name)
    trace_df = pd.concat(traces.values(), ignore_index=True)
    trace_df.to_csv(os.path.join(RESULTS, "fig4_episode_trace.csv"), index=False)

    q_steps = traces["Q-learning"]
    started = q_steps[q_steps.rebalance_started]
    example = (started.iloc[0] if len(started) else q_steps.iloc[60]).to_dict()
    transition = {
        "eval_seed": trace_seed,
        "step": int(example["t"]),
        "state_id": int(example["state"]),
        "state": env.decode(int(example["state"])),
        "action": ACTION_NAMES[int(example["action"])],
        "reward": round(float(example["reward"]), 3),
        "next_state_id": int(example["next_state"]),
        "next_state": env.decode(int(example["next_state"])),
        "info": {k: (v.item() if hasattr(v, "item") else v) for k, v in example.items()
                 if k in ["rps", "arrivals", "processed", "lag", "delay_s", "consumers",
                          "rebalancing", "rebalance_started", "sla_violated"]},
        "q_values_in_state": dict(zip(ACTION_NAMES,
                                      [round(float(x), 2) for x in agents[0].q[int(example["state"])]])),
    }
    with open(os.path.join(RESULTS, "transition_example.json"), "w") as f:
        json.dump(transition, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t_sec = q_steps["t"] * cfg["STEP_SECONDS"]
    axes[0].plot(t_sec, q_steps["rps"], color="#52514e", lw=2)
    axes[0].set_ylabel("Incoming RPS")
    axes[0].axhline(cfg["MAX_CONSUMERS"] * cfg["MU"], color="#52514e", ls=":", lw=1)
    axes[0].text(0, cfg["MAX_CONSUMERS"] * cfg["MU"], " max capacity (4 consumers)", fontsize=8,
                 va="bottom", color="#52514e")
    axes[0].set_title(f"Fig 4. One evaluation episode (seed {trace_seed}): Q-learning vs KEDA")
    for name, df in traces.items():
        axes[1].plot(t_sec, df["delay_s"], color=COLORS[name], lw=2, label=name)
        axes[2].step(t_sec, df["consumers"], where="post", color=COLORS[name], lw=2, label=name)
        reb = df["rebalancing"].to_numpy()
        for i in np.flatnonzero(reb):
            axes[2].axvspan(t_sec.iloc[i], t_sec.iloc[i] + cfg["STEP_SECONDS"],
                            color=COLORS[name], alpha=0.12, lw=0)
    axes[1].axhline(cfg["SLA_SECONDS"], color="#e34948", ls="--", lw=1.5, label="SLA = 30 s")
    axes[1].set_ylabel("Delay (s)")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel("Consumers")
    axes[2].set_yticks([1, 2, 3, 4])
    axes[2].set_xlabel("Time in episode (s); shaded = rebalance in progress")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    save_fig(fig, "fig4_episode_trace.png")

    # ------------------------------------------------------------------
    # Figure 3: rebalance duration study, R in {1, 2, 3}
    # ------------------------------------------------------------------
    rows, seed_rows = [], []
    for R in [1, 2, 3]:
        print(f"Rebalance study: R = {R} ...")
        cfg_r = make_config(REBALANCE_STEPS=R)
        if R == cfg["REBALANCE_STEPS"]:
            env_r, agents_r = env, agents      # reuse the main run
        else:
            env_r, agents_r, _ = train_many(cfg_r)
        q_df = eval_agents(env_r, agents_r, cfg_r)
        rows.append({"R": R, **summarize(q_df, "Q-learning")})
        seed_rows.append(per_seed(q_df).assign(R=R))
        for name in ["KEDA rule", "Static max"]:
            b_df = eval_baseline(env_r, baselines(cfg_r)[name], cfg_r)
            rows.append({"R": R, **summarize(b_df, name)})
    reb_table = pd.DataFrame(rows)
    reb_table.to_csv(os.path.join(RESULTS, "fig3_rebalance_study.csv"), index=False)
    pd.concat(seed_rows).to_csv(os.path.join(RESULTS, "fig3_q_per_seed.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    width = 0.27
    names = ["Q-learning", "KEDA rule", "Static max"]
    for ax, m, title in zip(axes, ["return", "rebalances", "sla_violation_pct"],
                            ["Return (higher is better)", "Rebalances per episode",
                             "SLA violation (% of steps)"]):
        for i, name in enumerate(names):
            d = reb_table[reb_table.policy == name]
            ax.bar(d["R"] + (i - 1) * width, d[f"{m}_mean"], width, yerr=d[f"{m}_std"],
                   capsize=3, color=COLORS[name], label=name, edgecolor="white", linewidth=1.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["R=1 (10 s)", "R=2 (20 s)", "R=3 (30 s)"])
        ax.set_xlabel("Rebalance duration")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 3. Effect of rebalance duration (Q-learning retrained for each R, 10 seeds)")
    save_fig(fig, "fig3_rebalance_study.png")

    # ------------------------------------------------------------------
    # Figure 5: with vs without the rps_level feature
    # ------------------------------------------------------------------
    print("Training Q-learning without rps_level (96 states) ...")
    env_n, agents_n, curves_n = train_many(cfg, use_rps_feature=False)
    curves_both = pd.concat([curves.assign(variant="with RPS (288 states)"),
                             curves_n.assign(variant="no RPS (96 states)")])
    curves_both.to_csv(os.path.join(RESULTS, "fig5_rps_feature_curves.csv"), index=False)
    n_eval = eval_agents(env_n, agents_n, cfg)
    rps_table = pd.DataFrame([summarize(q_eval, "Q-learning"),
                              summarize(n_eval, "Q-learning (no RPS)")])
    rps_table.to_csv(os.path.join(RESULTS, "fig5_rps_feature.csv"), index=False)
    pd.concat([per_seed(q_eval).assign(variant="with RPS"),
               per_seed(n_eval).assign(variant="no RPS")]).to_csv(
        os.path.join(RESULTS, "fig5_q_per_seed.csv"), index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    plot_curves(axes[0], curves, "with RPS", COLORS["Q-learning"])
    plot_curves(axes[0], curves_n, "no RPS", COLORS["Q-learning (no RPS)"])
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Greedy return per episode")
    axes[0].set_title("Learning curves", fontsize=10)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(alpha=0.3)
    variants = ["Q-learning", "Q-learning (no RPS)"]
    for ax, m, title in zip(axes[1:], ["return", "sla_violation_pct", "rebalances"],
                            ["Final return", "SLA violation (% of steps)", "Rebalances per episode"]):
        ax.bar([0, 1], rps_table[f"{m}_mean"], yerr=rps_table[f"{m}_std"], capsize=4,
               color=[COLORS[v] for v in variants], edgecolor="white", linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["with RPS\n(288 states)", "no RPS\n(96 states)"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig 5. Does observing the incoming RPS level help? (10 seeds each)")
    save_fig(fig, "fig5_rps_feature.png")

    # ------------------------------------------------------------------
    # Figure 6: greedy policy heatmap (seed 0)
    # consumers = 2, trend = stable, rebalancing = 0
    # ------------------------------------------------------------------
    grid = np.zeros((4, 3), dtype=int)
    rows = []
    for d in range(4):
        for r in range(3):
            s = env.encode([d, 1, r, 2 - 1, 0])
            grid[d, r] = agents[0].greedy_action(s)
            rows.append({"delay_level": d, "rps_level": r, "action": ACTION_NAMES[grid[d, r]],
                         "visited": bool(np.any(agents[0].q[s] != 0))})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fig6_policy_heatmap.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = matplotlib.colors.ListedColormap(["#d9d9d6", COLORS["Q-learning"], COLORS["KEDA rule"]])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
    for d in range(4):
        for r in range(3):
            visited = rows[d * 3 + r]["visited"]
            text = ACTION_NAMES[grid[d, r]] + ("" if visited else "\n(not visited)")
            ax.text(r, d, text, ha="center", va="center", fontsize=8,
                    color="black" if grid[d, r] == 0 else "white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low\n(<40)", "medium\n(40-90)", "high\n(≥90)"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["low (<5 s)", "medium (<15 s)", "high (<30 s)", "critical (≥30 s)"])
    ax.set_xlabel("rps_level (incoming RPS)")
    ax.set_ylabel("delay_level")
    ax.set_title("Fig 6. Greedy action, seed 0\n(consumers = 2, lag stable, no rebalance)", fontsize=10)
    save_fig(fig, "fig6_policy_heatmap.png")

    print(f"Done. Results in {RESULTS}")
    print(f"Total runtime: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
