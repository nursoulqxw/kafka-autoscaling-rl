"""Live demo: the trained Q-table scales real Kafka consumers.

Every 10 s it:
  1. reads the real lag (end offsets minus committed offsets),
  2. reads the incoming RPS (growth of end offsets per second),
  3. builds the same discrete state as the simulator,
  4. picks argmax Q from results/q_table.npy,
  5. adds or removes a consumer thread (a real rebalance),
  6. prints one log line.

Run from the project root:  python -m demo.agent
The demo only shows the trained policy on real Kafka.
All reported results come from the simulator.
"""

import os
import time

import numpy as np
from confluent_kafka import Consumer, TopicPartition

from demo.consumer_manager import BOOTSTRAP, GROUP, TOPIC, ConsumerManager
from lagpilot.config import make_config
from lagpilot.env import ACTION_NAMES, KafkaLagEnv, SCALE_DOWN, SCALE_UP

PARTITIONS = 4


def read_offsets(admin_consumer):
    """Return (sum of end offsets, sum of committed offsets) over all partitions."""
    parts = [TopicPartition(TOPIC, p) for p in range(PARTITIONS)]
    end_total = 0
    for tp in parts:
        _, high = admin_consumer.get_watermark_offsets(tp, timeout=5)
        end_total += high
    committed_total = 0
    for tp in admin_consumer.committed(parts, timeout=5):
        committed_total += max(tp.offset, 0)   # -1001 means "nothing committed yet"
    return end_total, committed_total


def main():
    cfg = make_config()
    env = KafkaLagEnv(cfg)          # used only for encode / decode
    q = np.load(os.path.join(os.path.dirname(__file__), "..", "results", "q_table.npy"))
    step = cfg["STEP_SECONDS"]
    rebalance_window = cfg["REBALANCE_STEPS"] * step

    # A consumer that never subscribes: it only reads offsets of the group.
    admin_consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": GROUP,
                               "enable.auto.commit": False})
    manager = ConsumerManager(mu=cfg["MU"], start=cfg["START_CONSUMERS"])
    prev_end, _ = read_offsets(admin_consumer)
    prev_lag = 0
    last_change = -1e9

    try:
        while True:
            time.sleep(step)
            end, committed = read_offsets(admin_consumer)
            lag = max(end - committed, 0)
            rps = (end - prev_end) / step
            prev_end = end
            c = manager.count

            # Same features as KafkaLagEnv._features()
            delay_s = lag / (c * cfg["MU"])
            delay_level = int(np.searchsorted(cfg["DELAY_EDGES"], delay_s, side="right"))
            band = max(cfg["TREND_REL_BAND"] * prev_lag, cfg["TREND_ABS_BAND"])
            trend = 2 if lag - prev_lag > band else (0 if lag - prev_lag < -band else 1)
            rps_level = int(np.searchsorted(cfg["RPS_EDGES"], rps, side="right"))
            # Simplification: we call it "rebalancing" for R steps after a change.
            rebalancing = 1 if time.time() - last_change < rebalance_window else 0
            state = env.encode([delay_level, trend, rps_level, c - 1, rebalancing])
            prev_lag = lag

            action = int(np.argmax(q[state]))
            if rebalancing:
                action = 0      # same rule as the simulator: no scaling during a rebalance
            note = ""
            if action == SCALE_UP and c < cfg["MAX_CONSUMERS"]:
                manager.add()
                last_change = time.time()
            elif action == SCALE_DOWN and c > cfg["MIN_CONSUMERS"]:
                manager.remove()
                last_change = time.time()
            elif action != 0:
                note = " (invalid here, acts as keep)"

            print(f"RPS={rps:6.1f}  lag={lag:6d}  delay={delay_s:6.1f}s  "
                  f"state={env.decode(state)}  action={ACTION_NAMES[action]}{note}  "
                  f"consumers={manager.count}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
        admin_consumer.close()


if __name__ == "__main__":
    main()
