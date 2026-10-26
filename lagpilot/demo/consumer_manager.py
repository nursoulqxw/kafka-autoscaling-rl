"""Runs N consumer threads in the group "lagpilot-group".

Each thread is one Kafka consumer. Adding or removing a thread makes the
group rebalance (a real Kafka rebalance, not a simulation).
Each thread sleeps 1 / MU seconds per message to simulate a consumer that
can process about MU messages per second (MU from lagpilot/config.py).
"""

import threading
import time

from confluent_kafka import Consumer

BOOTSTRAP = "localhost:9092"
TOPIC = "orders"
GROUP = "lagpilot-group"


class ConsumerThread(threading.Thread):
    def __init__(self, name, mu):
        super().__init__(name=name, daemon=True)
        self.mu = mu
        self.stop_event = threading.Event()

    def run(self):
        consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 1000,
            # Eager rebalance, like in the simulator.
            "partition.assignment.strategy": "range",
        })
        consumer.subscribe([TOPIC])
        while not self.stop_event.is_set():
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            time.sleep(1.0 / self.mu)   # "process" the order
        consumer.close()                # leaving the group triggers a rebalance


class ConsumerManager:
    def __init__(self, mu, start=2):
        self.mu = mu
        self.threads = []
        for _ in range(start):
            self.add()

    @property
    def count(self):
        return len(self.threads)

    def add(self):
        t = ConsumerThread(f"consumer-{len(self.threads) + 1}", self.mu)
        t.start()
        self.threads.append(t)

    def remove(self):
        t = self.threads.pop()
        t.stop_event.set()

    def stop_all(self):
        while self.threads:
            self.remove()
