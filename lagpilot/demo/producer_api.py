"""Tiny FastAPI app: POST /orders writes one message to the Kafka topic "orders".

Run:  uvicorn demo.producer_api:app --port 8000   (from the project root)
"""

import json
import time

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from fastapi import FastAPI

BOOTSTRAP = "localhost:9092"
TOPIC = "orders"
PARTITIONS = 4

app = FastAPI()
producer = Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 5})


@app.on_event("startup")
def create_topic():
    """Create the topic with 4 partitions if it does not exist yet."""
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    if TOPIC not in admin.list_topics(timeout=10).topics:
        admin.create_topics([NewTopic(TOPIC, num_partitions=PARTITIONS, replication_factor=1)])
        time.sleep(2)


@app.post("/orders")
def create_order(order: dict):
    order["created_at"] = time.time()
    producer.produce(TOPIC, json.dumps(order).encode())
    producer.poll(0)   # serve delivery callbacks, do not block
    return {"status": "queued"}
