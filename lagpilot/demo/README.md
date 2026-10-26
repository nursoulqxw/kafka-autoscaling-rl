# Optional live demo: LagPilot on real Kafka

**The demo shows the trained policy running on real Kafka. All reported
results come from the simulator.**

## What runs

| File | Role |
|---|---|
| `docker-compose.yml` | one Kafka broker in KRaft mode (no ZooKeeper) |
| `producer_api.py` | FastAPI app, `POST /orders` writes one message to topic `orders` (4 partitions, created at startup) |
| `load.py` | sends orders at a chosen RPS: 20 → 60 → 120 → spike 300 → 120 → 60 → 20 |
| `consumer_manager.py` | runs N consumer threads in group `lagpilot-group`; each handles about 50 msg/s (μ from `lagpilot/config.py`) |
| `agent.py` | every 10 s: reads real lag and RPS, builds the same state as the simulator, picks argmax Q from `results/q_table.npy`, adds or removes a consumer (real rebalance), prints one log line |

`sample_run.log` is the agent output of one real run of the full load scenario
(recorded on 26 October 2026, before the "(invalid here, acts as keep)" note
was added to the log line).

## Run (4 terminals, from the project root)

```bash
pip install -r demo/requirements.txt
docker compose -f demo/docker-compose.yml up -d     # 1. Kafka
uvicorn demo.producer_api:app --port 8000           # 2. API (creates the topic)
python -m demo.agent                                # 3. agent + consumers
python demo/load.py                                 # 4. traffic
```

Stop with Ctrl+C, then `docker compose -f demo/docker-compose.yml down`.

## Simplifications

- "rebalancing" in the state is set for R × 10 s after each change; we do not
  read the real group state.
- Lag is end offsets minus committed offsets. Offsets are auto-committed every
  second, so lag can be up to one second too high.
- A single load generator on a laptop may not reach exactly 300 RPS.
