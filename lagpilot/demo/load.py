"""Load generator: sends POST /orders at a chosen RPS.

Scenario: 20 RPS -> 120 RPS -> spike to 300 RPS -> back down.
Run:  python demo/load.py
"""

import asyncio
import time

import httpx

URL = "http://localhost:8000/orders"
# (RPS, duration in seconds)
SCENARIO = [(20, 60), (60, 40), (120, 60), (300, 40), (120, 40), (60, 40), (20, 60)]


async def send(client, i):
    try:
        await client.post(URL, json={"order_id": i})
    except httpx.HTTPError as e:
        print("request failed:", e)


async def main():
    i = 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as client:
        for rps, duration in SCENARIO:
            print(f"load: {rps} RPS for {duration} s")
            end = time.time() + duration
            while time.time() < end:
                start = time.time()
                # Send one second of requests, then wait for the rest of the second.
                await asyncio.gather(*(send(client, i + k) for k in range(rps)))
                i += rps
                await asyncio.sleep(max(0.0, 1.0 - (time.time() - start)))


if __name__ == "__main__":
    asyncio.run(main())
