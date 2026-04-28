from __future__ import annotations

import asyncio
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from energy_graph.model import EnergyFactorGraph
from rollout.simulator import WorldModelSimulator
from transition.model import ActionConditionedTransition
from transition.outcome_heads import OutcomeHeads

app = FastAPI(title="HF-EBWM Serving")

latent_dim = 128
transition = ActionConditionedTransition(latent_dim=latent_dim, action_dim=7)
energy = EnergyFactorGraph(obs_dim=64, latent_dim=latent_dim)
outcomes = OutcomeHeads(latent_dim=latent_dim)
simulator = WorldModelSimulator(transition, energy, outcomes)

request_queue: asyncio.Queue = asyncio.Queue()


class SimRequest(BaseModel):
    concept: str
    horizon: int = 8
    n_samples: int = 64
    batch_size: int = 1


@app.on_event("startup")
async def startup_worker():
    asyncio.create_task(worker())


async def worker():
    while True:
        payload, fut = await request_queue.get()
        try:
            B = payload.batch_size
            S = max(1, payload.n_samples // max(B, 1))
            z0 = torch.randn(B, latent_dim)
            actions = torch.randint(0, 2, (B, S, payload.horizon, 7)).float()
            t0 = time.time()
            result = simulator.simulate_concept(
                z0, actions, horizon=payload.horizon, n_samples=payload.n_samples
            )
            latency = time.time() - t0
            fut.set_result(
                {
                    "mode": "world_model_rollout",
                    "concept": payload.concept,
                    "latency_sec": latency,
                    "trajectories": int(B * S),
                    "utility_topk": torch.topk(
                        result.utility_logits, k=min(5, result.utility_logits.numel())
                    ).values.tolist(),
                    "safety_notice": (
                        "Research/education only. Candidate trajectories, "
                        "not treatment recommendations."
                    ),
                }
            )
        except Exception as e:
            fut.set_result({"error": str(e)})


@app.post("/simulate")
async def simulate(req: SimRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put((req, fut))
    return await fut
