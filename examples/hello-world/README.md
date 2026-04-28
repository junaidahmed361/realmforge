# RealmForge Hello World

This is a minimal, synthetic example that runs a tiny counterfactual rollout without any domain dataset.

What it demonstrates:
- creating an initial latent state (`z0`)
- generating candidate action trajectories
- simulating futures with RealmForge transition + energy + outcome modules
- ranking scenarios by utility logit

## Run

From repo root:

```bash
uv run python examples/hello-world/run.py
```

## Expected output

You should see:
- latent trajectory tensor shape
- utility logits shape
- top 3 scenario indices by utility

This example is intentionally data-free and safe for first-time users.
