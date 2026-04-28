# Supply Chain Mini Realm (Trivial Real-World Example)

This example shows a simple, realistic planning setup: a small warehouse deciding daily replenishment actions under demand uncertainty.

Why this is real-world applicable:
- maps directly to inventory planning and stockout prevention
- uses candidate action sequences (order none/small/medium)
- ranks scenarios by utility to compare policy choices

It is still intentionally lightweight and synthetic (no external data dependency).

## Run

```bash
uv run python examples/supply-chain-mini/run.py
```

## What it prints
- candidate policy names
- utility score per policy (higher is better)
- best policy under this sampled demand context
