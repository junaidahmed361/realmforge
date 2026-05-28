# Worldforge

**Control the future of your system, not just its code.**

---

## Why Worldforge Exists

Modern software development is fragmented across tools:

- GitHub tracks code
- Datadog tracks metrics
- Amplitude tracks users
- Jira tracks tasks
- Notion tracks decisions

None of these systems answer a fundamental question:

> *What will happen if we change something?*

Developers operate reactively:

- write code
- run tests
- fix failures
- deploy
- observe impact

This loop is inefficient, local, and blind to broader consequences.

---

## The Problem

Even with LLM agents:

- agents operate locally on code
- they lack awareness of business goals
- they cannot simulate long-term impact
- they do not align across systems

World models exist, but are:

- primitive
- single-domain
- disconnected from real software systems

There is no system that connects:

```text
code → runtime → users → business outcomes
```

---

## The Worldforge Approach

Worldforge builds a **multi-world model** of your system:

- Code World
- Runtime World
- Business World
- User World
- Test World
- Knowledge World

It forges them into a single belief graph.

Instead of editing code directly, you:

1. State an intent
2. See impacted parts of the system
3. Compare possible futures
4. Adjust tradeoffs
5. Approve a trajectory

Worldforge then generates the implementation.

---

## What Makes It Different

Worldforge does not optimize for writing code faster.

It optimizes for:

- choosing the right changes
- understanding system-wide impact
- reducing unintended consequences
- aligning engineering with business goals

GitHub tells you what changed.

Worldforge tells you what will happen.

---

## Core Concepts

- **Intent**: What you want to achieve
- **Impact Surface**: What parts of the system are affected
- **Trajectory**: A candidate path to the goal
- **Simulation**: Predicted outcome across code, runtime, and business
- **Calibration**: User-defined tradeoffs
- **Work Unit**: A semantic change ready for execution
- **Evidence**: Code, metrics, and docs supporting each claim

---

## Current State

Worldforge is an early-stage system.

The MVP focuses on:

- code ingestion
- impact surface generation
- trajectory planning
- evidence-backed work units

Future versions will integrate:

- runtime telemetry
- business metrics
- causal inference
- multi-world simulation

---

## Philosophy

Software development should not be about navigating diffs.

It should be about:

- defining outcomes
- exploring possibilities
- understanding tradeoffs
- selecting futures

Worldforge exists to make that possible.

---

## Getting Started (Planned)

```bash
pnpm install
pnpm dev
```

Then:

1. Connect a GitHub repo
2. Enter an intent
3. Explore impact
4. Approve a trajectory
5. Generate a PR

---

## Open-source backend: Bring Your Own CodeWorld (BYOC)

Enterprise teams can run Worldforge with their own source-control system and data boundary.

Current open backend adapters include:

- GitHub
- GitLab
- Bitbucket
- Gitea
- Local filesystem repo mirror

See `docs/open-source-backend-byocodeworld.md` and `apps/api/src/providers/` for extension points.

---

## Vision

Worldforge is a step toward simulation-native software development.

A world where:

- agents reason before acting
- systems are understood holistically
- changes are chosen, not guessed

---

**Build the future before you commit to it.**
