# Worldforge (MVP)

Worldforge is an intent-to-impact backend that helps teams reason about change before implementation.

This repo currently contains a practical MVP:
- BYOC repo ingestion (GitHub/GitLab/Bitbucket/Gitea/local)
- Intent parsing + heuristic planning
- WorkUnit generation with trajectories + simulation reports
- Entity-to-trajectory/action mapping for UI node-unpack drilldown

If older RealmForge concepts conflict with this flow, treat this README as the source of truth for current usage.

## 1) Quick start

Requirements:
- Node.js 20+
- npm (or pnpm)

Install and run:

```bash
cd /Users/junaidahmed/Downloads/realmforge
npm install
npm run dev
```

API starts on:
- http://127.0.0.1:8787

Health check:

```bash
curl -s http://127.0.0.1:8787/health | jq
```

## 2) Practical demo (copy/paste)

### A) Ingest a public repo

```bash
curl -s -X POST http://127.0.0.1:8787/repos/ingest \
  -H 'content-type: application/json' \
  -d '{
    "provider": "github",
    "owner": "pytorch",
    "repo": "examples",
    "ref": "main"
  }' | jq
```

### B) Generate a WorkUnit from a business goal

```bash
curl -s -X POST http://127.0.0.1:8787/intent \
  -H 'content-type: application/json' \
  -d '{
    "repo": {
      "provider": "github",
      "owner": "pytorch",
      "repo": "examples",
      "ref": "main"
    },
    "intent": "Reduce TensorFlow-to-PyTorch migration churn for enterprise teams while improving reliability and p95 latency.",
    "calibration": {
      "riskTolerance": 0.35,
      "confidenceThreshold": 0.68
    }
  }' | tee /tmp/worldforge-workunit.json | jq
```

What to look for in output:
- `impactSurface.entities[]`
- `trajectories[]`
- `simulationReports[]`
- `impactSurface.entities[].trajectoryIds`
- `impactSurface.entities[].plannedActions`

### C) Export WorkUnit JSON for visualization

```bash
curl -s -X POST http://127.0.0.1:8787/demo/export-workunit \
  -H 'content-type: application/json' \
  -d "$(jq -c '{workUnit:.}' /tmp/worldforge-workunit.json)" | jq
```

This returns a file path under `realmforge/exports/`.

## 3) BYOC backend (enterprise)

Supported providers now:
- `github`
- `gitlab`
- `bitbucket`
- `gitea`
- `local`

Provider adapters live in:
- `apps/api/src/providers/`

BYOC backend details:
- `docs/open-source-backend-byocodeworld.md`

## 4) Local repo mode example

```bash
curl -s -X POST http://127.0.0.1:8787/intent \
  -H 'content-type: application/json' \
  -d '{
    "repo": {
      "provider": "local",
      "repo": "public-pytorch-examples-demo",
      "localPath": "/Users/junaidahmed/Downloads/public-pytorch-examples-demo"
    },
    "intent": "Prioritize safe migration steps with measurable reliability gains.",
    "calibration": {
      "riskTolerance": 0.25,
      "confidenceThreshold": 0.7
    }
  }' | jq
```

## 5) Current scope

This is an MVP backend for intent planning and explainable impact simulation output.
Frontend visualization lives in a separate repo and consumes exported WorkUnit JSON.
