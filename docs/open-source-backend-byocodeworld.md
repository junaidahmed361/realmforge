# Open Source Backend: Bring Your Own CodeWorld (BYOC)

This document mirrors the current Worldforge MVP behavior in this repo.

## What BYOC means here

Enterprise teams can run the backend in their own trust boundary and point Worldforge at their own source control.

Supported provider values:
- `github`
- `gitlab`
- `bitbucket`
- `gitea`
- `local`

Provider adapters are in:
- `apps/api/src/providers/`

## API flow (practical)

1) Ingest repository files:
- `POST /repos/ingest`

2) Generate an intent work unit:
- `POST /intent`

3) Export work unit JSON for visualization:
- `POST /demo/export-workunit`

Health:
- `GET /health`

## Minimal demo payloads

### Ingest

```json
{
  "provider": "github",
  "owner": "pytorch",
  "repo": "examples",
  "ref": "main"
}
```

### Intent

```json
{
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
}
```

## WorkUnit shape (current MVP)

The backend currently emits:
- `impactSurface.entities[]`
  - `id`
  - `world`
  - `type`
  - `name`
  - `confidence`
  - `trajectoryIds[]` (explicit entity -> trajectory mapping)
  - `plannedActions[]` (entity-level practical actions)
  - `evidenceRefs[]` (provider-native links)
- `trajectories[]`
- `simulationReports[]`
- `calibrationProfile`

This explicit mapping powers node-level unpacking in the visualization repo.

## Provider interface

Each provider implements:
- `listFiles(repo, ref)`
- `readFile(repo, path, ref)`
- `listCommits(repo, limit)`
- `buildEvidenceRef(repo, path, lineStart, lineEnd)`

## Security model (MVP)

- Credentials/tokens are provided via environment variables.
- No mandatory external SaaS dependency for core API operation.
- Evidence refs are URIs to source evidence (plus optional metadata).

## Local mode example

Use `provider: "local"` with `localPath`:

```json
{
  "repo": {
    "provider": "local",
    "repo": "public-pytorch-examples-demo",
    "localPath": "/Users/junaidahmed/Downloads/public-pytorch-examples-demo"
  },
  "intent": "Prioritize safe migration steps with measurable reliability gains."
}
```

For runnable curl examples, use the root README as the canonical quickstart.
