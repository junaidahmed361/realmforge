# @worldforge/api

Worldforge MVP backend API.

This service powers:
- BYOC repo ingestion
- intent -> impact surface generation
- trajectory + simulation report generation
- export of WorkUnit JSON for visualization

## Run locally

From repo root:

```bash
npm install
npm run dev
```

Server default:
- http://127.0.0.1:8787

## Endpoints

- `GET /health`
- `POST /repos/ingest`
- `POST /intent`
- `POST /demo/export-workunit`

## Minimal smoke test

```bash
curl -s http://127.0.0.1:8787/health | jq
```

For full demo payloads, use the root README.