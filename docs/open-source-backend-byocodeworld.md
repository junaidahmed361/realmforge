# Open Source Backend: Bring Your Own CodeWorld (BYOC)

Worldforge supports enterprise deployment where customers keep code and telemetry in their own trust boundary.

## Goals
- Self-hostable backend components
- Pluggable SCM providers
- Evidence refs built from provider-native links
- No mandatory SaaS dependency

## Provider interface
Any backend provider implements:
- `listFiles(repo, ref)`
- `readFile(repo, path, ref)`
- `listCommits(repo, limit)`
- `buildEvidenceRef(repo, path, lineStart, lineEnd)`

## Built-in adapters (MVP)
- GitHub
- GitLab
- Bitbucket
- Gitea
- Local mirror (filesystem)

## Enterprise BYOC flow
1. Register provider config (token/base URL)
2. Ingest selected repo/ref
3. Build code entities + evidence refs
4. Run `/intent` to generate work units
5. Keep execution via internal CI/CD or GitHub PR bridge

## Security model
- Tokens are read from env only
- No token persistence in DB by default
- Evidence refs include only URI and optional excerpt
- Deploy with private Postgres + pgvector
