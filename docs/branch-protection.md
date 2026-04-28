# Branch protection recommendations

Recommended main branch rules:
- Require pull request before merging
- Require 1+ approvals
- Require status checks:
  - CI / lint-test (3.10)
  - CI / lint-test (3.11)
  - Security / security-scan
- Require branches up to date before merge
- Restrict force pushes and deletions
- Require signed commits (optional but recommended)
