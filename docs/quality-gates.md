# Quality gates

Run before commit:

1) pip install -e .[dev]
2) pre-commit run --all-files
3) ruff check .
4) ruff format --check .
5) mypy wm_app domains tests --ignore-missing-imports
6) pytest --cov=wm_app --cov-report=term-missing
7) bandit -r wm_app domains -c pyproject.toml
8) pip-audit

CI mirrors these via:
- .github/workflows/ci.yml
- .github/workflows/security.yml
- .github/workflows/release.yml
