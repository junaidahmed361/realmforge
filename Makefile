.PHONY: install-dev precommit lint format-check typecheck test security ci

install-dev:
	python -m pip install --upgrade pip
	pip install -e .[dev]
	pre-commit install

precommit:
	pre-commit run --all-files

lint:
	ruff check .

format-check:
	ruff format --check .

typecheck:
	mypy app realms tests --ignore-missing-imports

test:
	pytest --cov=app/wm_app --cov-report=term-missing

security:
	bandit -r app realms -c pyproject.toml
	python -m pip install pip-audit && pip-audit

release-rc:
	python -m build
	@echo "Create git tag: v0.1.2"

ci: lint format-check typecheck test security
