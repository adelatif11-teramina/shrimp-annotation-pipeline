# Repository Guidelines

## Project Structure & Module Organization
- `services/`: Python microservices (ingestion, candidates, rules, triage, automation, api); share DTOs/utilities from `shared/` before adding local copies.
- `shared/`: Ontology, JSON schemas, prompt templates; keep schema and prompt revisions synchronized with exporter and UI code.
- `ui/`: React app under `ui/src`; align API contracts with `services/api/mock_api.py` and update local mocks when endpoints change.
- `tests/`: Pytest suite mirroring service layout; fixtures in `tests/fixtures/`.
- `scripts/`, `config/`, `docs/`, `data/`: automation entry points, deployment config, annotator manuals, and working datasets. Keep generated files out of Git.

## Build, Test, and Development Commands
- `python scripts/setup_project.py`: bootstrap virtualenv, install requirements, seed `.env.local`.
- `docker-compose up -d`: start API, Postgres, Redis, and supporting services.
- `./start_local.sh`: launch mocked API plus React UI for fast UX work.
- `pytest` or `pytest -m "not slow"`: run unit/integration suites with 70% coverage gate.
- `./test_api.sh` and `./test_all_features.sh`: smoke-test HTTP endpoints against a running stack.

## Coding Style & Naming Conventions
- Use PEP 8 with 4-space indentation, type hints, and concise docstrings describing behavior, not implementation details.
- Format with `black` and lint with `flake8`; run both on touched modules before committing.
- React components use PascalCase files; hooks/utilities stay camelCase. Co-locate styles with the component.
- Branch names follow `feature/<topic>` or `fix/<topic>`; keep module names domain-focused (e.g., `triage_queue_service.py`).

## Testing Guidelines
- Name files `test_<target>.py`; mark async cases with `@pytest.mark.asyncio` and slow suites with `@pytest.mark.slow`.
- Mock external systems (LLM, Redis, Label Studio) via helpers in `tests/fixtures/`; cover success, failure, and edge-rule paths.
- Preserve the `--cov-report` outputs in CI, but avoid committing `htmlcov/`; investigate when coverage dips below 70%.

## Commit & Pull Request Guidelines
- Commits typically use Sentence case subjects without prefixes (see `git log`); keep them scoped and explain non-obvious changes in the body.
- Reference issues with `Refs #123` or `Fixes #123`; squash setup or experiment commits before opening a PR.
- PRs must outline behavior changes, reviewer checklist (tests run, scripts executed), and screenshots for UI adjustments.
- Call out required config updates (`.env` keys, migrations) and refresh related docs (`README.md`, `docs/`) when behavior shifts.
