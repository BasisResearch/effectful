.PHONY: lint format test test-notebooks rebuild-fixtures FORCE

lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test: lint FORCE
	./scripts/test.sh

test-notebooks: lint FORCE
	./scripts/test_notebooks.sh

rebuild-fixtures:
	REBUILD_FIXTURES=true uv run pytest tests/test_handlers_llm_provider.py

FORCE:
