lint: FORCE
	./scripts/lint.sh

lint-notebooks:
	./scripts/lint_notebooks.sh

format:
	./scripts/clean.sh

format-notebooks:
	./scripts/clean_notebooks.sh

test: lint FORCE
	./scripts/test.sh

test-notebooks: lint FORCE
	./scripts/test_notebooks.sh

FORCE:
