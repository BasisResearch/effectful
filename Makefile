lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test: lint FORCE
	./scripts/test.sh

test-notebooks: lint FORCE
	./scripts/test_notebooks.sh

FORCE:
