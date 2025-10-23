.PHONY: lint test

lint:
	@echo "Linting gwassess code"
	@python3 -m flake8 gwassess tests example_usage.py

test:
	@echo "Running unit tests"
	@python3 -m pytest tests/
