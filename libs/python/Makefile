SRC := src
TESTS := tests

.PHONY: install install-dev build test lint fmt clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

build:
	rm -rf build dist $(SRC)/*.egg-info
	pip install -U build wheel
	python -m build

test:
	PYTHONPATH=$(SRC) python -m pytest $(TESTS)

lint:
	black --check $(SRC) $(TESTS)
	isort --check-only $(SRC) $(TESTS)
	flake8 $(SRC) $(TESTS)
fmt:
	isort $(SRC) $(TESTS)
	black $(SRC) $(TESTS)

clean:
	rm -rf build dist *.egg-info $(SRC)/*.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
