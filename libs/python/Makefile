SRC := src

.PHONY: install install-dev build test lint fmt clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

build:
	pip install -U build wheel
	python -m build

test:
	@echo "no python tests defined yet"

lint:
	black --check $(SRC)
	isort --check-only $(SRC)
	flake8 $(SRC)
fmt:
	isort $(SRC)
	black $(SRC)

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
