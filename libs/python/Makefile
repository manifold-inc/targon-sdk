SRC := src

.PHONY: build test lint fmt clean

build:
	python -m build

test:
	@echo "no python tests defined yet"

lint:
	black --check $(SRC)
	isort --check-only $(SRC)
	flake8 $(SRC)
	mypy $(SRC)

fmt:
	isort $(SRC)
	black $(SRC)

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
