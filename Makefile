# Makefile for Targon SDK development
.PHONY: help install install-dev build proto test test-cov format lint type-check check clean clean-proto

help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .

build:
	pip install -U build wheel
	python -m build

proto:
	@echo "Compiling protocol buffers."
	cd src/targon/proto && \
	python -m grpc_tools.protoc \
		-I. \
		--python_out=. \
		--grpc_python_out=. \
		--pyi_out=. \
		function_execution.proto
	@echo "Fixing imports in generated files."
	sed -i '' 's/^import function_execution_pb2/from . import function_execution_pb2/' src/targon/proto/function_execution_pb2_grpc.py
	@echo "Proto compilation complete."

test-cov:
	python -m pytest tests/ --cov=src/targon --cov-report=html --cov-report=term

format: 
	black src/ 

lint: 
	flake8 src/

type-check: 
	mypy src/

check: format lint type-check test 

clean: 
	@echo "Cleaning generated files."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaning complete."

clean-proto:
	rm -f src/targon/proto/*_pb2.py src/targon/proto/*_pb2_grpc.py src/targon/proto/*_pb2.pyi

