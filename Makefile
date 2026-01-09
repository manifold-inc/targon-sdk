# Makefile for Targon SDK development
.PHONY: proto install install-dev test format lint type-check clean 

help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

proto:
	@echo "Compiling protocol buffers..."
	cd src/targon/proto && \
	python -m grpc_tools.protoc \
		-I. \
		--python_out=. \
		--grpc_python_out=. \
		--pyi_out=. \
		function_execution.proto
	@echo "Fixing imports in generated files..."
	sed -i '' 's/^import function_execution_pb2/from . import function_execution_pb2/' src/targon/proto/function_execution_pb2_grpc.py
	@echo "✓ Proto compilation complete"

test-cov:
	python -m pytest tests/ --cov=src/targon --cov-report=html --cov-report=term

format: ## Format code with black
	black src/ 

lint: ## Run flake8 linter
	flake8 src/

type-check: ## Run mypy type checking
	mypy src/

check: format lint type-check test ## Run all checks

clean: ## Clean build artifacts
	@echo "Cleaning generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*_pb2.py" -delete
	find . -type f -name "*_pb2_grpc.py" -delete
	find . -type f -name "*_pb2.pyi" -delete
	@echo "✓ Clean complete"

