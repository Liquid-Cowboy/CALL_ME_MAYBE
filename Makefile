VENV :=.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

FLAKE8_FLAGS := --per-file-ignores=__init__.py:F401

MYPY_FLAGS := --warn-return-any \
--warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs

install:
	@ echo "Installing dependencies..."
	@ uv sync
run:
	@ echo "⚙️ Running program..."
	@ uv run $(PYTHON) -m src

debug:
	@ $(PYTHON) -m pdb srcs/main.py

clean:
	@echo "🧹 Cleaning files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete

lint:
	@echo "🔎 Running linter..."
	@ $(VENV)/bin/flake8 src $(FLAKE8_FLAGS)\
	&& $(VENV)/bin/mypy src $(MYPY_FLAGS)
	@echo "✅ No linting issues detected!"

lint-strict:
	@echo "🔎 Running strict linter..."
	@ $(VENV)/bin/flake8 src $(FLAKE8_FLAGS) && \
	$(VENV)/bin/mypy src --exclude '.venv' --strict 
	@echo "✅ No linting issues detected!"

tests:
	@uv run python -m pytest -v -s

clean_output: clean
	@find ./data/ -type d -name "output" -exec rm -rf {} +

.PHONY: install run debug clean lint lint-strict tests