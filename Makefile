.PHONY: setup data train eval report app test lint clean all format

VENV ?= .venv
PYTHON := PYTHONPATH=src $(VENV)/bin/python
UV ?= $(shell which uv 2>/dev/null || echo /Users/krishnanamdev0101gmail.com/.local/bin/uv)

# Default target
all: setup lint test

# Install dependencies into virtualenv
setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(UV) venv --python 3.11 $(VENV); \
	fi
	$(UV) pip install -e ".[dev]"
	@echo "✅ Setup complete. Virtual environment ready in $(VENV)."

# Data pipeline: ingest → bronze → silver → gold
data:
	$(PYTHON) -m complaints.data.ingest
	$(PYTHON) -m complaints.data.clean
	$(PYTHON) -m complaints.data.sample
	$(PYTHON) -m complaints.data.split
	$(PYTHON) -m complaints.data.quality

# Train all classical and embedding models
train:
	$(PYTHON) -m complaints.models.tfidf_lr
	$(PYTHON) -m complaints.models.tfidf_svm
	$(PYTHON) -m complaints.models.embed_lgbm

# Run evaluation and generate reports
eval:
	$(PYTHON) -m complaints.eval.report

# Generate all reports
report:
	$(PYTHON) -m complaints.eval.report
	$(PYTHON) -m complaints.eval.gate

# Run the Streamlit app
app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

# Run tests
test:
	$(VENV)/bin/pytest tests/ -v --tb=short

# Lint and type check
lint:
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/ruff format --check src/ tests/
	$(VENV)/bin/mypy src/complaints/

# Format code
format:
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

# Clean generated files
clean:
	rm -rf data/bronze/ data/silver/ data/gold/ data/cache/
	rm -rf reports/*.md reports/figures/
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
