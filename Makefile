# Makefile for Data Analysis Project
.PHONY: install format lint test clean coverage pytest

ARGS ?= .

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

# make format ARGS="abc.py" / make format ARGS="src tests"
format:
	python -m black $(ARGS)

# make lint ARGS="abc.py"
lint:
	python -m flake8 $(ARGS)

test:
	python - <<'PY'
	import pandas as pd, numpy as np, shap, sklearn
	print("pandas:", pd.__version__)
	print("numpy:", np.__version__)
	print("shap:", shap.__version__)
	print("scikit-learn:", sklearn.__version__)
	PY

clean:
	rm -rf __pycache__ .pytest_cache .ipynb_checkpoints
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + || true

pytest:
	pytest

coverage:
	pytest --cov=src --cov-report=term-missing