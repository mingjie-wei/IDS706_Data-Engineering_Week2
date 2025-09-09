# Makefile for Data Analysis Project

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

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

