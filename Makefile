.PHONY: setup test bench demo lint format clean

PYTHON ?= python

setup:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v

bench:
	python -m bench.run_all

demo:
	python -m uvicorn demo.server:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check .

format:
	ruff format .

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
