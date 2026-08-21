.PHONY: help install install-heavy setup run test lint fmt typecheck evaluate dash clean doctor schemas

PY ?= python

help:
	@echo "install        Core deps + editable install (offline demo works after this)"
	@echo "install-heavy  Adds faster-whisper + edge-tts (needs ffmpeg on PATH)"
	@echo "setup          install + copy .env"
	@echo "run            Full pipeline: seed -> transcribe -> enrich -> gold"
	@echo "evaluate       Score against ground truth, fit ASR calibration"
	@echo "dash           Launch the Streamlit dashboard"
	@echo "test           pytest"
	@echo "lint / fmt     ruff check / ruff format"
	@echo "typecheck      mypy"
	@echo "doctor         Environment and backend health check"
	@echo "schemas        Export JSON Schemas for every persisted model"
	@echo "clean          Remove generated lake artefacts"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

install-heavy:
	$(PY) -m pip install -r requirements-heavy.txt
	$(PY) -m pip install -e .

setup: install
	@test -f .env || cp .env.example .env
	@echo "Wrote .env — edit it if you want real ASR or Ollama."

doctor:
	ccma doctor

run:
	ccma run-all

evaluate:
	ccma evaluate

dash:
	streamlit run dashboard/app.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests dashboard pipelines

fmt:
	$(PY) -m ruff format src tests dashboard pipelines

typecheck:
	$(PY) -m mypy src/ccma

schemas:
	ccma schemas --output schemas

clean:
	rm -rf data/bronze data/silver data/gold data/audio data/eval data/dq data/models
	rm -f data/*.duckdb data/*.duckdb.wal
	@echo "Lake cleared. Re-run 'make run'."
