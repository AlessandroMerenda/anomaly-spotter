# Anomaly Spotter - Makefile
# Automazione tasks di sviluppo e deployment

.PHONY: help install install-dev install-prod install-test install-docker install-tools
.PHONY: clean security audit outdated tree test lint format
.PHONY: run-train run-test run-overlay docker-build docker-run

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
PROJECT_NAME := anomaly-spotter
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Default target
help: ## Mostra questo help
	@echo "Anomaly Spotter - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
venv: ## Crea virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment creato in $(VENV)/"
	@echo "Attivalo con: source $(VENV)/bin/activate"

# Installation targets
install: ## Installa dipendenze base
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Installa dipendenze per sviluppo
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

install-prod: ## Installa dipendenze per produzione
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-prod.txt

install-test: ## Installa dipendenze per testing
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-test.txt

install-docker: ## Installa dipendenze per Docker
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-docker.txt

install-tools: ## Installa tools per gestione dipendenze
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-tools.txt

# Dependency management
security: ## Controlla vulnerabilità di sicurezza
	@echo "🔍 Controllo vulnerabilità..."
	safety check || true
	pip-audit || true

audit: ## Esegue audit completo delle dipendenze
	@echo "📋 Audit dipendenze..."
	pip-audit --desc
	pipdeptree --warn silence

outdated: ## Mostra pacchetti obsoleti
	@echo "📦 Controllo pacchetti obsoleti..."
	pip list --outdated --format=columns

tree: ## Mostra albero delle dipendenze
	@echo "🌳 Albero dipendenze..."
	pipdeptree

# Code quality
lint: ## Esegue linting del codice
	@echo "🔍 Linting codice..."
	flake8 src/ --max-line-length=100 --exclude=__pycache__
	pylint src/ --max-line-length=100 || true

format: ## Formatta il codice
	@echo "✨ Formattazione codice..."
	black src/ --line-length=100
	isort src/ --profile black

# Testing
test: ## Esegue tutti i test
	@echo "🧪 Esecuzione test..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast: ## Esegue test veloci (senza coverage)
	@echo "⚡ Test veloci..."
	pytest tests/ -v -x

# Application commands
run-train: ## Esegue training del modello
	@echo "🚀 Training modello..."
	$(PYTHON) src/train_model.py

run-test: ## Esegue test del modello
	@echo "🔬 Test modello..."
	$(PYTHON) src/test_model.py

run-overlay: ## Esegue overlay batch
	@echo "🎨 Generazione overlay..."
	$(PYTHON) src/run_overlay_batch.py

run-process-all: ## Esegue processo completo
	@echo "⚙️ Processo completo..."
	$(PYTHON) src/process_all.py

# Notebook
notebook: ## Avvia Jupyter notebook
	@echo "📓 Avvio Jupyter notebook..."
	jupyter notebook notebooks/

# Docker commands
docker-build: ## Builda immagine Docker
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## Esegue container Docker
	@echo "🐳 Running Docker container..."
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs $(DOCKER_IMAGE)

# Cleanup
clean: ## Pulisce file temporanei
	@echo "🧹 Pulizia file temporanei..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .pytest_cache/ dist/ build/

clean-outputs: ## Pulisce file di output
	@echo "🗑️ Pulizia outputs..."
	rm -rf outputs/test_results/* outputs/overlay/* outputs/stats/*
	@echo "⚠️ File di output puliti (modello e soglie mantenuti)"

# Development workflow
dev-setup: venv install-dev install-tools ## Setup completo per sviluppo
	@echo "✅ Setup sviluppo completato!"
	@echo "Attiva l'ambiente: source $(VENV)/bin/activate"

prod-setup: install-prod ## Setup per produzione
	@echo "✅ Setup produzione completato!"

# CI/CD helpers
ci-install: install-test ## Installa dipendenze per CI
	@echo "🤖 Dipendenze CI installate"

ci-test: lint test ## Esegue test completi per CI
	@echo "🤖 Test CI completati"

# Status check
status: ## Mostra status del progetto
	@echo "📊 Status Anomaly Spotter:"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Virtual env: $(shell echo $$VIRTUAL_ENV)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'N/A')"
	@echo "Git status: $(shell git status --porcelain 2>/dev/null | wc -l) modified files"
	@echo "Dependencies: $(shell $(PIP) list | wc -l) packages installed"
