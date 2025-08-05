# Anomaly Spotter - Makefile
# Automazione tasks di sviluppo e deployment

.PHONY: help install install-dev install-prod install-test install-docker install-tools
.PHONY: clean security audit outdated tree test lint format
.PHONY: run-train run-test run-overlay docker-build docker-run
.PHONY: train train-core train-parallel train-all train-debug train-quick resume
.PHONY: evaluate evaluate-all compute-thresholds test-model
.PHONY: check-data stats monitor config help-train help-test help-data help-eval

# Variables
PYTHON := python3
PIP := pip3
CATEGORY ?= capsule
DATA_DIR ?= data/mvtec_ad
OUTPUT_DIR ?= outputs
CONFIG ?= config/training_config.yaml
NUM_WORKERS ?= 4
BATCH_SIZE ?= 32
EPOCHS ?= 100
MODEL_PATH ?= outputs/model.pth
VENV := venv
PROJECT_NAME := anomaly-spotter
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Default target
help: ## Mostra tutti i comandi disponibili
	@echo "🚀 Anomaly Spotter - Sistema di Rilevamento Anomalie"
	@echo "====================================================="
	@echo ""
	@echo "📦 SETUP E GESTIONE:"
	@echo "  make install         - Installa dipendenze base"
	@echo "  make install-dev     - Installa dipendenze sviluppo"
	@echo "  make setup           - Setup ambiente completo"
	@echo "  make clean           - Pulisci output e cache"
	@echo ""
	@echo "🚀 TRAINING:"
	@echo "  make train           - Addestra singola categoria"
	@echo "  make train-core      - Addestra categorie principali"
	@echo "  make train-parallel  - Training parallelo"
	@echo "  make train-all       - Addestra tutte le categorie"
	@echo "  make resume          - Riprendi training da checkpoint"
	@echo ""
	@echo "📊 EVALUATION:"
	@echo "  make evaluate        - Valutazione completa modello"
	@echo "  make evaluate-all    - Valuta tutti i modelli"
	@echo "  make compute-thresholds - Calcola soglie ottimali"
	@echo "  make test-model      - Test modello su campioni"
	@echo ""
	@echo "📈 MONITORING:"
	@echo "  make monitor         - Avvia Tensorboard"
	@echo "  make stats           - Statistiche dataset"
	@echo "  make check-data      - Valida struttura dataset"
	@echo ""
	@echo "ℹ️  HELP DETTAGLIATO:"
	@echo "  make help-train      - Help comandi training"
	@echo "  make help-eval       - Help comandi evaluation"
	@echo "  make help-data       - Help comandi dataset"
	@echo ""
	@echo "⚙️  CONFIGURAZIONE:"
	@echo "  CATEGORY=<nome>      - Categoria (capsule/hazelnut/screw)"
	@echo "  BATCH_SIZE=<n>       - Dimensione batch (default: 32)"
	@echo "  EPOCHS=<n>           - Numero epoche (default: 100)"
	@echo "  MODEL_PATH=<path>    - Percorso modello per evaluation"
	@echo ""
	@echo "📚 ESEMPI RAPIDI:"
	@echo "  make train CATEGORY=capsule EPOCHS=50"
	@echo "  make evaluate CATEGORY=hazelnut MODEL_PATH=outputs/hazelnut_*/model.pth"
	@echo "  make train-parallel BATCH_SIZE=16"

# Help sections
help-train: ## Help dettagliato per training
	@echo "🚀 TRAINING - Comandi Dettagliati"
	@echo "================================="
	@echo ""
	@echo "📝 COMANDI PRINCIPALI:"
	@echo "  make train CATEGORY=capsule      - Addestra categoria capsule"
	@echo "  make train CATEGORY=hazelnut     - Addestra categoria hazelnut"  
	@echo "  make train CATEGORY=screw        - Addestra categoria screw"
	@echo ""
	@echo "🔧 PARAMETRI CONFIGURABILI:"
	@echo "  CATEGORY=<nome>     - Categoria da addestrare (obbligatorio)"
	@echo "  EPOCHS=<numero>     - Numero epoche (default: 100)"
	@echo "  BATCH_SIZE=<numero> - Dimensione batch (default: 32)"
	@echo "  LEARNING_RATE=<val> - Learning rate (default: 0.001)"
	@echo ""
	@echo "🚀 MODALITÀ AVANZATE:"
	@echo "  make train-core     - Addestra capsule, hazelnut, screw in sequenza"
	@echo "  make train-parallel - Training parallelo (più veloce, richiede più GPU)"
	@echo "  make train-all      - Addestra tutte le categorie disponibili"
	@echo ""
	@echo "💾 CHECKPOINT E RESUME:"
	@echo "  make resume CATEGORY=capsule - Riprende training da ultimo checkpoint"
	@echo ""
	@echo "📊 MONITORAGGIO:"
	@echo "  make monitor        - Avvia Tensorboard per visualizzare metriche"
	@echo "  tail -f logs/training_<categoria>.log  - Log in tempo reale"

help-eval: ## Help dettagliato per evaluation
	@echo "📊 EVALUATION - Comandi Dettagliati"
	@echo "==================================="
	@echo ""
	@echo "📝 COMANDI PRINCIPALI:"
	@echo "  make evaluate CATEGORY=capsule   - Valuta modello categoria capsule"
	@echo "  make evaluate-all               - Valuta tutti i modelli disponibili"
	@echo ""
	@echo "🔧 PARAMETRI CONFIGURABILI:"
	@echo "  CATEGORY=<nome>     - Categoria da valutare (obbligatorio per evaluate)"
	@echo "  MODEL_PATH=<path>   - Percorso specifico al modello"
	@echo "  THRESHOLD=<valore>  - Soglia manuale (default: da thresholds.json)"
	@echo ""
	@echo "📈 CALCOLO SOGLIE:"
	@echo "  make compute-thresholds CATEGORY=capsule - Calcola soglie ottimali"
	@echo "  make compute-thresholds - Calcola per tutte le categorie"
	@echo ""
	@echo "🧪 TEST E VALIDAZIONE:"
	@echo "  make test-model CATEGORY=capsule - Test rapido con campioni casuali"
	@echo ""
	@echo "📁 OUTPUT GENERATI:"
	@echo "  outputs/<categoria>_*/evaluation_report.html - Report dettagliato"
	@echo "  outputs/<categoria>_*/roc_curves.png         - Curve ROC/PR"
	@echo "  outputs/<categoria>_*/confusion_matrix.png   - Matrice confusione"

help-data: ## Help dettagliato per dataset
	@echo "📁 DATASET - Comandi Dettagliati"
	@echo "================================"
	@echo ""
	@echo "📊 ANALISI DATASET:"
	@echo "  make check-data     - Verifica struttura e integrità dataset"
	@echo "  make stats          - Statistiche dettagliate su immagini e labels"
	@echo ""
	@echo "🔍 STRUTTURA PREVISTA:"
	@echo "  data/mvtec_ad/"
	@echo "    ├── capsule/"
	@echo "    │   ├── train/good/     - Immagini di training normali"
	@echo "    │   ├── test/good/      - Immagini test normali"
	@echo "    │   ├── test/<defect>/  - Immagini test con difetti"
	@echo "    │   └── ground_truth/<defect>/ - Maschere ground truth"
	@echo "    ├── hazelnut/"
	@echo "    └── screw/"
	@echo ""
	@echo "✅ VALIDAZIONI ESEGUITE:"
	@echo "  - Presenza cartelle obbligatorie"
	@echo "  - Corrispondenza immagini test/ground_truth"
	@echo "  - Formati immagini supportati"
	@echo "  - Dimensioni e risoluzioni"

# Environment setup
venv: ## Crea virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment creato in $(VENV)/"
	@echo "Attivalo con: source $(VENV)/bin/activate"

# Installation targets
install: ## Installa dipendenze base
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/base.txt

install-dev: ## Installa dipendenze per sviluppo
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/dev.txt

install-prod: ## Installa dipendenze per produzione
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/prod.txt

install-test: ## Installa dipendenze per testing
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/test.txt

install-docker: ## Installa dipendenze per Docker
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/docker.txt

install-tools: ## Installa tools per gestione dipendenze
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/tools.txt

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

# =============================================================================
# TRAINING PIPELINE COMMANDS
# =============================================================================

# Training targets
train: ## Train single category
	@echo "🚀 Training $(CATEGORY) category..."
	$(PYTHON) src/train_main.py \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--config $(CONFIG) \
		--num-workers $(NUM_WORKERS) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS)

train-core: ## Train core categories (capsule, hazelnut, screw)
	@echo "🚀 Training core categories (capsule, hazelnut, screw)..."
	$(PYTHON) src/train_batch.py \
		--categories capsule hazelnut screw \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--config $(CONFIG) \
		--num-workers $(NUM_WORKERS) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--parallel 1

train-parallel: ## Train core categories in parallel
	@echo "🚀 Training core categories in parallel..."
	$(PYTHON) src/train_batch.py \
		--categories capsule hazelnut screw \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--config $(CONFIG) \
		--num-workers $(NUM_WORKERS) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--parallel 3

train-all: ## Train all MVTec categories
	@echo "🚀 Training all MVTec categories..."
	$(PYTHON) src/train_batch.py \
		--all-categories \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--config $(CONFIG) \
		--num-workers $(NUM_WORKERS) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--parallel 2

train-debug: ## Debug training (2 epochs, small batch)
	@echo "🐛 Debug training for $(CATEGORY)..."
	$(PYTHON) src/train_main.py \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)/debug \
		--config $(CONFIG) \
		--num-workers 1 \
		--batch-size 4 \
		--epochs 2 \
		--debug

train-quick: ## Quick test training (2 epochs)
	@echo "⚡ Quick training test..."
	$(PYTHON) src/train_main.py \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)/quick_test \
		--batch-size 4 \
		--epochs 2 \
		--num-workers 1

resume: ## Resume training from checkpoint (requires CHECKPOINT=path)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT variable not set"; \
		echo "Usage: make resume CHECKPOINT=path/to/checkpoint.pth CATEGORY=category_name"; \
		exit 1; \
	fi
	@echo "🔄 Resuming training from $(CHECKPOINT)..."
	$(PYTHON) src/train_main.py \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--config $(CONFIG) \
		--num-workers $(NUM_WORKERS) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--resume $(CHECKPOINT)

# Dataset utilities
check-data: ## Check dataset structure
	@echo "🔍 Checking dataset structure..."
	@for category in capsule hazelnut screw; do \
		echo "Checking $$category..."; \
		if [ -d "$(DATA_DIR)/$$category/train/good" ]; then \
			echo "  ✅ Training data: $$(ls $(DATA_DIR)/$$category/train/good/*.png 2>/dev/null | wc -l) images"; \
		else \
			echo "  ❌ Training data missing"; \
		fi; \
		if [ -d "$(DATA_DIR)/$$category/test" ]; then \
			echo "  ✅ Test data found"; \
		else \
			echo "  ❌ Test data missing"; \
		fi; \
		echo ""; \
	done

stats: ## Show dataset statistics
	@echo "📊 Dataset Statistics:"
	@echo "======================"
	@for category in capsule hazelnut screw; do \
		if [ -d "$(DATA_DIR)/$$category" ]; then \
			echo "📦 $$category:"; \
			echo "  Train (normal): $$(find $(DATA_DIR)/$$category/train/good -name "*.png" 2>/dev/null | wc -l)"; \
			echo "  Test (all): $$(find $(DATA_DIR)/$$category/test -name "*.png" 2>/dev/null | wc -l)"; \
			echo "  Test (normal): $$(find $(DATA_DIR)/$$category/test/good -name "*.png" 2>/dev/null | wc -l)"; \
			echo "  Test (anomaly): $$(find $(DATA_DIR)/$$category/test -name "*.png" -not -path "*/good/*" 2>/dev/null | wc -l)"; \
			echo ""; \
		fi; \
	done

# Monitoring and utilities
monitor: ## Start Tensorboard monitoring
	@echo "📈 Starting Tensorboard monitoring..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	tensorboard --logdir=runs --port=6006 --host=0.0.0.0

# Monitoring and utilities
monitor: ## Start Tensorboard monitoring
	@echo "📈 Starting Tensorboard monitoring..."
	@echo "🌐 Open http://localhost:6006 in your browser"
	tensorboard --logdir=runs --port=6006 --host=0.0.0.0

wandb-login: ## Login to Weights & Biases
	@echo "🔐 Logging in to Weights & Biases..."
	wandb login

wandb-demo: ## Run W&B integration demo
	@echo "🎯 Running W&B integration demo..."
	$(PYTHON) examples/train_with_wandb.py \
		--category $(or $(CATEGORY),capsule) \
		--use-wandb \
		--epochs 5 \
		--batch-size 16 \
		--wandb-project anomaly-demo

config: ## Show current training configuration
	@echo "⚙️  Current Training Configuration:"
	@echo "=================================="
	@echo "Category: $(CATEGORY)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Output Directory: $(OUTPUT_DIR)"
	@echo "Config File: $(CONFIG)"
	@echo "Batch Size: $(BATCH_SIZE)"
	@echo "Epochs: $(EPOCHS)"
	@echo "Workers: $(NUM_WORKERS)"
	@echo "Python: $(PYTHON)"

# Batch training commands
train-batch-core: ## Training batch su categorie principali (capsule, hazelnut, screw)
	@echo "🚀 Starting batch training on core categories..."
	./scripts/train_all_categories.sh --core-categories --batch-size $(BATCH_SIZE) --epochs $(EPOCHS)

train-batch-all: ## Training batch su tutte le categorie MVTec
	@echo "🌟 Starting batch training on all categories..."
	./scripts/train_all_categories.sh --all-categories --batch-size $(BATCH_SIZE) --epochs $(EPOCHS)

train-batch-parallel: ## Training batch parallelo (richiede --gpu-ids)
	@if [ -z "$(GPU_IDS)" ]; then \
		echo "❌ Error: GPU_IDS variable not set"; \
		echo "Usage: make train-batch-parallel GPU_IDS=\"0,1,2\""; \
		exit 1; \
	fi
	@echo "⚡ Starting parallel batch training..."
	./scripts/train_all_categories.sh --parallel --gpu-ids "$(GPU_IDS)" --core-categories

train-batch-quick: ## Training batch veloce per test (2 epoche)
	@echo "⚡ Quick batch training for testing..."
	./scripts/train_all_categories.sh --core-categories --epochs 2 --batch-size 16 --no-evaluation

# Advanced evaluation commands
evaluate-post-training: ## Valutazione post-training completa con visualizzazioni
	@echo "🔍 Starting comprehensive post-training evaluation..."
	$(PYTHON) src/evaluate_model_post_training.py \
		--model-dir $(or $(MODEL_DIR),outputs/capsule_latest) \
		--category $(or $(CATEGORY),capsule) \
		--data-dir $(DATA_DIR) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--num-samples $(or $(NUM_SAMPLES),10)

evaluate-post-training-optimal: ## Valutazione con ricerca soglie ottimali
	@echo "🎯 Post-training evaluation with optimal threshold search..."
	$(PYTHON) src/evaluate_model_post_training.py \
		--model-dir $(or $(MODEL_DIR),outputs/capsule_latest) \
		--category $(or $(CATEGORY),capsule) \
		--data-dir $(DATA_DIR) \
		--find-optimal \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

evaluate-post-training-all: ## Valutazione tutte le categorie con soglie ottimali
	@echo "🌟 Comprehensive evaluation for all categories..."
	$(PYTHON) src/evaluate_model_post_training.py \
		--model-dir $(or $(MODEL_DIR),outputs/multi_latest) \
		--category all \
		--data-dir $(DATA_DIR) \
		--find-optimal \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--num-samples 15

# Help for training commands
help-train: ## Show training command help
	@echo "🚀 Training Commands:"
	@echo "===================="
	@echo "  make train           - Train single category"
	@echo "  make train-core      - Train capsule, hazelnut, screw sequentially"
	@echo "  make train-parallel  - Train core categories in parallel"
	@echo "  make train-all       - Train all 15 MVTec categories"
	@echo "  make train-debug     - Debug training (2 epochs, small batch)"
	@echo "  make train-quick     - Quick test training"
	@echo "  make resume          - Resume from checkpoint"
	@echo ""
	@echo "📋 Configuration:"
	@echo "  CATEGORY=<name>      - Set category (default: capsule)"
	@echo "  BATCH_SIZE=<size>    - Set batch size (default: 32)"
	@echo "  EPOCHS=<num>         - Set epochs (default: 100)"
	@echo ""
	@echo "📚 Examples:"
	@echo "  make train CATEGORY=hazelnut EPOCHS=50"
	@echo "  make train-all BATCH_SIZE=16"
	@echo "  make resume CHECKPOINT=outputs/checkpoint_best.pth CATEGORY=screw"

help-data: ## Show dataset command help
	@echo "📊 Dataset Commands:"
	@echo "===================="
	@echo "  make check-data      - Validate dataset structure"
	@echo "  make stats           - Show dataset statistics"

# =============================================================================
# EVALUATION AND TESTING COMMANDS  
# =============================================================================

# Model evaluation
evaluate: ## Evaluate trained model with comprehensive metrics
	@echo "📊 Evaluating $(CATEGORY) model..."
	$(PYTHON) src/evaluate_model.py \
		--model-path $(MODEL_PATH) \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)/evaluation \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--compute-thresholds \
		--create-plots

evaluate-all: ## Evaluate all trained models
	@echo "📊 Evaluating all trained models..."
	@for category in capsule hazelnut screw; do \
		echo "Evaluating $$category..."; \
		if [ -f "$(OUTPUT_DIR)/$$category*/model.pth" ]; then \
			MODEL_FILE=$$(ls $(OUTPUT_DIR)/$$category*/model.pth | head -1); \
			$(PYTHON) src/evaluate_model.py \
				--model-path $$MODEL_FILE \
				--category $$category \
				--data-dir $(DATA_DIR) \
				--output-dir $(OUTPUT_DIR)/evaluation_$$category \
				--batch-size $(BATCH_SIZE) \
				--compute-thresholds \
				--create-plots; \
		else \
			echo "❌ No trained model found for $$category"; \
		fi; \
		echo ""; \
	done

compute-thresholds: ## Compute optimal thresholds for trained model
	@echo "🎯 Computing thresholds for $(CATEGORY)..."
	$(PYTHON) src/compute_thresholds_advanced.py \
		--model-path $(MODEL_PATH) \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--output-path $(OUTPUT_DIR)/thresholds_$(CATEGORY).json \
		--batch-size $(BATCH_SIZE) \
		--strategies statistical percentile mad validation_f1 \
		--use-validation

test-model: ## Test model on specific samples
	@echo "🧪 Testing $(CATEGORY) model..."
	$(PYTHON) src/test_model.py \
		--category $(CATEGORY) \
		--data-dir $(DATA_DIR) \
		--model-path $(MODEL_PATH) \
		--output-dir $(OUTPUT_DIR)/test_results \
		--batch-size $(BATCH_SIZE)

# Help for evaluation commands
help-eval: ## Show evaluation command help
	@echo "📊 Evaluation Commands:"
	@echo "======================="
	@echo "  make evaluate          - Comprehensive model evaluation"
	@echo "  make evaluate-all      - Evaluate all trained models"
	@echo "  make compute-thresholds- Compute optimal thresholds"
	@echo "  make test-model        - Test model on samples"
	@echo ""
	@echo "📋 Configuration:"
	@echo "  MODEL_PATH=<path>      - Set model path (default: outputs/model.pth)"
	@echo "  CATEGORY=<name>        - Set category (default: capsule)"
	@echo ""
	@echo "📚 Examples:"
	@echo "  make evaluate CATEGORY=hazelnut MODEL_PATH=outputs/hazelnut_123/model.pth"
	@echo "  make compute-thresholds CATEGORY=screw"
	@echo "  make evaluate-all"
