# Gestione Dipendenze - Anomaly Spotter

## Panoramica

Il progetto utilizza un sistema strutturato di gestione delle dipendenze con file separati per diversi ambienti e casi d'uso.

## File di Requirements

### Core Requirements

- **`requirements.txt`** - Dipendenze base con versioni pinned per stabilità
- **`requirements-dev.txt`** - Dipendenze per sviluppo (include testing tools, linting, etc.)
- **`requirements-test.txt`** - Dipendenze specifiche per CI/CD e testing automatizzato
- **`requirements-prod.txt`** - Dipendenze ottimizzate per produzione (minimal set)
- **`requirements-docker.txt`** - Dipendenze per container Docker (CPU-only PyTorch)
- **`requirements-tools.txt`** - Tools per gestione e sicurezza delle dipendenze

## Strumenti di Gestione

### Script Bash (`deps.sh`)

Script helper per gestione automatizzata delle dipendenze:

```bash
# Installazione per ambiente specifico
./deps.sh install dev      # Sviluppo
./deps.sh install prod     # Produzione
./deps.sh install test     # Testing
./deps.sh install docker   # Docker

# Controlli di sicurezza
./deps.sh security         # Vulnerability scanning
./deps.sh audit           # Dependency audit

# Analisi dipendenze
./deps.sh tree            # Albero dipendenze
./deps.sh outdated       # Pacchetti obsoleti
```

### Makefile

Automazione completa del workflow:

```bash
# Setup ambiente
make dev-setup           # Setup completo sviluppo
make prod-setup          # Setup produzione

# Installazione dipendenze
make install-dev         # Dipendenze sviluppo
make install-prod        # Dipendenze produzione
make install-test        # Dipendenze testing

# Controlli sicurezza
make security           # Vulnerability check
make audit             # Dependency audit
make outdated          # Check aggiornamenti

# Qualità codice
make lint              # Code linting
make format            # Code formatting
make test              # Run test suite

# Applicazione
make run-train         # Training modello
make run-test          # Test modello
make notebook          # Jupyter notebook
```

## Workflow Raccomandati

### Sviluppo Locale

```bash
# 1. Setup iniziale
python -m venv venv
source venv/bin/activate
make dev-setup

# 2. Controlli regolari
make security          # Controllo vulnerabilità
make outdated         # Check aggiornamenti
make lint && make test # Quality check

# 3. Aggiornamento dipendenze
./deps.sh outdated
# Aggiorna manualmente requirements file se necessario
```

### Produzione

```bash
# 1. Installazione pulita
python -m venv venv-prod
source venv-prod/bin/activate
make prod-setup

# 2. Controlli sicurezza
make security
make audit

# 3. Deployment
make run-train  # o altri comandi specifici
```

### Docker

```bash
# Build con dipendenze ottimizzate
docker build -f Dockerfile.prod -t anomaly-spotter:prod .

# Oppure usa requirements-docker.txt per build custom
```

### CI/CD

```bash
# Pipeline di test
make ci-install
make ci-test
```

## Sicurezza e Compliance

### Controlli Automatici

1. **Safety**: Scansione vulnerabilità note
2. **pip-audit**: Audit completo dipendenze
3. **pip-licenses**: Verifica compliance licenze

### Best Practices

1. **Versioni Pinned**: Tutte le dipendenze hanno versioni esatte (==)
2. **Separazione Ambienti**: File separati per dev/test/prod
3. **Controlli Regolari**: Security scan prima di ogni deployment
4. **Minimal Production**: Set ridotto di dipendenze in produzione
5. **Reproducible Builds**: Stesso hash pip per stesso environment

## Aggiornamento Dipendenze

### Processo Sicuro

```bash
# 1. Backup requirements correnti
cp requirements.txt requirements.txt.backup

# 2. Check pacchetti obsoleti
make outdated

# 3. Test ambiente isolato
python -m venv test-upgrade
source test-upgrade/bin/activate

# 4. Upgrade selettivo e test
pip install package==new_version
make test

# 5. Update requirements se test OK
# 6. Commit changes
```

### Monitoraggio Continuo

- **Weekly**: `make outdated` per check aggiornamenti
- **Monthly**: `make security` per vulnerability scan
- **Before Release**: `make audit` per dependency audit completo

## Troubleshooting

### Problemi Comuni

1. **Conflitti di dipendenze**:
   ```bash
   pip-tools compile requirements.in
   pip-sync requirements.txt
   ```

2. **Virtual environment corrotto**:
   ```bash
   rm -rf venv
   make dev-setup
   ```

3. **Dipendenze mancanti**:
   ```bash
   ./deps.sh tree  # Analizza albero dipendenze
   ```

4. **Problemi di sicurezza**:
   ```bash
   ./deps.sh security  # Identifica vulnerabilità
   # Aggiorna pacchetti specifici
   ```

## Integrazione con IDE

### VS Code

Il progetto include configurazioni per:
- Python interpreter detection
- Linting automatico
- Testing integration
- Docker support

### Jupyter

```bash
make notebook  # Avvia con kernel corretto
```

## Performance e Ottimizzazioni

### Dipendenze Lean

- **Produzione**: Solo pacchetti essenziali
- **Docker**: Versioni CPU-only per container leggeri
- **CI**: Parallelizzazione test con pytest-xdist

### Caching

```bash
# Pip cache per build veloci
pip install --cache-dir .pip-cache -r requirements.txt
```

Questo sistema fornisce controllo completo, sicurezza e reproducibilità per tutte le fasi del progetto.
