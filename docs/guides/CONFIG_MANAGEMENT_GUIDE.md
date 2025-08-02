# 🔧 Configuration Management Guide

## 📋 Panoramica

Il sistema di Configuration Management di Anomaly Spotter fornisce un approccio completo e flessibile per gestire tutte le configurazioni dell'applicazione attraverso diversi ambienti e deployment scenarios.

## 🏗️ Architettura

### Componenti Principali

1. **Config Manager** (`src/utils/config_manager.py`)
   - Sistema centralizzato per caricamento e gestione configurazioni
   - Supporto per YAML, JSON e environment variables
   - Hot-reload automatico delle configurazioni
   - Validazione completa dei parametri

2. **Configuration Profiles** (`config/`)
   - File separati per ogni ambiente (dev/staging/prod)
   - Personalizzazione per casi d'uso specifici
   - Ereditarietà e override delle configurazioni

3. **Environment Variables** (`.env.example`)
   - Override runtime per deployment
   - Sicurezza per credenziali sensibili
   - Compatibilità Docker/Kubernetes

## 🚀 Utilizzo

### Caricamento Configurazione Base

```python
from src.utils.config_manager import get_config

# Carica configurazione per l'ambiente corrente
config = get_config()

# Accesso alle configurazioni
batch_size = config.training.batch_size
model_path = config.paths.model_file
data_root = config.paths.data_root
```

### Configurazione per Ambiente Specifico

```python
from src.utils.config_manager import get_config_manager

# Carica configurazione per un ambiente specifico
config_manager = get_config_manager()
config = config_manager.load_config(environment="production")
```

### Override con Environment Variables

```bash
# Imposta environment variables
export ANOMALY_SPOTTER_ENVIRONMENT=production
export ANOMALY_SPOTTER_BATCH_SIZE=64
export ANOMALY_SPOTTER_DATA_ROOT=/production/data

# Le configurazioni saranno automaticamente sovrascritte
python src/train_model.py
```

## 📁 Struttura Configurazioni

### File Configurazione (YAML)

```yaml
# config/production.yaml
environment: production
debug: false

model:
  input_size: [128, 128]
  channels: 3
  latent_dim: 512

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001

paths:
  data_root: "/var/lib/anomaly-spotter/data"
  output_root: "/var/lib/anomaly-spotter/outputs"
```

### Environment Variables

```bash
# Ambiente
ANOMALY_SPOTTER_ENVIRONMENT=production

# Training
ANOMALY_SPOTTER_BATCH_SIZE=64
ANOMALY_SPOTTER_LEARNING_RATE=0.0001
ANOMALY_SPOTTER_NUM_EPOCHS=200

# Paths
ANOMALY_SPOTTER_DATA_ROOT=/custom/data/path
ANOMALY_SPOTTER_OUTPUT_ROOT=/custom/output/path
```

## 🎯 Configurazioni per Ambiente

### Development
- Debug abilitato
- Epoche ridotte per testing veloce
- Logging verboso
- Percorsi relativi

### Staging
- Configurazione simile a produzione
- Logging intermedio
- Validazione completa

### Production
- Debug disabilitato
- Logging minimale (solo errori)
- Percorsi assoluti
- Ottimizzazioni performance

## 🔄 Hot-Reload

Il sistema supporta hot-reload automatico delle configurazioni:

```python
# La configurazione viene ricaricata automaticamente se il file cambia
config = get_config(reload_if_changed=True)
```

## 🛡️ Validazione

Tutte le configurazioni vengono validate automaticamente:

```python
# Validazione automatica al caricamento
config = get_config()  # Solleva ConfigError se non valida

# Validazione manuale
try:
    config.validate()
except ConfigError as e:
    logger.error(f"Configurazione non valida: {e}")
```

## 📊 Esempi di Deployment

### Docker Deployment

```dockerfile
# Dockerfile
ENV ANOMALY_SPOTTER_ENVIRONMENT=production
ENV ANOMALY_SPOTTER_DATA_ROOT=/app/data
ENV ANOMALY_SPOTTER_OUTPUT_ROOT=/app/outputs
```

### Kubernetes Deployment

```yaml
# kubernetes-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-spotter-config
data:
  ANOMALY_SPOTTER_ENVIRONMENT: "production"
  ANOMALY_SPOTTER_BATCH_SIZE: "32"
  ANOMALY_SPOTTER_DATA_ROOT: "/mnt/data"
```

### Local Development

```bash
# .env file locale
ANOMALY_SPOTTER_ENVIRONMENT=development
ANOMALY_SPOTTER_DEBUG=true
ANOMALY_SPOTTER_LOG_LEVEL=DEBUG
```

## 🔧 Personalizzazione Avanzata

### Creazione Nuovo Ambiente

```python
from src.utils.config_manager import ConfigManager, Environment

# Estendi enum Environment se necessario
# Crea configurazione custom
config_manager = ConfigManager()
custom_config = config_manager._create_default_config(Environment.DEVELOPMENT)

# Personalizza
custom_config.training.batch_size = 16
custom_config.debug = True

# Salva
config_manager.save_config(custom_config, "config/custom.yaml")
```

### Override Programmatico

```python
# Carica configurazione base
config = get_config()

# Override specifici per run
if torch.cuda.is_available():
    config.training.batch_size *= 2  # Batch più grandi su GPU
    config.training.num_workers = 6

# Usa configurazione modificata
model = train_with_config(config)
```

## ✅ Best Practices

1. **Usa environment variables per informazioni sensibili**
   ```bash
   ANOMALY_SPOTTER_DATA_ROOT=/secure/path/data
   ```

2. **Mantieni configurazioni per ambiente separate**
   - `development.yaml` per sviluppo
   - `production.yaml` per produzione
   - `testing.yaml` per CI/CD

3. **Valida sempre le configurazioni**
   ```python
   config = get_config()
   config.validate()  # Sempre prima dell'uso
   ```

4. **Usa hot-reload per development**
   ```python
   # Sviluppo: ricarica automatica
   config = get_config(reload_if_changed=True)
   ```

5. **Log delle configurazioni caricate**
   ```python
   logger.info(f"Usando ambiente: {config.environment.value}")
   logger.debug(f"Configurazione: {config.to_dict()}")
   ```

## 🚨 Migrazione da Config Vecchio

Il vecchio sistema (`src/config.py`) è deprecato ma mantiene compatibilità:

```python
# VECCHIO (deprecato - emette warning)
from src.config import MODEL_CONFIG, TRAINING_CONFIG

# NUOVO (raccomandato)
from src.utils.config_manager import get_config
config = get_config()
model_config = config.model
training_config = config.training
```

## 🎭 Troubleshooting

### Errori Comuni

1. **ConfigError: Ambiente non valido**
   ```bash
   # Controlla valore environment variable
   echo $ANOMALY_SPOTTER_ENVIRONMENT
   ```

2. **File configurazione non trovato**
   ```python
   # Crea configurazioni di esempio
   from src.utils.config_manager import ConfigManager
   ConfigManager().create_sample_configs()
   ```

3. **Validazione fallita**
   ```python
   # Debug configurazione
   config = get_config()
   print(config.to_dict())  # Ispeziona valori
   ```

Il Configuration Management è ora completamente integrato e production-ready! 🚀
