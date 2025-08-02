# Guida: Sistema di Logging e Error Handling

## 📋 Panoramica

Il progetto è stato aggiornato con un sistema di logging strutturato e error handling robusto che migliora significativamente la robustezza e la debuggabilità del codice.

## 🔧 Componenti Principali

### 1. Logging Strutturato (`src/utils/logging_utils.py`)

**Caratteristiche:**
- ✅ Output colorato su console
- ✅ Log dettagliati su file con timestamp
- ✅ Rotazione automatica log giornaliera
- ✅ Livelli configurabili (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- ✅ Context tracking per debug

**Utilizzo:**
```python
from src.utils.logging_utils import setup_logger

logger = setup_logger("my_module", level="INFO")
logger.info("Operazione completata")
logger.error("Errore critico rilevato")
```

### 2. Error Handling Strutturato

**Eccezioni Custom:**
- `AnomalySpotterError`: Eccezione base del progetto
- `ModelError`: Errori relativi al modello (caricamento, inferenza)
- `DataError`: Errori relativi ai dati (caricamento, validazione)
- `ConfigError`: Errori di configurazione
- `ResourceError`: Errori di risorse (GPU, memoria)

**Utilizzo:**
```python
from src.utils.logging_utils import ModelError, handle_exception

try:
    # operazione rischiosa
    model.load_state_dict(state_dict)
except Exception as e:
    handle_exception(logger, e, "caricamento modello")
    raise ModelError(f"Impossibile caricare modello: {e}")
```

### 3. Validazione Input

**Funzioni di Sicurezza:**
- `validate_file_path()`: Valida percorsi file con controlli di sicurezza
- `validate_image_file()`: Valida specificamente file immagine
- `check_system_resources()`: Monitoring risorse sistema

### 4. Safe Execution

**Esecuzione Protetta:**
```python
from src.utils.logging_utils import safe_execute

result = safe_execute(
    func=lambda: risky_operation(),
    logger=logger,
    context="operazione rischiosa",
    default_return=None
)
```

## 📁 File di Log

I log vengono salvati in:
```
logs/
├── anomaly-spotter_20250802.log    # Log generale
├── train_model_20250802.log        # Log training
├── test_model_20250802.log         # Log testing
└── MVTecDataset_20250802.log       # Log dataset
```

## 🚨 Gestione Errori Migliorata

### Prima (Problematico):
```python
# Nessuna gestione errori
model.load_state_dict(torch.load(model_path))
image = Image.open(img_path).convert('RGB')
```

### Dopo (Robusto):
```python
# Error handling completo
try:
    validated_path = validate_file_path(model_path, must_exist=True)
    state_dict = torch.load(str(validated_path), map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Modello caricato con successo")
except Exception as e:
    handle_exception(logger, e, "caricamento modello")
    raise ModelError(f"Impossibile caricare modello da {model_path}")
```

## ⚡ Benefici

1. **Debug Facilitato**: Log strutturati con context completo
2. **Produzione Ready**: Gestione errori robusta senza crash
3. **Monitoring**: Tracciamento automatico delle risorse
4. **Sicurezza**: Validazione input contro path traversal
5. **Manutenibilità**: Codice più pulito e diagnosticabile

## 🎯 Prossimi Passi

Con questa base solida di logging ed error handling, possiamo ora procedere con:

1. **Configuration Management**: Environment variables e config files
2. **Requirements Pinning**: Versioni specifiche per produzione
3. **Input Validation**: Validazione più estesa per sicurezza
4. **Code Quality**: Type hints e documentazione

Il sistema è ora molto più robusto e pronto per un ambiente di produzione! 🚀
