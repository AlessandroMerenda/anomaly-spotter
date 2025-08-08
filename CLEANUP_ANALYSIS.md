# 🧹 Analisi File Obsoleti - Anomaly Spotter

## 📋 Status Analisi Codebase

### ✅ **Struttura Modulare Corrente (MANTENERE)**

#### `src/core/` - **✅ ATTIVO**
- `model.py` - AutoencoderUNetLite principale
- `config.py` - Sistema configurazione principale  
- `losses.py` - Loss functions avanzate
- `model_config.py` - Configurazioni modello

#### `src/data/` - **✅ ATTIVO**
- `loaders.py` - DataLoader avanzato con ground truth masks
- `preprocessing.py` - Preprocessing con Albumentations

#### `src/training/` - **✅ ATTIVO**
- `trainer.py` - Trainer avanzato con AMP, scheduling
- Quello che dovremmo mantenere come main training

#### `src/evaluation/` - **✅ ATTIVO**
- `evaluator.py` - Sistema evaluation completo
- `metrics.py` - Metriche avanzate
- `compute_thresholds.py` - Threshold computation

#### `src/utils/` - **✅ ATTIVO**
- `logging_utils.py` - Sistema logging
- `wandb_logger.py` - W&B integration

---

### ❌ **File OBSOLETI da Rimuovere/Consolidare**

#### **1. Training Scripts Duplicati**
- ❌ `src/training/train_model.py` - **OBSOLETO**
  - Usa import obsoleti: `from src.model import AutoencoderUNetLite`
  - Dovrebbe usare: `from src.core.model import AutoencoderUNetLite`
  - Ha classe MVTecGoodDataset duplicata (ora in src/data/loaders.py)
  - **AZIONE**: Rimuovere, già sostituito da src/train_main.py

- ✅ `src/train_main.py` - **MANTIENI** (usa imports corretti)
- ✅ `src/train_batch.py` - **MANTIENI** (per batch processing)

#### **2. Evaluation Scripts Duplicati**
- ❌ `src/evaluation/compute_thresholds.py` - **PROBLEMATICO**
  - Usa import obsoleto: `from src.model import AutoencoderUNetLite`
  - **AZIONE**: Aggiornare imports o consolidare con compute_thresholds_advanced.py

- ✅ `src/compute_thresholds_advanced.py` - **MANTIENI** (più avanzato)
- ✅ `src/evaluate_model.py` - **MANTIENI** (usa imports corretti)
- ✅ `src/evaluate_model_post_training.py` - **MANTIENI** (specifico post-training)

#### **3. Legacy Files** - **src/legacy/** 
- ❌ `extract_metrics.py` - **OBSOLETO** (funzionalità in evaluator.py)
- ❌ `overlay.py` - **OBSOLETO** (funzionalità in evaluation/)  
- ❌ `process_all.py` - **OBSOLETO** (sostituito da batch scripts)
- ❌ `run_overlay_batch.py` - **OBSOLETO** (sostituito da evaluation pipeline)
- ❌ `test_model.py` - **OBSOLETO** (sostituito da evaluate_model.py)

#### **4. File Root Obsoleti**
- ❌ `test_config.py` - **OBSOLETO** (solo per test, non necessario)

---

### 🔧 **AZIONI RICHIESTE**

#### **Immediate (Priorità Alta)**

1. **Rimuovere File Obsoleti**
```bash
rm src/training/train_model.py
rm -rf src/legacy/
rm test_config.py
```

2. **Aggiornare Import Obsoleti**
   - `src/evaluation/compute_thresholds.py` - cambiare import da `src.model` a `src.core.model`

3. **Consolidare Duplicati**
   - Valutare se mantenere sia `compute_thresholds_advanced.py` che `src/evaluation/compute_thresholds.py`

#### **Cleanup (Priorità Media)**

4. **Verificare Notebooks**
   - Controllare se notebooks usano import obsoleti
   - Aggiornare references ai file rimossi

5. **Aggiornare Makefile**
   - Rimuovere references a file obsoleti
   - Aggiornare paths ai nuovi file

#### **Ottimizzazione (Priorità Bassa)**

6. **Consolidare Scripts**
   - Verificare se alcuni script in scripts/ usano file obsoleti
   - Aggiornare documentation

---

### 📊 **Riepilogo Impatto**

**File da Rimuovere**: 7 file obsoleti
**Import da Aggiornare**: 1 file principale  
**Scripts da Verificare**: Makefile, notebooks, scripts/

**Benefici della Pulizia**:
- ✅ Eliminazione confusion da duplicati
- ✅ Imports coerenti con struttura modulare
- ✅ Codebase più pulita e maintainable
- ✅ Riduzione rischio errori import

---

### 🎯 **Struttura Finale Target**

```
src/
├── core/           # Componenti core (model, config, losses)
├── data/           # Data loading e preprocessing  
├── training/       # Solo trainer.py avanzato
├── evaluation/     # Sistema evaluation completo
├── utils/          # Utilities (logging, wandb)
├── train_main.py   # Main training script
├── train_batch.py  # Batch training script
├── evaluate_model.py               # Main evaluation
├── evaluate_model_post_training.py # Post-training evaluation
└── compute_thresholds_advanced.py  # Advanced thresholds
```

**No more duplicates, clean imports, modular architecture! 🚀**
