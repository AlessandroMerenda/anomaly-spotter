# Conda Environments

Questa cartella contiene i file di configurazione per gli ambienti Conda del progetto Anomaly Spotter.

## 📁 File Disponibili

### 🎯 **environment.yml** (CUDA-enabled)
- **Scopo**: Ambiente completo per development e training
- **Hardware**: GPU con CUDA 11.8+ 
- **PyTorch**: Con supporto CUDA
- **Uso**: Sviluppo locale, training modelli

### 💻 **environment-cpu.yml** (CPU-only)
- **Scopo**: Ambiente leggero per deployment
- **Hardware**: Solo CPU
- **PyTorch**: CPU-only build
- **Uso**: Container, server di inferenza, CI/CD

## 🚀 Utilizzo

```bash
# Ambiente CUDA (raccomandato per development)
conda env create -f envs/environment.yml
conda activate anomaly-spotter

# Ambiente CPU-only (per deployment)
conda env create -f envs/environment-cpu.yml -n anomaly-spotter-cpu
conda activate anomaly-spotter-cpu
```

## 🔧 Script di Setup

Usa lo script automatizzato per semplificare il setup:

```bash
# Setup automatico ambiente CUDA
./scripts/setup_environment.sh

# Setup automatico ambiente CPU
./scripts/setup_environment.sh --cpu-only
```

## 📋 Aggiornamenti

```bash
# Aggiornare ambiente esistente
conda env update -f envs/environment.yml

# Esportare ambiente corrente
conda env export > envs/environment-backup.yml
```

## ⚙️ Personalizzazioni

Per modifiche permanenti, edita direttamente i file `.yml`.
Per override temporanei, usa le variabili d'ambiente in `config/env-vars/.env.example`.
