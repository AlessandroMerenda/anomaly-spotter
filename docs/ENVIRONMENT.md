# Virtual Environment Management

## 🎯 Strategia Consigliata: Conda

Il progetto Anomaly Spotter utilizza **Conda** come gestore di ambienti virtuali per la gestione ottimale delle dipendenze CUDA e machine learning.

### 🔍 Perché Conda?

Il progetto ha **60+ riferimenti CUDA** nel codice e richiede:
- PyTorch 2.1.2 con supporto CUDA
- Gestione automatica di CUDA toolkit e cuDNN
- Risoluzione avanzata delle dipendenze per ML
- Isolation completo delle librerie sistema

### 🚀 Setup Rapido

```bash
# Setup ambiente CUDA (raccomandato)
./scripts/setup_environment.sh

# Setup ambiente CPU-only (per deploy leggeri)
./scripts/setup_environment.sh --cpu-only

# Setup con nome personalizzato
./scripts/setup_environment.sh --name my-anomaly-env
```

### 📋 Configurazioni Disponibili

#### 1. **envs/environment.yml** - CUDA-enabled (Raccomandato)
- PyTorch con supporto CUDA 11.8
- Ottimizzazioni GPU automatiche
- Ideale per training e development

#### 2. **envs/environment-cpu.yml** - CPU-only
- PyTorch CPU-only (più leggero)
- Per deployment senza GPU
- Container e server di inferenza

### 🔧 Comandi Essenziali

```bash
# Attivare ambiente
conda activate anomaly-spotter

# Verificare installazione
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Training
python src/train_model.py

# Aggiornare ambiente
conda env update -f envs/environment.yml

# Rimuovere ambiente
conda env remove -n anomaly-spotter
```

### 🎛️ Troubleshooting

#### Problemi CUDA
```bash
# Verificare driver NVIDIA
nvidia-smi

# Reinstallare con CUDA specifica
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Fallback CPU
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues
```bash
# Ridurre batch size
# Modificare config/training.yml
batch_size: 16  # invece di 32

# Monitorare memoria GPU
watch -n 1 nvidia-smi
```

### 🐳 Alternativa Docker

Per deployment isolato:
```bash
# Build con CPU requirements
docker build -f docker/Dockerfile.cpu .

# Build con CUDA (richiede nvidia-docker)
docker build -f docker/Dockerfile.gpu .
```

### 🔄 Migration da pip/venv

Se hai già un ambiente pip:
```bash
# Esportare dipendenze esistenti
pip freeze > old_requirements.txt

# Creare nuovo ambiente conda
./scripts/setup_environment.sh

# Verificare compatibilità
diff old_requirements.txt requirements/base.txt
```

### 📊 Monitoraggio Performance

```bash
# GPU utilization
nvidia-smi dmon

# Memory usage durante training
python -c "
import torch
model = torch.load('outputs/model.pth')
print(f'Model size: {sum(p.numel() for p in model.parameters())} parameters')
"
```

### 🎯 Best Practices

1. **Development**: Usa ambiente CUDA completo
2. **Testing**: Ambiente CPU per CI/CD
3. **Production**: Container ottimizzato per target
4. **Backup**: Esporta sempre envs/environment.yml aggiornato

```bash
# Esportare ambiente corrente
conda env export > envs/environment-current.yml
```
