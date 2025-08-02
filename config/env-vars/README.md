# Configurazione Environment Variables

La cartella `config/env-vars/` contiene le **variabili d'ambiente** per il progetto.

## 🎯 File Disponibili

- `.env.example` → Template con tutte le variabili supportate
- `.env` → File locale (non committato) con i tuoi valori

## 🔧 Setup

```bash
# Copia il template
cp config/env-vars/.env.example .env

# Personalizza i valori
nano .env
```

## 📋 Variabili Principali

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `ANOMALY_SPOTTER_ENVIRONMENT` | Ambiente (development/staging/production) | development |
| `ANOMALY_SPOTTER_DEBUG` | Debug mode | true |
| `ANOMALY_SPOTTER_LOG_LEVEL` | Livello logging | INFO |
| `ANOMALY_SPOTTER_BATCH_SIZE` | Batch size training | 32 |
| `ANOMALY_SPOTTER_DATA_ROOT` | Path dati | auto-detect |

## ⚠️ IMPORTANTE

**NON confondere con:**
- `envs/environment.yml` → Conda environment (dipendenze Python)
- `envs/environment-cpu.yml` → Conda environment CPU-only

Questi file gestiscono **variabili di configurazione**, non dipendenze.
