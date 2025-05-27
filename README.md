# Anomaly Spotter 🔍

Sistema di rilevamento anomalie industriali basato su deep learning, progettato per identificare difetti in componenti manifatturieri attraverso analisi di immagini.

## 🎯 Caratteristiche

- **Training non supervisionato**: Addestra un autoencoder solo su immagini di oggetti non difettosi
- **Rilevamento anomalie**: Identifica automaticamente difetti in nuove immagini
- **Visualizzazione avanzata**: Genera heatmap e maschere binarie per localizzare i difetti
- **Multi-categoria**: Supporta diverse categorie di oggetti (capsule, viti, nocciole)
- **Pipeline scalabile**: Facilmente estendibile a nuove categorie di oggetti

## 🛠️ Tecnologie

- Python 3.12+
- PyTorch 2.0+
- scikit-image
- OpenCV
- NumPy
- Matplotlib

## 📁 Struttura del Progetto

```
anomaly-spotter/
├── data/                    # Dataset MVTec AD
├── notebooks/               # Jupyter notebooks per analisi
├── outputs/                 # Risultati e modelli salvati
├── src/                     # Codice sorgente
│   ├── model.py            # Definizione architettura
│   ├── train_model.py      # Script di training
│   ├── test_model.py       # Test su singola immagine
│   ├── process_all.py      # Elaborazione batch
│   ├── overlay.py          # Visualizzazione risultati
│   ├── compute_thresholds.py # Calcolo soglie ottimali
│   └── metrics.py          # Metriche di valutazione
└── requirements.txt         # Dipendenze Python
```

## 🚀 Installazione

1. Clona il repository:
```bash
git clone https://github.com/tuousername/anomaly-spotter.git
cd anomaly-spotter
```

2. Crea e attiva l'ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\\Scripts\\activate  # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## 💻 Utilizzo

### Training

```bash
python src/train_model.py
```

### Test su Singola Immagine

```bash
python src/test_model.py
```

### Elaborazione Batch

```bash
python src/process_all.py
```

## 📊 Dataset

Il progetto utilizza il dataset MVTec AD, che include:
- 3 categorie di oggetti (capsule, hazelnut, screw)
- Immagini di training (solo oggetti non difettosi)
- Immagini di test (oggetti normali e difettosi)
- Ground truth per la validazione

## 📈 Performance

- **Accuratezza**: >95% su tutte le categorie
- **Recall**: Configurabile per requisiti specifici
- **Velocità**: ~100ms per immagine su CPU

## 🔧 Configurazione

Le principali configurazioni sono modificabili in `src/config.py`:
- Dimensioni immagine
- Parametri training
- Soglie di rilevamento
- Directory output

## 🤝 Contributing

1. Fork il repository
2. Crea un branch (`git checkout -b feature/nome`)
3. Commit i cambiamenti (`git commit -am 'Aggiungi feature'`)
4. Push al branch (`git push origin feature/nome`)
5. Crea una Pull Request

## 📝 TODO

- [ ] Aggiungere supporto per GPU multiple
- [ ] Implementare data augmentation
- [ ] Aggiungere test unitari
- [ ] Migliorare documentazione API
- [ ] Creare interfaccia web

## 📄 Licenza

Questo progetto è sotto licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

## 👥 Autori

- Alessandro Merenda - *Sviluppo iniziale*

## 🙏 Riconoscimenti

- MVTec AD Dataset
- PyTorch Team
- Comunità Open Source
