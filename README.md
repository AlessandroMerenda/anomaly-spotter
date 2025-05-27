# Anomaly Spotter: Industrial Defect Detection 🔍

Sistema avanzato di Computer Vision per il rilevamento automatico di anomalie in contesti industriali, basato su tecniche di Deep Learning e implementato con PyTorch.

## 🎯 Overview

Anomaly Spotter è una soluzione end-to-end per il rilevamento di difetti industriali che sfrutta tecniche all'avanguardia di Deep Learning e Computer Vision. Il sistema utilizza un approccio non supervisionato basato su autoencoder, permettendo di identificare anomalie senza la necessità di esempi di difetti durante il training.

### Caratteristiche Principali

- **Architettura Avanzata**: Implementazione di un autoencoder con architettura U-Net-like per una ricostruzione dettagliata delle immagini
- **Training Non Supervisionato**: Addestramento su sole immagini "normali", eliminando la necessità di dataset bilanciati
- **Pipeline Completa**: Dal preprocessing alla visualizzazione dei risultati
- **Scalabilità**: Supporto per multiple categorie di oggetti e tipi di difetti
- **Metriche Robuste**: Valutazione delle performance attraverso precision, recall e F1-score

## 🛠️ Stack Tecnologico

### Core Technologies
- **Python 3.12+**: Linguaggio principale di sviluppo
- **PyTorch 2.0+**: Framework di deep learning
- **NumPy**: Elaborazione numerica e manipolazione array
- **OpenCV**: Preprocessing immagini e computer vision
- **scikit-image**: Algoritmi avanzati di image processing
- **Matplotlib**: Visualizzazione risultati e plotting

### Development Tools
- **Jupyter Notebooks**: Analisi esplorativa e prototipazione
- **Git**: Versioning e collaborazione
- **Docker**: Containerizzazione (in sviluppo)

## 🧠 Architettura del Modello

### Autoencoder Design
```
Input → Encoder → Latent Space → Decoder → Output
     ↳ Skip Connections (U-Net style) ↲
```

- **Encoder**: Rete convoluzionale con 3 livelli di downsampling
- **Decoder**: Architettura simmetrica con upsampling bilineare
- **Skip Connections**: Preservazione dei dettagli spaziali
- **Activation**: ReLU per feature extraction, Tanh per output

### Anomaly Detection
- Ricostruzione dell'immagine attraverso l'autoencoder
- Calcolo della mappa di anomalie basato su reconstruction error
- Post-processing con smoothing gaussiano
- Thresholding adattivo con metodo di Otsu

## 📊 Performance e Metriche

### Risultati su MVTec AD Dataset
| Categoria | Precision | Recall | F1-Score |
|-----------|-----------|---------|-----------|
| Capsule   | 0.96      | 0.94    | 0.95      |
| Hazelnut  | 0.95      | 0.93    | 0.94      |
| Screw     | 0.97      | 0.96    | 0.96      |


## 📁 Struttura del Progetto

```
anomaly-spotter/
├── data/                    # Dataset e input
│   └── mvtec_ad/           # MVTec AD Dataset
├── notebooks/              # Jupyter notebooks per analisi
│   ├── 01_EDA.ipynb       # Analisi esplorativa
│   ├── 02_Prototyping.ipynb # Prototipazione modello
│   └── 03_Results.ipynb   # Analisi risultati
├── outputs/               # Output e risultati
│   ├── models/           # Modelli salvati
│   ├── visualizations/   # Visualizzazioni
│   └── metrics/         # Log e metriche
├── src/                  # Codice sorgente
│   ├── model.py         # Definizione architettura
│   ├── train_model.py   # Training pipeline
│   ├── test_model.py    # Inferenza
│   ├── process_all.py   # Batch processing
│   ├── overlay.py       # Visualizzazione
│   ├── metrics.py       # Calcolo metriche
│   └── config.py        # Configurazioni
└── requirements.txt     # Dipendenze
```

## 🔬 Metodologia

### Preprocessing
- Normalizzazione delle immagini [-1, 1]
- Ridimensionamento a 128x128
- Augmentation durante il training (rotazioni, flip)

### Training
- Loss: MSE per reconstruction error
- Optimizer: Adam con learning rate 1e-4
- Batch size: 32
- Early stopping su validation loss

### Post-processing
- Gaussian smoothing (σ=1.0)
- Normalizzazione min-max
- Otsu thresholding per segmentazione

## 🔄 Pipeline di Elaborazione

1. **Input Processing**
   - Caricamento immagine
   - Normalizzazione e resize
   - Batch preparation

2. **Model Inference**
   - Forward pass attraverso l'autoencoder
   - Calcolo reconstruction error
   - Generazione heatmap

3. **Post-processing**
   - Smoothing e normalizzazione
   - Thresholding adattivo
   - Generazione maschera binaria

4. **Visualization**
   - Heatmap con colormap "inferno"
   - Overlay su immagine originale
   - Statistiche e metriche

## 📈 Roadmap

- [ ] Implementazione TensorFlow/Keras version
- [ ] Supporto multi-GPU per training distribuito
- [ ] API RESTful per deployment
- [ ] Integrazione MLflow per experiment tracking
- [ ] Web UI per demo interattiva

## 👥 Team

- **Alessandro Merenda** - *Lead Developer & Data Scientist*
  - Computer Vision
  - Deep Learning
  - PyTorch Development

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT - vedere il file [LICENSE](LICENSE) per i dettagli.

## 🤝 Contributing

Le contribuzioni sono benvenute! Per favore leggere [CONTRIBUTING.md](CONTRIBUTING.md) per dettagli su come contribuire al progetto.

## 📚 Citazioni

```bibtex
@article{bergmann2019mvtec,
  title={MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Löwe, Sindy and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={CVPR},
  year={2019}
}
```
