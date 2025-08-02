import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import AutoencoderUNetLite
from src.utils.config_manager import get_config
from src.utils.logging_utils import (
    setup_logger, ModelError, DataError, ResourceError, 
    handle_exception, check_system_resources, safe_execute
)

class MVTecGoodDataset(Dataset):
    """Dataset che carica solo le immagini 'good' da più categorie MVTec con error handling robusto."""
    
    def __init__(self, root_dir: str, categories: List[str], transform=None, logger: logging.Logger = None):
        self.logger = logger or setup_logger("MVTecGoodDataset")
        self.image_paths = []
        self.transform = transform
        self.root_dir = root_dir
        
        try:
            self._load_images(categories)
            self.logger.info(f"Dataset inizializzato con {len(self.image_paths)} immagini da {len(categories)} categorie")
        except Exception as e:
            handle_exception(self.logger, e, "Inizializzazione dataset training")
            raise DataError(f"Impossibile inizializzare dataset training", {"categories": categories, "root_dir": root_dir})
    
    def _load_images(self, categories: List[str]) -> None:
        """Carica tutti i percorsi delle immagini good dalle categorie specificate."""
        if not os.path.exists(self.root_dir):
            raise DataError(f"Directory dataset non trovata: {self.root_dir}")
        
        for category in categories:
            good_dir = os.path.join(self.root_dir, category, "train", "good")
            
            if not os.path.exists(good_dir):
                self.logger.warning(f"Directory good non trovata per categoria {category}: {good_dir}")
                continue
            
            category_images = 0
            try:
                for img_name in sorted(os.listdir(good_dir)):
                    if img_name.lower().endswith(".png"):
                        img_path = os.path.join(good_dir, img_name)
                        
                        # Validazione base del file
                        if self._validate_image_file(img_path):
                            self.image_paths.append(img_path)
                            category_images += 1
                
                self.logger.info(f"Categoria {category}: caricati {category_images} immagini")
                
            except OSError as e:
                self.logger.error(f"Errore lettura directory {good_dir}: {e}")
                continue
    
    def _validate_image_file(self, file_path: str) -> bool:
        """Validazione rapida di un file immagine."""
        try:
            return (os.path.exists(file_path) and 
                    os.access(file_path, os.R_OK) and 
                    os.path.getsize(file_path) > 0)
        except OSError:
            return False
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """Carica un'immagine con error handling robusto."""
        if idx < 0 or idx >= len(self.image_paths):
            raise DataError(f"Indice dataset non valido: {idx}")
        
        img_path = self.image_paths[idx]
        
        try:
            # Caricamento immagine
            image = Image.open(img_path).convert('RGB')
            
            # Applicazione trasformazioni
            if self.transform:
                image = self.transform(image)
                
            return image
            
        except Exception as e:
            self.logger.error(f"Errore caricamento immagine {img_path}: {e}")
            raise DataError(f"Impossibile caricare immagine training", {"path": img_path, "error": str(e)})

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device, 
                epoch: int, num_epochs: int, logger: logging.Logger) -> float:
    """
    Esegue un epoch di training con error handling robusto.
    
    Returns:
        Average loss dell'epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    try:
        progress = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs} [Train]', 
                       leave=True, ncols=100)
        
        for batch_idx, batch in enumerate(progress):
            try:
                # Gestione batch (può essere solo immagini o (immagini, paths))
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                
                # Verifica validità del batch
                if images is None or images.size(0) == 0:
                    logger.warning(f"Batch vuoto saltato all'indice {batch_idx}")
                    continue
                
                images = images.to(device, non_blocking=True)
                
                # Forward pass
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                
                # Verifica che la loss sia valida
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Loss non valida rilevata: {loss.item()}")
                    raise ModelError("Loss non valida durante training")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping per stabilità
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error("Out of memory durante training - riducendo batch size potrebbe aiutare")
                raise ResourceError("Memoria GPU insufficiente", {"suggested_action": "reduce_batch_size"})
            except Exception as e:
                logger.error(f"Errore nel batch {batch_idx}: {e}")
                # Continua con il prossimo batch invece di crashare tutto il training
                continue
        
        if num_batches == 0:
            raise ModelError("Nessun batch processato con successo nell'epoch")
        
        avg_loss = total_loss / num_batches
        logger.debug(f"Epoch {epoch} training completato: avg_loss={avg_loss:.6f}")
        return avg_loss
        
    except Exception as e:
        handle_exception(logger, e, f"Training epoch {epoch}")
        raise

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device, epoch: int, num_epochs: int, logger: logging.Logger) -> float:
    """
    Esegue un epoch di validazione con error handling.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    try:
        progress = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs} [Valid]', 
                       leave=True, ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress):
                try:
                    # Gestione batch
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        images, _ = batch
                    else:
                        images = batch
                    
                    if images is None or images.size(0) == 0:
                        continue
                    
                    images = images.to(device, non_blocking=True)
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                    
                    # Verifica loss valida
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Loss non valida in validazione batch {batch_idx}: {loss.item()}")
                        continue
                    
                    total_loss += loss.item()
                    num_batches += 1
                    progress.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    logger.warning(f"Errore nel batch validazione {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            raise ModelError("Nessun batch di validazione processato con successo")
        
        avg_loss = total_loss / num_batches
        logger.debug(f"Epoch {epoch} validazione completata: avg_loss={avg_loss:.6f}")
        return avg_loss
        
    except Exception as e:
        handle_exception(logger, e, f"Validation epoch {epoch}")
        raise

def plot_training_history(train_losses: List[float], val_losses: List[float], 
                          save_path: str, logger: logging.Logger) -> None:
    """
    Salva il grafico della storia del training con error handling.
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Assicurati che la directory esista
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Grafico training salvato: {save_path}")
        
    except Exception as e:
        logger.error(f"Errore salvataggio grafico training: {e}")
        # Non bloccare il training per questo errore
        pass


def main():
    """Funzione principale di training con error handling completo."""
    logger = setup_logger("train_model", level="INFO")
    
    try:
        logger.info("🚀 Avvio training anomaly detection model")
        
        # Carica configurazione
        config = get_config()
        logger.info(f"Configurazione caricata per ambiente: {config.environment.value}")
        
        # Controllo risorse sistema
        resources = check_system_resources(logger)
        
        # Verifica risorse minime
        if resources['memory_available_gb'] < 2.0:
            raise ResourceError("Memoria RAM insufficiente (< 2GB disponibili)")
        
        # Device selection con fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device selezionato: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU Memory: {resources.get('cuda_memory_gb', 'Unknown')} GB")
            # Ottimizzazioni memoria GPU
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        # Verifica esistenza directory dati
        if not os.path.exists(config.paths.data_root):
            raise DataError(f"Directory dati non trovata: {config.paths.data_root}")
        
        # Trasformazioni con validazione
        try:
            transform = transforms.Compose([
                transforms.Resize(config.model.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*config.model.channels, 
                                   [0.5]*config.model.channels)
            ])
            logger.debug("Trasformazioni inizializzate")
        except Exception as e:
            raise ModelError(f"Errore inizializzazione trasformazioni: {e}")
        
        # Caricamento dataset
        logger.info("📁 Caricamento dataset...")
        categories = ["screw", "capsule", "hazelnut"]  # TODO: rendere configurabile
        dataset = MVTecGoodDataset(config.paths.data_root, categories, transform=transform, logger=logger)
        
        if len(dataset) == 0:
            raise DataError("Dataset vuoto - nessuna immagine caricata")
        
        # Validazione split
        val_size = int(len(dataset) * config.training.validation_split)
        train_size = len(dataset) - val_size
        
        if train_size < 1 or val_size < 1:
            raise DataError(f"Dataset troppo piccolo per split train/val: total={len(dataset)}")
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # DataLoaders con error handling
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.training.batch_size, 
                shuffle=True, 
                num_workers=min(config.training.num_workers, resources['cpu_count']),
                pin_memory=(device.type == 'cuda'),
                drop_last=True  # Evita batch size inconsistenti
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.training.batch_size, 
                shuffle=False, 
                num_workers=min(config.training.num_workers, resources['cpu_count']),
                pin_memory=(device.type == 'cuda')
            )
            
            logger.info(f"📊 Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}")
            
        except Exception as e:
            raise DataError(f"Errore creazione DataLoaders: {e}")
        
        # Inizializzazione modello
        logger.info("🔧 Inizializzazione modello...")
        try:
            model = AutoencoderUNetLite().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.training.learning_rate
            )
            
            # Conteggio parametri
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Parametri totali: {total_params:,}, Trainable: {trainable_params:,}")
            
        except Exception as e:
            raise ModelError(f"Errore inizializzazione modello: {e}")
        
        # Training loop
        logger.info("⚡ Avvio training loop...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = config.training.early_stopping_patience
        
        for epoch in range(config.training.num_epochs):
            try:
                # Training
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer, 
                    device, epoch+1, config.training.num_epochs, logger
                )
                train_losses.append(train_loss)
                
                # Validation
                val_loss = validate_epoch(
                    model, val_loader, criterion, 
                    device, epoch+1, config.training.num_epochs, logger
                )
                val_losses.append(val_loss)
                
                # Logging risultati epoch
                logger.info(f"📈 Epoch {epoch+1}/{config.training.num_epochs}: "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    try:
                        # Assicurati che la directory esista
                        os.makedirs(os.path.dirname(config.paths.model_file), exist_ok=True)
                        torch.save(model.state_dict(), config.paths.model_file)
                        logger.info(f"✨ Nuovo best model salvato (val_loss: {val_loss:.6f})")
                    except Exception as e:
                        logger.error(f"Errore salvataggio modello: {e}")
                        # Continua il training anche se il salvataggio fallisce
                
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping dopo {epoch+1} epochs (patience: {max_patience})")
                        break
                
                # Plot training history
                plot_training_history(train_losses, val_losses, config.paths.training_plot, logger)
                
                # Memory cleanup
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                logger.info("Training interrotto dall'utente")
                break
            except Exception as e:
                handle_exception(logger, e, f"Epoch {epoch+1}")
                # Decidi se continuare o fermare
                if isinstance(e, ResourceError):
                    logger.error("Errore critico di risorse - stopping training")
                    break
                else:
                    logger.warning("Errore nell'epoch - continuando con il prossimo")
                    continue
        
        logger.info("✅ Training completato con successo!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
    except Exception as e:
        handle_exception(logger, e, "Training principale")
        logger.error("❌ Training fallito")
        sys.exit(1)
    
    finally:
        # Cleanup risorse
        if 'device' in locals() and device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("🧹 Cleanup risorse completato")

if __name__ == "__main__":
    main() 