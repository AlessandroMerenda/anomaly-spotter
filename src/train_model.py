import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Gestione PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import AutoencoderUNetLite
from src.config import MODEL_CONFIG, TRAINING_CONFIG, PATHS, CATEGORIES, DATA_ROOT

class MVTecGoodDataset(Dataset):
    """Dataset che carica solo le immagini 'good' da più categorie MVTec."""
    
    def __init__(self, root_dir, categories, transform=None):
        self.image_paths = []
        self.transform = transform
        
        # Raccogli tutti i percorsi delle immagini good
        for category in categories:
            good_dir = os.path.join(root_dir, category, "train", "good")
            if os.path.exists(good_dir):
                for img_name in os.listdir(good_dir):
                    if img_name.endswith(".png"):
                        self.image_paths.append(os.path.join(good_dir, img_name))
                print(f"✓ Loaded {category}: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    
    progress = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs} [Train]', 
                   leave=True, ncols=100)
    
    for batch in progress:
        images = batch.to(device)
        
        # Forward pass
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    total_loss = 0
    
    progress = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs} [Valid]', 
                   leave=True, ncols=100)
    
    with torch.no_grad():
        for batch in progress:
            images = batch.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def plot_training_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")
    
    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*MODEL_CONFIG['channels'], 
                           [0.5]*MODEL_CONFIG['channels'])
    ])
    
    print("\n📁 Loading dataset...")
    # Dataset e DataLoader
    dataset = MVTecGoodDataset(DATA_ROOT, list(CATEGORIES.keys()), transform=transform)
    
    # Split train/val
    val_size = int(len(dataset) * TRAINING_CONFIG['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Model, Loss, Optimizer
    print("\n🔧 Initializing model...")
    model = AutoencoderUNetLite().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate']
    )
    
    print("\n⚡ Starting training loop...")
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch+1, TRAINING_CONFIG['num_epochs']
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch(
            model, val_loader, criterion, 
            device, epoch+1, TRAINING_CONFIG['num_epochs']
        )
        val_losses.append(val_loss)
        
        # Print summary
        print(f"\n📈 Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} Summary:")
        print(f"   Training Loss: {train_loss:.6f}")
        print(f"   Validation Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATHS['model'])
            print(f"   ✨ New best model saved! (val_loss: {val_loss:.6f})")
        
        # Plot training history
        plot_training_history(
            train_losses, 
            val_losses, 
            PATHS['training_plot']
        )
        
        # Force stdout flush
        sys.stdout.flush()

if __name__ == "__main__":
    main() 