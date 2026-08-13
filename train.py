import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# 1. Custom Dataset classes
class KeyboardMouseDataset(Dataset):
    """
    Dataset class to scan the root directory and locate keyboard and mouse images.
    It maps class names to indices and builds a list of image path and label tuples.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['keyboard', 'mouse']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Directory '{cls_dir}' not found.")
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
        
        print(f"Found {len(self.samples)} total images.")

class ImageDataset(Dataset):
    """
    A PyTorch Dataset that loads images on-the-fly from given sample paths 
    and applies torchvision transforms.
    """
    def __init__(self, samples, transform=None): 
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    root_dir = "images"
    
    # 2. Define transforms (with data augmentation for training)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load sample paths
    base_dataset = KeyboardMouseDataset(root_dir)
    samples = base_dataset.samples
    if not samples:
        print("Error: No images found. Ensure images are placed in 'images/keyboard' and 'images/mouse'.")
        return

    # Shuffle and split into Train (70%) and Val (30%)
    random.seed(42)
    random.shuffle(samples)
    
    train_size = int(0.7 * len(samples))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    print(f"Training on {len(train_samples)} samples, validating on {len(val_samples)} samples.")
    
    train_dataset = ImageDataset(train_samples, transform=train_transform)
    val_dataset = ImageDataset(val_samples, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # 3. Load pre-trained MobileNetV2
    print("Loading pre-trained MobileNetV2...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace the classifier layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # Ensure classifier weights are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Set up optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # 4. Training loop
    num_epochs = 15
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.1f}%")
              
        # Check if validation accuracy is better, or if accuracy is the same but loss is lower
        is_best = False
        if epoch_val_acc > best_val_acc:
            is_best = True
        elif epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss:
            is_best = True
            
        if is_best:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "mobilenet_v2_keyboard_mouse_best.pth")
            print(f"  --> Best model updated and saved (Val Acc: {best_val_acc*100:.1f}%, Val Loss: {best_val_loss:.4f})")
               
    # 5. Save the trained model weights, metrics and plots
    import json
    import matplotlib.pyplot as plt
    
    # Save final epoch checkpoint
    save_path = "mobilenet_v2_keyboard_mouse.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model checkpoint saved to {save_path}")
    print(f"Best model checkpoint was saved to mobilenet_v2_keyboard_mouse_best.pth at epoch {best_epoch} with Val Acc: {best_val_acc*100:.1f}%")

    # Plot loss and accuracy curves
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in train_accs], 'b-o', label='Train Acc')
    plt.plot(epochs, [acc * 100 for acc in val_accs], 'r-o', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "training_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to {plot_path}")
    
    # Save metrics JSON
    metrics = {
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accs[-1],
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accs[-1],
        "train_loss_history": train_losses,
        "train_accuracy_history": train_accs,
        "val_loss_history": val_losses,
        "val_accuracy_history": val_accs
    }
    
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
