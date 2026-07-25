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
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.1f}%")
              
    # 5. Save the trained model weights
    save_path = "mobilenet_v2_keyboard_mouse.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
