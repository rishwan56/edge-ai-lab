import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def main():
    # 1. Initialize model structure
    model = models.mobilenet_v2()
    # Replace classifier layer with a 2-class linear layer to match training
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # 2. Load model weights
    weights_path = "mobilenet_v2_keyboard_mouse.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights file '{weights_path}' not found. Please run train.py first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded trained model weights from '{weights_path}' on device '{device}'.")
    
    # 3. Define preprocessing pipeline (same as validation/inference in pytorch)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classes = ['keyboard', 'mouse']
    
    # 4. Perform test predictions
    test_dirs = ["images/keyboard", "images/mouse"]
    
    for test_dir in test_dirs:
        if not os.path.isdir(test_dir):
            continue
            
        print(f"\n--- Predictions for folder: {test_dir} ---")
        img_names = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_names:
            print("  No images found.")
            continue
            
        # Run predictions on up to 4 images per folder
        for img_name in sorted(img_names)[:4]:
            img_path = os.path.join(test_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_batch)
                
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            print(f"Image: {img_name:<18} | Probabilities: ", end="")
            for i, cls_name in enumerate(classes):
                pct = probabilities[i].item() * 100
                print(f"{cls_name}: {pct:.2f}%  ", end="")
            
            predicted_idx = torch.argmax(probabilities).item()
            print(f"-> Prediction: {classes[predicted_idx].upper()}")

if __name__ == "__main__":
    main()
