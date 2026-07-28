import os
import sys

# Reconfigure stdout and stderr to handle UTF-8 symbols (like checkmarks) on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import models

def main():
    # 1. Initialize model structure (matching your fine-tuned keyboard/mouse classifier)
    print("Initializing MobileNetV2 model structure...")
    model = models.mobilenet_v2()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # 2. Load model weights
    weights_path = "mobilenet_v2_keyboard_mouse.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights '{weights_path}' not found.")
        return
        
    print(f"Loading trained weights from '{weights_path}'...")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    # 3. Create dummy input tensor matching standard ImageNet input shape: [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Export to ONNX format
    onnx_path = "mobilenet_v2_keyboard_mouse.onnx"
    print(f"Exporting model to ONNX format at '{onnx_path}'...")
    
    torch.onnx.export(
        model,                          # model to export
        dummy_input,                    # model input (dummy tensor)
        onnx_path,                      # where to save the model
        export_params=True,             # store the trained weights inside the model file
        opset_version=18,               # the ONNX opset version to target (matches current PyTorch version)
        do_constant_folding=True,       # optimize model by folding constant nodes
        input_names=["input"],          # the model's input name
        output_names=["output"]         # the model's output name
    )
    
    print("Export successful!")
    print(f"PyTorch Model Size: {os.path.getsize(weights_path) / (1024*1024):.2f} MB")
    print(f"ONNX Model Size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
