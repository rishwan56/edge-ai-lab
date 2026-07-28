import os
import sys
import numpy as np
from PIL import Image

# Reconfigure stdout/stderr for Windows console unicode support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import onnxruntime as ort

def main():
    onnx_path = "mobilenet_v2_keyboard_mouse.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model file '{onnx_path}' not found.")
        return
        
    print(f"Loading ONNX model from '{onnx_path}'...")
    session = ort.InferenceSession(onnx_path)
    
    # Print input and output details
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print("\n--- Model Metadata ---")
    for i, inp in enumerate(inputs):
        print(f"Input {i}: name='{inp.name}', shape={inp.shape}, type={inp.type}")
    for i, out in enumerate(outputs):
        print(f"Output {i}: name='{out.name}', shape={out.shape}, type={out.type}")
        
    # Preprocess a test image
    image_path = "images/mouse/mouse1.png"
    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found.")
        return
        
    print(f"\nLoading and preprocessing test image: '{image_path}'...")
    image = Image.open(image_path).convert("RGB")
    
    # MobileNetV2 normal preprocessing:
    # 1. Resize to 256x256, then Center Crop to 224x224
    w, h = image.size
    new_w, new_h = (256, int(256 * h / w)) if w < h else (int(256 * w / h), 256)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Center Crop 224x224
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # Convert to float numpy array and normalize
    img_data = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension [1, 3, 224, 224]
    input_tensor = np.expand_dims(img_data, axis=0)
    
    # Run inference using ONNX Runtime
    print("Running inference via ONNX Runtime...")
    input_name = inputs[0].name
    output_name = outputs[0].name
    
    raw_outputs = session.run([output_name], {input_name: input_tensor})
    logits = raw_outputs[0]
    
    # Convert logits to probabilities (Softmax)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probabilities = probabilities[0]
    
    classes = ['keyboard', 'mouse']
    print("\n--- Predictions ---")
    for idx, cls_name in enumerate(classes):
        print(f"{cls_name:<10}: {probabilities[idx] * 100:.2f}%")
        
    predicted_idx = np.argmax(probabilities)
    print(f"\nFinal Prediction: {classes[predicted_idx].upper()}")

if __name__ == "__main__":
    main()
