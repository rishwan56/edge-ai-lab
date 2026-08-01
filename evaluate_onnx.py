import os
import sys
import numpy as np
from PIL import Image

# Reconfigure stdout/stderr for Windows console unicode support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import onnxruntime as ort
except ImportError:
    print("Error: 'onnxruntime' is not installed. Please install it to evaluate ONNX models.")
    sys.exit(1)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # MobileNetV2 normal preprocessing:
    # 1. Resize to 256x256, keeping aspect ratio
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
    return np.expand_dims(img_data, axis=0)

def main():
    # Default to the best quantized model if it exists, otherwise fall back to base
    default_model = "mobilenet_v2_keyboard_mouse_best_quantized.onnx"
    if not os.path.exists(default_model):
        default_model = "mobilenet_v2_keyboard_mouse_best.onnx"
    if not os.path.exists(default_model):
        default_model = "mobilenet_v2_keyboard_mouse.onnx"

    onnx_path = sys.argv[1] if len(sys.argv) > 1 else default_model
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model file '{onnx_path}' not found.")
        print("Please verify the model path or run export_onnx.py and quantize_onnx.py first.")
        return

    print(f"Loading ONNX model from: '{os.path.abspath(onnx_path)}'...")
    session = ort.InferenceSession(onnx_path)
    
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_name = inputs[0].name
    output_name = outputs[0].name
    
    classes = ['keyboard', 'mouse']
    test_dirs = ["images/keyboard", "images/mouse"]
    
    # Collect all image samples
    samples = []
    for cls in classes:
        cls_dir = os.path.join("images", cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                samples.append((img_path, cls))
                
    if not samples:
        print("Error: No images found in 'images/keyboard' or 'images/mouse'.")
        return
        
    print(f"Starting batch evaluation on {len(samples)} images...")
    print("-" * 85)
    print(f"{'Image File':<30} | {'Ground Truth':<12} | {'Prediction':<12} | {'Confidence':<10} | {'Status'}")
    print("-" * 85)
    
    correct_count = 0
    
    for img_path, ground_truth in sorted(samples, key=lambda x: x[0]):
        input_tensor = preprocess_image(img_path)
        
        # Run inference
        raw_outputs = session.run([output_name], {input_name: input_tensor})
        logits = raw_outputs[0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = probabilities[0]
        
        predicted_idx = np.argmax(probs)
        predicted_cls = classes[predicted_idx]
        confidence = probs[predicted_idx] * 100
        
        is_correct = (predicted_cls == ground_truth)
        if is_correct:
            correct_count += 1
            status = "✓ CORRECT"
        else:
            status = "✗ INCORRECT"
            
        img_name = os.path.basename(img_path)
        print(f"{img_name:<30} | {ground_truth:<12} | {predicted_cls.upper():<12} | {confidence:>8.2f}% | {status}")
        
    print("-" * 85)
    total_images = len(samples)
    accuracy = (correct_count / total_images) * 100
    print(f"Evaluation Summary:")
    print(f"  Model Tested:     {os.path.basename(onnx_path)}")
    print(f"  Total Images:     {total_images}")
    print(f"  Correct:          {correct_count}")
    print(f"  Incorrect:        {total_images - correct_count}")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print("-" * 85)

if __name__ == "__main__":
    main()
