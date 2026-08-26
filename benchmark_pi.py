import os
import sys
import time
import numpy as np
from PIL import Image

# Unicode console support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import onnxruntime as ort
except ImportError:
    print("Error: 'onnxruntime' is not installed.")
    print("Run: pip install onnxruntime")
    sys.exit(1)

def get_file_size_mb(path):
    if not os.path.exists(path):
        return 0.0
    size = os.path.getsize(path)
    if os.path.exists(path + ".data"):
        size += os.path.getsize(path + ".data")
    return size / (1024 * 1024)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # MobileNetV2 normal preprocessing:
    # 1. Resize to 256x256, keeping aspect ratio
    w, h = image.size
    new_w, new_h = (256, int(256 * h / w)) if w < h else (int(256 * w / h), 256)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 2. Center crop 224x224
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # 3. Normalize to [-mean/std]
    img_data = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # 4. HWC -> CHW and add batch dimension [1, 3, 224, 224]
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

def benchmark_model(model_path, num_warmup=10, num_iterations=100, num_threads=4):
    if not os.path.exists(model_path):
        print(f"Warning: Model '{model_path}' not found. Skipping benchmark.")
        return None

    # Set ONNX Runtime session options for optimal Raspberry Pi CPU usage
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Dummy input for latency benchmarking
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print(f"  Performing {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = session.run([output_name], {input_name: dummy_input})
        
    # Latency measurement
    print(f"  Benchmarking over {num_iterations} iterations (CPU threads: {num_threads})...")
    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # convert to ms
        
    latencies = np.array(latencies)
    
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    
    # Accuracy test on dataset
    classes = ['keyboard', 'mouse']
    samples = []
    for cls in classes:
        cls_dir = os.path.join("images", cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(cls_dir, f), cls))
                    
    correct = 0
    total = len(samples)
    if total > 0:
        for img_path, gt in samples:
            tensor = preprocess_image(img_path)
            res = session.run([output_name], {input_name: tensor})[0]
            pred_idx = np.argmax(res[0])
            if classes[pred_idx] == gt:
                correct += 1
        accuracy = (correct / total) * 100.0
    else:
        accuracy = 0.0
        
    return {
        "model_name": os.path.basename(model_path),
        "size_mb": get_file_size_mb(model_path),
        "mean_ms": mean_lat,
        "median_ms": median_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "std_ms": std_lat,
        "fps": fps,
        "accuracy": accuracy,
        "samples_tested": total
    }

def main():
    print("=" * 70)
    print("   EDGE AI LAB - RASPBERRY PI ONNX INFERENCE BENCHMARK")
    print("=" * 70)
    
    models_to_test = [
        ("mobilenet_v2_keyboard_mouse_best.onnx", "FP32 Base ONNX Model"),
        ("mobilenet_v2_keyboard_mouse_best_quantized.onnx", "INT8 Quantized ONNX Model")
    ]
    
    # Check if a custom model was passed
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        models_to_test = [(custom_path, f"Custom: {custom_path}")]
        
    results = []
    
    for model_path, description in models_to_test:
        print(f"\n[+] Testing: {description} ({model_path})")
        if not os.path.exists(model_path):
            print(f"    File not found at '{model_path}'.")
            continue
            
        stats = benchmark_model(model_path, num_warmup=10, num_iterations=100, num_threads=4)
        if stats:
            results.append((description, stats))
            
    if not results:
        print("\nNo valid models were found to benchmark.")
        return
        
    print("\n" + "=" * 85)
    print(f"{'Model / Optimization':<26} | {'Size':<8} | {'Avg Latency':<12} | {'FPS':<8} | {'Accuracy':<10}")
    print("-" * 85)
    
    for desc, s in results:
        print(f"{desc:<26} | {s['size_mb']:>5.2f} MB | {s['mean_ms']:>7.2f} ms   | {s['fps']:>5.1f}  | {s['accuracy']:>6.1f}% ({s['samples_tested']} imgs)")
        
    print("=" * 85)
    
    if len(results) >= 2:
        fp32_lat = results[0][1]['mean_ms']
        int8_lat = results[1][1]['mean_ms']
        fp32_size = results[0][1]['size_mb']
        int8_size = results[1][1]['size_mb']
        
        speedup = fp32_lat / int8_lat if int8_lat > 0 else 1.0
        size_reduction = (1.0 - int8_size / fp32_size) * 100.0 if fp32_size > 0 else 0.0
        
        print("\nOptimization Summary:")
        print(f"  • Model Compression:  {size_reduction:.1f}% smaller")
        print(f"  • Inference Speedup:  {speedup:.2f}x faster on CPU")

if __name__ == "__main__":
    main()
