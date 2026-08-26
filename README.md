# Edge AI Lab

This repository documents the development, training, optimization, and deployment of lightweight deep learning models on edge devices.

---

## 🚀 What We Have Achieved

- [x] **Environment Setup**: Python 3.12, PyTorch, TorchVision, Pillow, ONNX, and ONNX Runtime.
- [x] **Pre-trained Model Evaluation**: Verified standard ImageNet inference using `MobileNetV2` and `EfficientNet_V2_S` on test images (e.g. classifying bottles, screenshots, etc.).
- [x] **Custom Fine-Tuning (`train.py`)**: 
  - Fine-tuned MobileNetV2 specifically for a 2-class classification problem (`keyboard` vs `mouse`).
  - Saved model weights to `mobilenet_v2_keyboard_mouse.pth`.
  - Achieved high validation accuracy on local dataset images.
- [x] **ONNX Export (`export_onnx.py`)**: 
  - Exported the custom model to `mobilenet_v2_keyboard_mouse.onnx` using static shapes (batch-size 1) for optimal compatibility on edge hardware.
- [x] **INT8 Quantization (`quantize_onnx.py`)**: 
  - Applied shape pre-processing and 8-bit dynamic quantization to the ONNX graph.
  - Successfully reduced the model footprint from **8.76 MB** to **2.33 MB** (a **73.4% size reduction**).
- [x] **ONNX Runtime Validation (`test_onnx.py`)**:
  - Ran predictions on the edge-ready `.onnx` model using only the lightweight `onnxruntime` library.

### Model Prediction & Training Results
- **Task**: Binary Classification (`keyboard` vs `mouse`)
- **Dataset**: `images/keyboard/` and `images/mouse/` folders
- **ONNX Model Inference (`test_onnx.py`) Output for `mouse1.png`**:
  - `mouse` class probability: **84.09%**
  - `keyboard` class probability: **15.91%**
  - **Final Prediction**: **`MOUSE`** (Correct!)

---

## ⏳ Pending Tasks & Next Steps

- [x] **Transfer to Edge Device**: Created lightweight `requirements_pi.txt`, `benchmark_pi.py`, and `pi_camera_test.py` for direct deployment onto Raspberry Pi.
- [x] **Live Camera Feed Application**: Implemented `pi_camera_test.py` utilizing OpenCV and ONNX Runtime to perform real-time classification with live HUD stats.
- [ ] **On-Device Incremental Learning**: Implement local training loop (e.g. KNN or classifier fine-tuning on top of extracted embeddings) directly on device.

---

## 🛠️ Project Structure

- [mobilenet_test.py](file:///c:/rishwan/Projects/edge-ai-lab/mobilenet_test.py): Standard PyTorch MobileNetV2 test script.
- [high_accuracy_test.py](file:///c:/rishwan/Projects/edge-ai-lab/high_accuracy_test.py): Evaluation using higher capacity pre-trained `EfficientNet_V2_S`.
- [train.py](file:///c:/rishwan/Projects/edge-ai-lab/train.py): PyTorch training script for custom `keyboard` vs `mouse` classification.
- [test_fine_tuned.py](file:///c:/rishwan/Projects/edge-ai-lab/test_fine_tuned.py): Verification of custom PyTorch checkpoint.
- [export_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/export_onnx.py): Script to convert `.pth` to `.onnx`.
- [quantize_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/quantize_onnx.py): Script to quantize ONNX model to 8-bit integers.
- [test_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/test_onnx.py): ONNX Runtime verification script on single image.
- [evaluate_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/evaluate_onnx.py): Batch accuracy verification across image sets.
- [benchmark_pi.py](file:///c:/rishwan/Projects/edge-ai-lab/benchmark_pi.py): Edge device performance & latency benchmarking tool.
- [pi_camera_test.py](file:///c:/rishwan/Projects/edge-ai-lab/pi_camera_test.py): Real-time live camera classification with HUD overlay.
- [requirements_pi.txt](file:///c:/rishwan/Projects/edge-ai-lab/requirements_pi.txt): Minimal dependencies for Raspberry Pi.

---

## 🍓 Raspberry Pi Deployment Guide

### 1. Transfer Files to Raspberry Pi
From your development PC (PowerShell / Command Prompt / Git Bash):
```bash
# Replace 'pi' with your Raspberry Pi username and 'raspberrypi.local' with your Pi IP/hostname
scp -r mobilenet_v2_keyboard_mouse_best_quantized.onnx mobilenet_v2_keyboard_mouse_best.onnx* images benchmark_pi.py pi_camera_test.py requirements_pi.txt pi@raspberrypi.local:~/edge-ai-lab/
```

### 2. Setup Environment on Raspberry Pi
SSH into your Raspberry Pi:
```bash
ssh pi@raspberrypi.local
cd ~/edge-ai-lab

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install lightweight dependencies (NO heavy PyTorch needed!)
pip install -r requirements_pi.txt
```

### 3. Run Benchmark on Raspberry Pi
```bash
python benchmark_pi.py
```

### 4. Run Real-Time Camera Feed
```bash
python pi_camera_test.py
```
