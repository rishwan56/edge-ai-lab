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

- [ ] **Transfer to Edge Device**: Move optimized `.onnx` models and validation scripts to a physical edge device (e.g. Raspberry Pi).
- [ ] **Live Camera Feed Application**: Write a script utilizing OpenCV and ONNX Runtime to perform real-time classification on the Raspberry Pi camera module.
- [ ] **On-Device Incremental Learning**: Implement the local training loop (e.g. KNN on extracted 1280-D embeddings or single-layer SGD updates) to allow the model to learn new objects directly on the edge.

---

## 🛠️ Project Structure

- [mobilenet_test.py](file:///c:/rishwan/Projects/edge-ai-lab/mobilenet_test.py): Standard PyTorch MobileNetV2 test script.
- [high_accuracy_test.py](file:///c:/rishwan/Projects/edge-ai-lab/high_accuracy_test.py): Evaluation using higher capacity pre-trained `EfficientNet_V2_S`.
- [train.py](file:///c:/rishwan/Projects/edge-ai-lab/train.py): PyTorch training script for custom `keyboard` vs `mouse` classification.
- [test_fine_tuned.py](file:///c:/rishwan/Projects/edge-ai-lab/test_fine_tuned.py): Verification of custom PyTorch checkpoint.
- [export_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/export_onnx.py): Script to convert `.pth` to `.onnx`.
- [quantize_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/quantize_onnx.py): Script to quantize ONNX model to 8-bit integers.
- [test_onnx.py](file:///c:/rishwan/Projects/edge-ai-lab/test_onnx.py): ONNX Runtime verification script.

---

## 💻 How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pillow onnx onnxruntime matplotlib
   ```

2. **Train the Model**:
   ```bash
   python train.py
   ```

3. **Export to ONNX**:
   ```bash
   python export_onnx.py
   ```

4. **Quantize the Model**:
   ```bash
   python quantize_onnx.py
   ```

5. **Run Inference**:
   ```bash
   python test_onnx.py
   ```
