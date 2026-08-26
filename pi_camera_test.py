import os
import sys
import time
import numpy as np

# Unicode console support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' is not installed.")
    print("Run: pip install opencv-python")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: 'onnxruntime' is not installed.")
    print("Run: pip install onnxruntime")
    sys.exit(1)

def preprocess_frame(frame):
    # Convert BGR (OpenCV) to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize keeping aspect ratio to 256
    h, w, _ = rgb.shape
    new_w, new_h = (256, int(256 * h / w)) if w < h else (int(256 * w / h), 256)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop 224x224
    start_x = (new_w - 224) // 2
    start_y = (new_h - 224) // 2
    cropped = resized[start_y:start_y+224, start_x:start_x+224]
    
    # Normalize with ImageNet mean and std
    img_data = cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # HWC -> CHW -> NCHW
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

def main():
    model_path = "mobilenet_v2_keyboard_mouse_best_quantized.onnx"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    elif not os.path.exists(model_path):
        model_path = "mobilenet_v2_keyboard_mouse_best.onnx"

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading ONNX Model: {model_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    classes = ['keyboard', 'mouse']

    print("Opening camera (device index 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Warning: Could not open camera device 0.")
        print("Trying device index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No camera detected. Make sure your USB webcam or Pi Camera is connected.")
            return

    # Set camera resolution to 640x480 for smooth performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] Starting Live Classification...")
    print("Press 'q' in the window or Ctrl+C in terminal to exit.")

    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera.")
            break

        t_start = time.perf_counter()
        
        # Preprocess and inference
        input_tensor = preprocess_frame(frame)
        raw_outputs = session.run([output_name], {input_name: input_tensor})
        logits = raw_outputs[0]
        
        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
        
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        confidence = probs[pred_idx] * 100
        
        t_end = time.perf_counter()
        infer_time_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / infer_time_ms if infer_time_ms > 0 else 0
        
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # Draw UI overlay on the frame
        h, w, _ = frame.shape
        
        # Header banner
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        
        # Prediction Text
        color = (0, 255, 0) if confidence > 70 else (0, 200, 255)
        text = f"{pred_label.upper()} ({confidence:.1f}%)"
        cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
        
        # FPS and Latency info
        info_text = f"Latency: {infer_time_ms:.1f}ms | FPS: {avg_fps:.1f}"
        cv2.putText(frame, info_text, (w - 290, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Display window
        cv2.imshow("Raspberry Pi Edge AI - Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stream ended.")

if __name__ == "__main__":
    main()
