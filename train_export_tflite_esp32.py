import os
import sys
import glob
import numpy as np
from PIL import Image

# Force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading TensorFlow...")
import tensorflow as tf

def load_and_preprocess_dataset(img_dir, target_size=(96, 96)):
    classes = ['keyboard', 'mouse']
    images = []
    labels = []
    
    for label_idx, cls in enumerate(classes):
        cls_path = os.path.join(img_dir, cls)
        if not os.path.exists(cls_path):
            continue
        file_list = glob.glob(os.path.join(cls_path, "*.[pP][nN][gG]")) + \
                    glob.glob(os.path.join(cls_path, "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(cls_path, "*.[jJ][pP][eE][gG]"))
        
        for f in file_list:
            try:
                img = Image.open(f).convert("RGB")
                img = img.resize(target_size, Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
                images.append(arr)
                labels.append(label_idx)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, classes

def build_esp32_model(input_shape=(96, 96, 3), num_classes=2):
    # Lightweight MobileNetV2 architecture with 96x96 resolution for ESP32 SRAM
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.35,  # Lightweight 0.35 width multiplier for microcontrollers
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    # Freeze bottom layers, fine-tune top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def convert_to_c_header(tflite_path, header_path, variable_name="g_model"):
    with open(tflite_path, "rb") as f:
        bytes_data = f.read()
        
    hex_array = [f"0x{b:02x}" for b in bytes_data]
    lines = []
    lines.append("// ESP32 TensorFlow Lite Micro Model Data")
    lines.append("// Automatically generated for Edge AI Lab")
    lines.append("#ifndef MODEL_DATA_H")
    lines.append("#define MODEL_DATA_H")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append(f"alignas(16) const unsigned char {variable_name}[] = {{")
    
    # Group in chunks of 12 bytes per line
    chunk_size = 12
    for i in range(0, len(hex_array), chunk_size):
        chunk = hex_array[i:i+chunk_size]
        lines.append("  " + ", ".join(chunk) + ",")
        
    lines.append("};")
    lines.append(f"const unsigned int {variable_name}_len = {len(bytes_data)};")
    lines.append("")
    lines.append("#endif // MODEL_DATA_H")
    
    with open(header_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Saved C++ header file: '{header_path}' ({len(bytes_data)} bytes)")

def main():
    img_dir = "images"
    X, y, classes = load_and_preprocess_dataset(img_dir, target_size=(96, 96))
    print(f"Loaded {len(X)} dataset images for ESP32 model training.")
    
    if len(X) == 0:
        print("Error: No images found in 'images/keyboard' or 'images/mouse'.")
        return
        
    model = build_esp32_model(input_shape=(96, 96, 3), num_classes=len(classes))
    model.summary()
    
    print("\nTraining ESP32 MobileNetV2 (alpha=0.35, 96x96)...")
    model.fit(X, y, epochs=15, batch_size=8, verbose=1)
    
    # Representative dataset generator for INT8 Quantization
    def representative_dataset_gen():
        for i in range(len(X)):
            sample = np.expand_dims(X[i], axis=0)
            yield [sample]
            
    print("\nConverting model to INT8 Quantized TensorFlow Lite (.tflite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    tflite_path = "mobilenet_v2_keyboard_mouse_esp32.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
        
    size_mb = len(tflite_quant_model) / (1024 * 1024)
    print(f"Successfully saved ESP32 INT8 TFLite model to '{tflite_path}' ({size_mb:.2f} MB / {len(tflite_quant_model)} bytes)")
    
    # Generate C Header file for ESP32 Arduino / ESP-IDF
    header_path = "model_data.h"
    convert_to_c_header(tflite_path, header_path)
    
    # Test TFLite model prediction
    print("\nEvaluating INT8 TFLite model on dataset...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    correct = 0
    for i in range(len(X)):
        # Quantize input [0, 1] float to int8
        input_data = (X[i] / input_scale + input_zero_point).astype(np.int8)
        input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Dequantize output
        output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        pred_class = np.argmax(output_float[0])
        gt_class = y[i]
        
        if pred_class == gt_class:
            correct += 1
            
    acc = (correct / len(X)) * 100.0
    print(f"INT8 TFLite ESP32 Model Accuracy: {correct}/{len(X)} ({acc:.2f}%)")

if __name__ == "__main__":
    main()
