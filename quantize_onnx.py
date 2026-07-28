import os
import sys

# Reconfigure stdout/stderr for Windows console unicode support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

def main():
    input_model = "mobilenet_v2_keyboard_mouse.onnx"
    preprocessed_model = "mobilenet_v2_keyboard_mouse_preprocessed.onnx"
    output_model = "mobilenet_v2_keyboard_mouse_quantized.onnx"
    
    if not os.path.exists(input_model):
        print(f"Error: Base ONNX model '{input_model}' not found. Please run export_onnx.py first.")
        return
        
    print(f"1. Preprocessing and running shape inference on '{input_model}'...")
    try:
        quant_pre_process(
            input_model_path=input_model,
            output_model_path=preprocessed_model,
            skip_optimization=False
        )
        print("Preprocessing successful!")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        preprocessed_model = input_model

    print(f"2. Applying 8-bit dynamic quantization...")
    # Run dynamic quantization (converts weight parameters to INT8)
    quantize_dynamic(
        model_input=preprocessed_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8  # Quantize weights to unsigned 8-bit integers
    )
    
    print("\nQuantization completed successfully!")
    
    # Calculate sizes (including external data files if present)
    def get_model_size(base_path):
        size = os.path.getsize(base_path)
        data_path = base_path + ".data"
        if os.path.exists(data_path):
            size += os.path.getsize(data_path)
        return size / (1024 * 1024)
        
    orig_size = get_model_size(input_model)
    quant_size = get_model_size(output_model)
    
    print(f"Original ONNX model total size: {orig_size:.2f} MB")
    print(f"Quantized ONNX model total size: {quant_size:.2f} MB")
    print(f"Size Reduction: {(1 - quant_size/orig_size)*100:.1f}%")

if __name__ == "__main__":
    main()
