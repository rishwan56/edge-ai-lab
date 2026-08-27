/*
 * ESP32-CAM Keyboard vs Mouse Detection
 * Powered by TensorFlow Lite Micro & MobileNetV2 (0.35 alpha)
 * 
 * Hardware: ESP32-CAM (AI Thinker / ESP32-S3 Eye)
 * Requirements:
 *  - Arduino IDE with ESP32 board support (version 2.0+)
 *  - TensorFlowLite_ESP32 or EloquentTinyML library
 *  - Include model_data.h in the same folder as this sketch
 */

#include "esp_camera.h"
#include <Arduino.h>

// Include TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include quantized model header
#include "model_data.h"

// ----------------------------------------------------
// AI-THINKER ESP32-CAM Pin Configuration
// ----------------------------------------------------
#define PWDN_GPIO_NUM    32
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27

#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// Onboard LED indicator (GPIO 33 on AI Thinker, active LOW)
#define LED_BUILTIN_PIN  33

// Model Parameters
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 3;

// Allocate Tensor Arena memory for TFLite Micro operations
// Using PSRAM if available, otherwise internal SRAM
constexpr int kTensorArenaSize = 160 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// TFLite pointers
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
static tflite::AllOpsResolver resolver;

const char* LABELS[] = {"KEYBOARD", "MOUSE"};

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; // Capture in RGB565 format

  // Use 96x96 frame size matching model input
  if (psramFound()) {
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    while (true);
  }
  Serial.println("✓ Camera initialized successfully.");
}

void setupTFLite() {
  // Load model flatbuffer from model_data.h
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch! Model: %d, Supported: %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    while (true);
  }

  // Create Micro Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (true);
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  Serial.println("✓ TFLite Micro Interpreter initialized successfully.");
  Serial.printf("Input tensor size: %d bytes, Type: %d\n", input_tensor->bytes, input_tensor->type);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=============================================");
  Serial.println("   ESP32-CAM KEYBOARD vs MOUSE DETECTOR      ");
  Serial.println("=============================================");

  pinMode(LED_BUILTIN_PIN, OUTPUT);
  digitalWrite(LED_BUILTIN_PIN, HIGH); // Off

  setupCamera();
  setupTFLite();
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed!");
    delay(1000);
    return;
  }

  uint32_t t_start = millis();

  // Quantization scale and zero point for INT8 model
  float input_scale = input_tensor->params.scale;
  int32_t input_zero_point = input_tensor->params.zero_point;

  // Preprocess RGB565 framebuffer into 96x96x3 INT8 input tensor [-1, 1]
  int8_t* dst = input_tensor->data.int8;
  uint16_t* src = (uint16_t*)fb->buf;

  for (int i = 0; i < kNumRows * kNumCols; i++) {
    uint16_t pixel = src[i];
    // Extract RGB components from RGB565
    float r = (float)(((pixel >> 11) & 0x1F) << 3);
    float g = (float)(((pixel >> 5) & 0x3F) << 2);
    float b = (float)((pixel & 0x1F) << 3);

    // Normalize to [-1, 1] and quantize to int8
    dst[i * 3 + 0] = (int8_t)(((r / 127.5f) - 1.0f) / input_scale + input_zero_point);
    dst[i * 3 + 1] = (int8_t)(((g / 127.5f) - 1.0f) / input_scale + input_zero_point);
    dst[i * 3 + 2] = (int8_t)(((b / 127.5f) - 1.0f) / input_scale + input_zero_point);
  }

  // Release camera buffer
  esp_camera_fb_return(fb);

  // Run TFLite inference on ESP32
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  uint32_t latency = millis() - t_start;

  // Read output probabilities (Dequantize int8 outputs)
  float output_scale = output_tensor->params.scale;
  int32_t output_zero_point = output_tensor->params.zero_point;

  int8_t keyboard_score_int = output_tensor->data.int8[0];
  int8_t mouse_score_int = output_tensor->data.int8[1];

  float keyboard_prob = (keyboard_score_int - output_zero_point) * output_scale;
  float mouse_prob = (mouse_score_int - output_zero_point) * output_scale;

  int pred_idx = (mouse_prob > keyboard_prob) ? 1 : 0;
  float max_prob = (pred_idx == 1) ? mouse_prob : keyboard_prob;

  // Print results
  Serial.printf("[%lu ms] Prediction: %-8s | Confidence: %5.1f%% | Keyboard: %5.1f%%, Mouse: %5.1f%%\n",
                latency, LABELS[pred_idx], max_prob * 100.0f, keyboard_prob * 100.0f, mouse_prob * 100.0f);

  // Flash onboard LED if Keyboard or Mouse detected with high confidence (>70%)
  if (max_prob > 0.70f) {
    digitalWrite(LED_BUILTIN_PIN, LOW); // ON
    delay(50);
    digitalWrite(LED_BUILTIN_PIN, HIGH); // OFF
  }

  delay(500); // 2 FPS loop delay
}
