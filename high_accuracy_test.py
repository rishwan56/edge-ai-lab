import os
print("Running:", os.path.abspath(__file__))

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# 1. Load high-accuracy model (EfficientNet_V2_S)
# EfficientNetV2-S offers significantly higher top-1 accuracy (~84.3%) compared to MobileNetV2 (~72%)
weights = models.EfficientNet_V2_S_Weights.DEFAULT
model = models.efficientnet_v2_s(weights=weights)
model.eval()

# 2. Load image and ensure RGB channels
image_path = "images/mouse/mouse1.png"
if not os.path.exists(image_path):
    # Fallback if specific screenshot is missing
    image_path = "images/Passport_Size_photo_rishwan.jpg"

image = Image.open(image_path).convert("RGB")

# 3. Use official weights preprocessing pipeline
preprocess = weights.transforms()

# Optional: preview resized visualization
resized = transforms.Resize((224, 224))(image)
resized.show()

# 4. Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 5. Perform inference without tracking gradients
with torch.no_grad():
    output = model(input_batch)

# 6. Convert output logits to probabilities via softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 7. Retrieve categories and display Top 5 predictions
categories = weights.meta["categories"]
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\nTop 5 Predictions (High Accuracy Model - EfficientNetV2):")
for i in range(top5_prob.size(0)):
    category = categories[top5_catid[i]]
    percentage = top5_prob[i].item() * 100
    print(f"{category:<25} {percentage:.2f}%")
