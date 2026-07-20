import os
print("Running:", os.path.abspath(__file__))

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

# 3. Load image and ensure it has 3 channels (RGB)
image_path = "images/apple.png"
image = Image.open(image_path).convert("RGB")

# 4. Define preprocessing pipeline
preprocess = weights.transforms()

resized = transforms.Resize((224, 224))(image)
resized.show() 


# 5. Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 6. Perform inference without tracking gradients
with torch.no_grad():
    output = model(input_batch)

# 7. Convert output logits to probabilities via softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 8. Retrieve categories and display Top 5 predictions
categories = weights.meta["categories"]
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\nTop 5 Predictions:")
for i in range(top5_prob.size(0)):
    category = categories[top5_catid[i]]
    percentage = top5_prob[i].item() * 100
    print(f"{category:<25} {percentage:.2f}%")
