import torch
from torchvision import models

# Download pretrained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Inference mode
model.eval()

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy)

print(output.shape)