import os
print("Running:", os.path.abspath(__file__))

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

model.eval()

image = Image.open("images/whatsappimage.jpeg")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]) 

input_tensor = preprocess(image)

input_batch = input_tensor.unsqueeze(0)
#print(input_batch.shape)

with torch.no_grad():
    output = model(input_batch)
predicted_index = output.argmax(dim=1).item()

print(predicted_index)

categories = weights.meta["categories"]
predicted_label = categories[predicted_index]
probabilities = torch.nn.functional.softmax(output[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)

for prob, idx in zip(top5_prob, top5_catid):
    print(f"{categories[idx]} : {prob.item()*100:.2f}%")
