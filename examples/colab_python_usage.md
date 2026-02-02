# Google Colab Python usage (load once)

This example shows how to load the model once and reuse it multiple times in Python.

---

## Steps

### 1. Setup environment
```python
!git clone https://github.com/mariamvol/shoulder-xray-multitask-model
%cd shoulder-xray-multitask-model
!pip install -r requirements.txt
!wget https://github.com/mariamvol/shoulder-xray-multitask-model/releases/download/v1.1/shoulder_fracture_densenet121_infer.pt
```

### 2. Load model
```python
import torch
from PIL import Image
from torchvision import transforms
from load_model import load_model

model, meta = load_model("shoulder_fracture_densenet121_infer.pt")

tfm = transforms.Compose([
    transforms.Resize((meta["img_size"], meta["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=meta["mean"], std=meta["std"]),
])

print("Model loaded")
```

### 3. Define prediction function
```python
def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(meta["device"])
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    pred = int(prob >= meta["threshold"])
    return prob, pred
```
