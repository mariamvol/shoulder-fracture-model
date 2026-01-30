import torch
import torch.nn as nn
from torchvision import models

def _build_model():
    m = models.densenet121(weights=None)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, 1)
    return m

def load_model(ckpt_path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)

    model = _build_model().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    meta = {
        "img_size": ck.get("img_size", 224),
        "mean": ck.get("mean", [0.485, 0.456, 0.406]),
        "std":  ck.get("std",  [0.229, 0.224, 0.225]),
        "threshold": ck.get("threshold", 0.5),
        "device": device,
    }
    return model, meta
