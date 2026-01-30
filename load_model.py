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

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    # поддержка двух форматов:
    sd = ck.get("state_dict") or ck.get("model_state") or ck

    model = _build_model().to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    meta = {
        "img_size": int(ck.get("img_size", 224)) if isinstance(ck, dict) else 224,
        "mean": [0.485, 0.456, 0.406] if not isinstance(ck, dict) else ck.get("mean", [0.485, 0.456, 0.406]),
        "std":  [0.229, 0.224, 0.225] if not isinstance(ck, dict) else ck.get("std",  [0.229, 0.224, 0.225]),
        "threshold": float(ck.get("threshold", 0.5)) if isinstance(ck, dict) else 0.5,
        "device": device,
    }
    return model, meta
