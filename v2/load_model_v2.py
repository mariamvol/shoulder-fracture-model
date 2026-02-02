import torch
from .model_v2 import build_model

DEFAULT_META = {
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "heads": ["fracture", "projection", "hardware"],
    "projection_map": {0: "D", 1: "S"},
}


def load_model(ckpt_path: str, device: str | torch.device = "cpu"):
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model().to(device)

    # 1) {"model_state": ...}
    # 2) plain state_dict
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    meta = dict(DEFAULT_META)

    if isinstance(ckpt, dict):
        if "img_size" in ckpt:
            meta["img_size"] = int(ckpt["img_size"])
        if "mean" in ckpt:
            meta["mean"] = list(ckpt["mean"])
        if "std" in ckpt:
            meta["std"] = list(ckpt["std"])

    return model, meta
