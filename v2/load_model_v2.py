import torch
from typing import Tuple, Dict, Any

from .model_v2 import build_model_v2


def load_model_v2(ckpt_path: str, device: str | torch.device = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads v2 multi-head model.

    Supports checkpoints:
      1) {"model_state": state_dict, "img_size": 224, "hpo_params": {...}, ...}
      2) plain state_dict (inference-only)
    """
    device = torch.device(device)
    ck = torch.load(ckpt_path, map_location=device)

    model = build_model_v2(pretrained=False).to(device)

    # case 1: dict with model_state
    if isinstance(ck, dict) and "model_state" in ck:
        state = ck["model_state"]
        model.load_state_dict(state, strict=True)
        img_size = int(ck.get("img_size", 224))
        meta = {
            "img_size": img_size,
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
            "has_heads": ["fracture", "projection", "hardware"],
            "ckpt_type": "train_ckpt",
        }
        if "hpo_params" in ck and isinstance(ck["hpo_params"], dict):
            meta["hpo_params"] = ck["hpo_params"]
        return model.eval(), meta

    # case 2: plain state_dict
    if isinstance(ck, dict):
        try:
            model.load_state_dict(ck, strict=True)
            meta = {
                "img_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225],
                "has_heads": ["fracture", "projection", "hardware"],
                "ckpt_type": "state_dict",
            }
            return model.eval(), meta
        except Exception as e:
            raise RuntimeError(f"Не смог загрузить чекпоинт: {ckpt_path}. Формат неизвестен. Ошибка: {e}")

    raise RuntimeError(f"Неизвестный формат чекпоинта: {type(ck)}")

