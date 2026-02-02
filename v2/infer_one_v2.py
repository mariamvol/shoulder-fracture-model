import argparse
import torch
from PIL import Image
from torchvision import transforms

from .load_model_v2 import load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--img", required=True, help="Path to image")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    model, meta = load_model(args.ckpt, device=args.device)

    tfm = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
    ])

    img = Image.open(args.img).convert("RGB")
    x = tfm(img).unsqueeze(0).to(args.device)

    out = model(x)
    p_frac = torch.sigmoid(out["fracture"]).item()
    p_proj = torch.sigmoid(out["projection"]).item()
    p_hw   = torch.sigmoid(out["hardware"]).item()

    frac_pred = int(p_frac >= 0.5)
    proj_pred = int(p_proj >= 0.5)
    hw_pred   = int(p_hw >= 0.5)

    proj_label = meta["projection_map"].get(proj_pred, str(proj_pred))

    print("=== Multi-head prediction ===")
    print(f"Fracture:   prob={p_frac:.4f} | pred={frac_pred}")
    print(f"Projection: prob={p_proj:.4f} | pred={proj_label} (S=1, D=0)")
    print(f"Hardware:   prob={p_hw:.4f} | pred={hw_pred}")


if __name__ == "__main__":
    main()
