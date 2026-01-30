import argparse
import torch
from PIL import Image
from torchvision import transforms

from load_model import load_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()

    model, meta = load_model(args.ckpt)
    thr = meta["threshold"] if args.thr is None else args.thr

    tfm = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
    ])

    img = Image.open(args.img).convert("RGB")
    x = tfm(img).unsqueeze(0).to(meta["device"])

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    pred = int(prob >= thr)
    print(f"prob={prob:.4f} pred={pred}")

if __name__ == "__main__":
    main()
