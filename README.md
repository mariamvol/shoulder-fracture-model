# Shoulder fracture classifier

Inference-only deep learning model for shoulder fracture detection on X-ray images.

This repository provides a ready-to-use interface for model inference.
Training code is intentionally not included.

---

## Model description

- Architecture: DenseNet-121
- Task: binary classification (fracture / no fracture)
- Input: RGB X-ray image, resized to 224×224
- Output: probability of shoulder fracture

---

## Installation

```bash
pip install -r requirements.txt
```

## Examples

- Google Colab quick start: `examples/colab_quickstart.md`
- Google Colab Python usage (load once): `examples/colab_python_usage.md`

--- 

# v2 — multi-head shoulder X-ray classifier (inference only)

Outputs:
- **Fracture** (0/1)
- **Projection**: **S** (1) or **D** (0)
- **Internal fixation hardware** (0/1)

## Run

From repository root:
```bash
python v2/infer_one_v2.py --ckpt shoulder_3heads_densenet121_infer.pt --img xray.png
```

---

## License
MIT License

