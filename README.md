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

## License
MIT License

