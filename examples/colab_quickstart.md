# Google Colab quick start

This example shows how to run the model in Google Colab using a single command.

---

## Steps

Open a new Google Colab notebook and run the following cells.

### 1. Clone repository and install dependencies
```python
!git clone https://github.com/mariamvol/shoulder-fracture-model
%cd shoulder-fracture-model
!pip install -r requirements.txt
```
### 2. Download model weights
```python
!wget https://github.com/mariamvol/shoulder-fracture-model/releases/download/v1.1/shoulder_fracture_densenet121_infer.pt
```

### 3. Upload an X-ray image
```python
from google.colab import files
files.upload()  # upload xray.png
```

### 4. Run inference
```python
!python infer_one.py --ckpt shoulder_fracture_densenet121_infer.pt --img xray.png
```

---

## v2 (multi-head)

```bash
!git clone https://github.com/<you>/shoulder-fracture-model
%cd shoulder-fracture-model
!pip install -r requirements.txt

# download weights (from Releases)
!wget <release_link_to>/shoulder_3heads_densenet121_infer.pt

# upload image (xray.png)
from google.colab import files
files.upload()

!python -m v2.infer_one_v2 --ckpt shoulder_3heads_densenet121_infer.pt --img xray.png

