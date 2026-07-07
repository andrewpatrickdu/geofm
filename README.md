# GeoFM

GeoFM provides pretrained Prithvi v2 foundation models, distilled variants, and evaluation scripts for cloud classification and cloud segmentation on Sentinel-2 imagery.

## Installation

Create a Conda environment and install the required dependencies:

```bash
conda create --name geofm python=3.10
conda activate geofm

pip install -r requirements.txt
```

## Download Data and Model Weights

Download the datasets, pretrained model weights, and task-specific checkpoints from:

**[GeoFM Resources](https://adelaideuniversity.box.com/s/vjzkyxhcntrw2vksj5ctlfwbtimq4rm5)**

After downloading, place the files in a local directory and update the dataset and checkpoint paths in the evaluation scripts and example commands below.

The download package contains:

```text
geofm-resources/
├── Prithvi_EO_V2_300M.pt
├── Sentinel-2-Cloud-Mask-Catalogue/
├── ckpt-distillation-256/
├── ckpt-cloud-classification-distillation-256/
├── ckpt-cloud-classification-baseline-1024/
└── ckpt-cloud-segmentation-distillation-256/
```

Where:

* `Prithvi_EO_V2_300M.pt` – Original Prithvi v2 foundation model.
* `ckpt-distillation-256/` – Distilled Prithvi v2 encoder checkpoints (256-dimensional embedding).
* `ckpt-cloud-classification-distillation-256/` – Cloud classification head trained on the distilled encoder.
* `ckpt-cloud-classification-baseline-1024/` – Cloud classification head trained on the original Prithvi v2 model.
* `ckpt-cloud-segmentation-distillation-256/` – Cloud segmentation decoder trained on the distilled encoder.
* `Sentinel-2-Cloud-Mask-Catalogue/` – Dataset used for cloud classification and cloud segmentation evaluation.

---

## Dataset Configuration

Before running the evaluation scripts, update the dataset paths to match your local installation of the Sentinel-2 Cloud Mask Catalogue dataset.

### Cloud Classification

Edit `evaluate_cloud_classification.py`:

```python
img_dir = '/path/to/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
lab_dir_70 = '/path/to/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
```

### Cloud Segmentation

Edit `evaluate_cloud_segmentation.py`:

```python
img_dir = '/path/to/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
mask_dir = '/path/to/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/cloudmasks'
lab_dir_70 = '/path/to/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
```

---

## Cloud Classification

### Distilled Prithvi v2 (Embedding Dimension = 256)

Cloud classification using a Prithvi v2 encoder pretrained via Dual-MAE Distillation.

```bash
python evaluate_cloud_classification.py \
    --encoder /path/to/ckpt-distillation-256/student-final.pt \
    --encoder_embed_dim 256 \
    --classifier /path/to/ckpt-cloud-classification-distillation-256/model-final.pt
```

Expected output:

```text
Test accuracy: 87.93%
False positive rate: 3.45%
Test F1: 87.27%
```

### Original Prithvi v2 (Embedding Dimension = 1024)

Cloud classification using the original Prithvi v2 foundation model.

```bash
python evaluate_cloud_classification.py \
    --encoder /path/to/Prithvi_EO_V2_300M.pt \
    --encoder_embed_dim 1024 \
    --classifier /path/to/ckpt-cloud-classification-baseline-1024/model-final.pt
```

Expected output:

```text
Test accuracy: 87.93%
False positive rate: 3.79%
Test F1: 87.36%
```

---

## Cloud Segmentation

### Distilled Prithvi v2 (Embedding Dimension = 256)

Cloud segmentation using a Prithvi v2 encoder pretrained via Dual-MAE Distillation.

```bash
python evaluate_cloud_segmentation.py \
    --encoder /path/to/ckpt-distillation-256/student-final.pt \
    --encoder_embed_dim 256 \
    --classifier /path/to/ckpt-cloud-segmentation-distillation-256/model-best.pt
```

Expected output:

```text
mIoU: 83.84
mean F1: 91.21
OA: 91.21
ACC (cloud): 91.21
FP (cloud): 1.63
```

---

## Training and Reproduction Scripts

The `extra/` directory contains scripts for:

* Prithvi v2 pretraining using Masked Autoencoders (MAE)
* Dual-MAE knowledge distillation
* Fine-tuning downstream task heads

These scripts are provided to support reproducibility and further research.

---

## Models Included

| Model                | Encoder Dimension | Description                                      |
| -------------------- | ----------------- | ------------------------------------------------ |
| Prithvi v2 Original  | 1024              | Original pretrained Prithvi v2 foundation model  |
| Distilled Prithvi v2 | 256               | Compact model obtained via Dual-MAE Distillation |

---
