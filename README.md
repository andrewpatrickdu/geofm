# GeoFM

GeoFM provides pretrained Prithvi v2 model, distilled variants, and evaluation scripts for cloud classification and cloud segmentation.

## Installation

Create a new Conda environment and install the required dependencies:

```bash
conda create --name geofm python=3.10
conda activate geofm

pip install -r requirements.txt
```

## Download Weights and Datasets

Download the pretrained model weights and datasets from:

**[Download Link Here]**

After downloading, update the paths in the commands below as required.

---

## Cloud Classification

### Distilled Prithvi v2 (Embedding Dimension = 256)

Cloud classification using a Prithvi v2 encoder distilled via Dual-MAE Distillation.

```bash
python evaluate_cloud_classification.py \
    --encoder /path/to/ckpt-distillation-256/student-final.pt \
    --encoder_embed_dim 256 \
    --classifier /path/to/ckpt-cloud-classification-distillation-256/model-final.pt
```

### Original Prithvi v2 (Embedding Dimension = 1024)

Cloud classification using the original Prithvi v2 model.

```bash
python evaluate_cloud_classification.py \
    --encoder /path/to/Prithvi_EO_V2_300M.pt \
    --encoder_embed_dim 1024 \
    --classifier /path/to/ckpt-cloud-classification-baseline-1024/model-final.pt
```

---

## Cloud Segmentation

### Distilled Prithvi v2 (Embedding Dimension = 256)

Cloud segmentation using a Prithvi v2 encoder distilled via Dual-MAE Distillation.

```bash
python evaluate_cloud_segmentation.py \
    --encoder /path/to/ckpt-distillation-256/student-final.pt \
    --encoder_embed_dim 256 \
    --classifier /path/to/ckpt-cloud-classification-distillation-256/model-best.pt
```

---

## Additional Training Scripts

The `extra/` directory contains scripts for:

* Prithvi v2 pretraining using Masked Autoencoders (MAE)
* Dual-MAE knowledge distillation
* Fine-tuning downstream task heads
* Training and evaluation workflows

These scripts are provided for research and reproducibility purposes.

---

## Models Included

| Model                | Encoder Dimension | Description                                      |
| -------------------- | ----------------- | ------------------------------------------------ |
| Prithvi v2 Original  | 1024              | Original pretrained Prithvi v2 model             |
| Distilled Prithvi v2 | 256               | Compact model obtained via Dual-MAE Distillation |

---

## Citation

If you use this repository in your research, please cite the associated publication.

```bibtex
[Citation to be added]
```
