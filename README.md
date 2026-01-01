# DF2023-XAI: Explainable Deep Learning for Image Forgery Localization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: Forensic Segmentation](https://img.shields.io/badge/Task-Forensic_Segmentation-blue.svg)]()

> **Official PyTorch Implementation** of the paper  
> *“Explainable Deep Learning for Image Forgery Detection: Interpreting Manipulation Cues in DF2023”*

---

## 📌 Overview

This repository addresses the **black-box problem in digital image forensics** by combining high-performance semantic segmentation models with post-hoc Explainable AI (XAI).

We benchmark:

- **SegFormer-B2** (Transformer-based)
- **U-Net R34** (CNN-based baseline)

on the **DF2023** dataset, evaluating not only localization accuracy (IoU, Dice, Pixel-F1) but also the **faithfulness and interpretability of forensic explanations**.

### Key Contributions

- **Architectural Benchmarking**  
  Comparative analysis of CNN vs. Transformer backbones for image forgery localization.

- **Forensic Rigor**  
  Scene-level *disjoint splitting* to prevent data leakage via background memorization.

- **Explainability**  
  Integrated XAI pipeline producing pixel-level attribution maps using:
  - SHAP  
  - Integrated Gradients  
  - Grad-CAM++

- **Robustness**  
  Training with a differentiable **Hybrid Loss** (Cross-Entropy + Soft Dice) and geometric data augmentations.

---

## ⚙️ Installation

### Prerequisites

- Python **3.10+**
- CUDA-enabled GPU  
  (Tested on **NVIDIA L40, 48GB VRAM**)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/sameraoudi/df2023-xai.git
cd df2023-xai

# Install dependencies
pip install -r requirements.txt
```
### 📂 Data Preparation

The project requires the **DF2023** dataset. We enforce a strict **80/10/10** split ensuring that all manipulations derived from the same source scene remain in the same partition.

### 1. Build Master Manifest
First, scan the raw image and mask directories to generate a single "Master CSV" containing all file paths and metadata.

```bash
# Index the raw dataset (update paths to match your local storage)
python -m df2023xai.cli.build_manifest \
  --images data/train/images \
  --masks data/train/masks \
  --images-val data/val/images \
  --masks-val data/val/masks \
  --out data/manifests/df2023_manifest.csv
```

### 2. Generate Reproducible Splits

After building the master manifest, generate the physical training and validation split files. This step ensures that the training script loads a deterministic, version-controlled subset of the data, rather than relying on random runtime splitting.

```bash
python scripts/create_splits.py data/manifests/df2023_manifest.csv
```
  
## 🚀 Usage
### Phase 1: Training (Predictive Models)

We support reproducible training with dynamic seeding. The pipeline uses Automatic Mixed Precision (AMP) and Distributed Data Parallel (compatible) loaders.

**Train SegFormer-B2 (Transformer)**
```bash
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_segformer_b2_full.yaml train
```
**Train U-Net R34 (CNN Baseline)**
```bash
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_unet_r34_full.yaml train
```
**Outputs (logs + checkpoints) are saved to:**
```bash
outputs/<model_name>/seed<SEED>/
```
including best.pt and last.pt.

### Phase 2: Forensic Evaluation

Evaluate the trained models on the Test set using standard segmentation metrics (IoU, Dice, Pixel-F1).

```bash
python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval.yaml
```

Metrics include:
- IoU
- Dice
- Pixel-F1

### Phase 3: XAI Generation

Generate attribution maps to visualize why the model flagged specific regions.

```bash
python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml
```
Supported Methods: 
- SHAP
- IntegratedGradients
- GradCAM++
- AttentionRollout

## 📦 Repository Structure

```text
df2023-xai/
|-- configs/                            # YAML configuration files
|   |-- train_segformer.yaml            # SegFormer hyperparameters
|   |-- train_unet.yaml                 # U-Net hyperparameters
|   |-- forensic_eval.yaml              # Forensic Evlauation settings
|   `-- xai_gen.yaml                    # XAI settings
|-- data/
|   `-- manifests/
|       `-- splits/                     # Official Train/Val/Test CSVs
|           |-- train_v2.csv
|           |-- val_v2.csv
|           `-- test_v2.csv
|-- scripts/
|   `-- create_splits.py               # Geberate splits
|-- src/
|   `-- df2023xai/
|       |-- cli/                       # Entry points
|       |   |-- build_manifest.py      # Scans raw dataset to create master CSV
|       |   `-- split_scenes.py        # Disjoint scene splitting logic
|       |   |-- run_train.py           # Main training launcher
|       |   |-- run_eval.py            # Testing & Metrics
|       |   |-- run_xai.py             # Attribution generation
|       |-- data/
|       |   |-- dataset.py             # Loader with geometric augmentations
|       |   `-- manifest.py            # CSV parsing & stratification
|       |-- eval/
|       |   |-- forensic.py            # Core Evaluation Logic for Phase 2
|       |-- models/
|       |   |-- factory.py             # SMP/Timm wrapper
|       |   `-- segformer.py           # Custom headers (if applicable)
|       |-- train/
|       |   |-- loop.py                # Training & Validation loops
|       |   `-- losses.py              # Hybrid Loss (CrossEntropy + SoftDice)
|       `-- xai/                       # Explainability Engine
|           |-- attention_rollout.py   # Robust Saliency & Attention approximation for Transformers
|           |-- ig.py                  # Implementation of Integrated Gradients (IG)
|           |-- gradcampp.py           # Implementation of Grad-CAM++
|           `-- shap.py                # Wrapper for SHAP 
|-- outputs/                           # Training artifacts (Gitignored)
`-- requirements.txt                   # Verified dependencies


### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 📜 Citation
Please cite the software using the generated DOI:
> Aoudi, S. (2026). *DF2023-XAI: Explainable Deep Learning for Image Forgery Localization (v1.0.1)* 
