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
# 🚀 Usage
## Phase 1: Data Preparation (Strict Disjoint Splits)

The project requires the **DF2023** dataset. The project enforces a strict **Scene-Disjoint Protocol**. You must generate these splits **before** training to ensure no background scenes leak between Train, Val, and Test sets.

### 1. Build Master Manifest
Scan raw directories to generate the master index. A single "Master CSV" containing all file paths and metadata.

```bash
# Index the raw dataset (update paths to match your local storage)
python -m df2023xai.cli.build_manifest \
  --images data/train/images \
  --masks data/train/masks \
  --images-val data/val/images \
  --masks-val data/val/masks \
  --out data/manifests/df2023_manifest.csv
```

### 2. Generate Scene-Disjoint Partitions
Transform the master manifest into forensic-ready disjoint splits.
```bash
python scripts/create_splits.py data/manifests/df2023_manifest.csv
```
**Outputs**: train_split.csv, val_split.csv, test_split.csv in data/manifests/splits/.

## Phase 2: Training (Predictive Models)
We use a Binary Hybrid Loss and enforce determinism for reproducibility.

**Note 1**: For SegFormer (Transformer), you must enable the CUBLAS determinism flag to ensure attention map consistency.We support reproducible training with dynamic seeding. The pipeline uses Automatic Mixed Precision (AMP) and Distributed Data Parallel (compatible) loaders.

**Note 2**: DL training is stochastic (random weight initialization, data shuffling). A single high score could be a statistical fluke (a "lucky initialization"). By training the identical model three times with different random starts, you calculate the Mean $\pm$ Standard Deviation (e.g., $84.2 \pm 0.3\%$). This proves your method is stable and consistently superior, not just lucky. The following seed numbers are mathematically arbitrary but culturally significant in computer science. They are chosen to be distinct and easily memorable: **1337**, **2027**, and **3141**.

```bash
# Enable Deterministic Mode (Critical for Transformers)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Train SegFormer-B2 (Transformer)
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_segformer_b2_full.yaml train

# Train U-Net R34 (CNN Baseline)
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_unet_r34_full.yaml train
```
**Outputs**: Logs and checkpoints (best.pt, last.pt, config_train.json) are saved to: outputs/<model_name>/seed<SEED>/

## Phase 3: Forensic Evaluation

Evaluate the trained models on the **Scene-Disjoint Test Set** (Unseen Scenes), using standard segmentation metrics (IoU, Dice, Pixel-F1).

```bash
python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval.yaml
```
Metrics include:
- IoU
- Dice
- Pixel-F1

## Phase 4: XAI Faithfulness Audit

Generate attribution maps to visualize why the model flagged specific regions. This step requires a trained model checkpoint.

```bash
python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml
```
**Supported Methods**: 
- **Grad-CAM++** (Class Activation Mapping)
- **Integrated Gradients** (Axiomatic Attribution)
- **SHAP** (Game Theoretic Estimation)
- **Attention Rollout **(Transformer Saliency)

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
|   |-- create_splits.py               # Generate splits
|-- src/
|   `-- df2023xai/
|       |-- cli/                       # Entry points
|       |   |-- build_manifest.py      # Scans raw dataset to create master CSV
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
