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
  - **Grad-CAM++** (Class Activation Mapping)
  - **Integrated Gradients** (Axiomatic Attribution)
  - **Input-Gradient Saliency** (Additional Utility)

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

# Activate the environment
source .venv/bin/activate
```
# 🚀 Usage
## Phase 1: Data Preparation (Strict Disjoint Splits)

The project requires the **DF2023** dataset. The project enforces a strict **Scene-Disjoint Protocol**. You must generate these splits **before** training to ensure no background scenes leak between Train, Val, and Test sets.

### 1.1. Build Master Manifest
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

### 1.2. Generate Scene-Disjoint Partitions
Transform the master manifest into forensic-ready disjoint splits. This script enforces the **80/10/10 split ratio** (Train/Val/Test) defined in the paper, ensuring no source scene overlaps between partitions.
```bash
python scripts/create_splits.py data/manifests/df2023_manifest.csv
```
**Outputs**: train_split.csv, val_split.csv, test_split.csv in data/manifests/splits/.

## Phase 2A: Training (Predictive Models)
We use a **Weighted Hybrid Loss** and enforce determinism for reproducibility.

### Reproducibility Configuration
To strictly replicate the results in **Table 2** of the paper:
* **Precision**: FP32 (AMP Disabled) — *Crucial for Transformer stability.*
* **Gradient Clipping**: Norm capped at 1.0.
* **Loss Weights**: $\lambda_{WCE}=0.5, \lambda_{Dice}=0.5$.

**Note**: For SegFormer (Transformer), you must enable the CUBLAS determinism flag.

### 2A.1. Train SegFormer-B2 (Transformer)
```bash
# Enable Deterministic Mode (Critical for Transformers)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Repeat for each seed
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_segformer_b2_full.yaml train
```
### 2A.2. Train U-Net R34 (CNN Baseline)
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Repeat for each seed
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_unet_r34_full.yaml train
```
**Outputs**: Logs and checkpoints (best.pt, last.pt, config_train.json) are saved to: outputs/<model_name>/seed<SEED>/

## Phase 2B: Leakage Ablation Study (Background Leakage Control)

To scientifically validate the necessity of the **Scene-Disjoint Protocol**, we conduct an ablation study using standard **Random Splits**.
* **Hypothesis:** A model trained on random splits will achieve artificially high scores (>90%) by memorizing background textures ("Background Leakage"), whereas the Scene-Disjoint model measures true forensic generalization.

### 2B.1. Generate Random Splits
Run this script to generate a shuffled partition that ignores scene IDs (simulating "flawed" standard practice).

```bash
# Run the random splitter
python scripts/create_random_splits.py
```

### 2B.2. Train Control Model
Train a single SegFormer baseline on the random splits.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=1337 python -m df2023xai.cli.run_train \
  --config configs/train_segformer_b2_random.yaml train
```

## Phase 3A: Forensic Evaluation

Evaluate the trained models on the **Scene-Disjoint Test Set** (Unseen Scenes), using standard segmentation metrics.

```bash
# Run this for ALL 6 models (3 SegFormer, 3 U-Net).
python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval.yaml
```
Metrics include:
- IoU (Intersection over Union)
- Dice (F1-Score)
- Precision (positive predictive value)
- Recall (sensitivity)
- False Positive Rate (FPR)

## Phase 3B: Random Evaluation

Measure how well a model "cheats" on seen scenes.
```bash
python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval_random.yaml
```
## Phase 3C: Stratified Evaluation (Manipulation Type Analysis)
Global metrics often hide specific failure modes. This phase decomposes performance by manipulation category (Splicing, Copy-Move, Inpainting) to validate the hypothesis that Inpainting is significantly harder to detect than Splicing due to the lack of high-frequency edge artifacts.

```bash
# Run stratified analysis on the Test Set
# (Uses the first model defined in the config)
python -m df2023xai.cli.run_stratified_eval configs/xai_gen.yaml
```

## Phase 4: XAI Faithfulness Audit

Generate attribution maps to visualize why the model flagged specific regions. This step requires a trained model checkpoint.

```bash
# Run this for seed 1337 SegFormer and U-Net
python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml
```
**Supported Methods**: 
- **Grad-CAM++** (Class Activation Mapping)
- **Integrated Gradients** (Axiomatic Attribution)
- **Input-Gradient Saliency** (Additional Utility)

### Audit the decision surface for "Semantic Coherence" using Saliency Total Variation (TV).
- Low TV = Coherent Object Focus (SegFormer)
- High TV = Fragmented Edge Focus (U-Net)

```bash
python -m df2023xai.cli.run_coherence_metric --config configs/forensic_eval.yaml --limit 200
```
## 📦 Repository Structure

```text
df2023-xai/
|-- configs/                            # YAML configuration files
|   |-- train_segformer_full.yaml       # SegFormer hyperparameters
|   |-- train_segformer_random.yaml     # SegFormer hyperparameters for random splits
|   |-- train_unet_full.yaml            # U-Net hyperparameters
|   |-- forensic_eval.yaml              # Forensic Evlauation settings
|   |-- forensic_eval_random.yaml       # Random Evlauation settings
|   `-- xai_gen.yaml                    # XAI settings
|-- data/
|   `-- manifests/
|       `-- df2023_manifest.csv         # Main manifest
|       `-- splits/                     # Official Train/Val/Test CSVs
|           |-- train_split.csv
|           |-- val_split.csv
|           |-- train_split_random.csv 
|           `-- val_split_random.csv
|   |-- train/                         
|   |   |-- images/                    # [Symlink to external DF2023_V15_train/COCO_V15]
|   |   `-- masks/                     # [Symlink to external DF2023_V15_train/COCO_V15_GT]
|   |-- val/                            
|   |   |-- images/                    # [Symlink to external DF2023_V15_val/COCO_V15]
|   |   `-- masks/                     # [Symlink to external DF2023_V15_val/COCO_V15_GT]
|-- scripts/
|   |-- create_splits.py               # Generate scene-disjoint splits
|   |-- create_random_splits.py        # Generate random splits
|-- src/
|   `-- df2023xai/
|       |-- cli/                         # Entry points
|       |   |-- build_manifest.py        # Scans raw dataset to create master CSV
|       |   |-- run_train.py             # Main training launcher
|       |   |-- run_eval.py              # Testing & Metrics
|       |   |-- run_stratified_eval.py   # Run stratified analysis on the Test Set
|       |   |-- run_xai.py               # Attribution generation
|       |   |-- run_coherence_metric.py  # XAI Audit
|       |-- data/
|       |   |-- dataset.py             # Loader with geometric augmentations
|       |   |-- manifest.py            # CSV parsing & stratification
|       |-- eval/
|       |   |-- forensic.py            # Core Evaluation Logic for Phase 2
|       |-- models/
|       |   |-- factory.py             # SMP/Timm wrapper
|       |   `-- segformer.py           # Custom headers (if applicable)
|       |-- train/
|       |   |-- loop.py                # Training & Validation loops
|       |   |-- losses.py              # Hybrid Loss (CrossEntropy + SoftDice)
|       `-- xai/                       # Explainability Engine
|           |-- attention_rollout.py   # Input-Gradient Saliency (Transformer Utility)
|           |-- ig.py                  # Implementation of Integrated Gradients (IG)
|           |-- gradcampp.py           # Implementation of Grad-CAM++
|-- outputs/                           # Training artifacts (Gitignored)
`-- requirements.txt                   # Verified dependencies


### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 📜 Citation
Please cite the software using the generated DOI:
> Aoudi, S. (2026). *DF2023-XAI: Explainable Deep Learning for Image Forgery Localization (v1.0.1)* 
