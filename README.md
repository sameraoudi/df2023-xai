# DF2023-XAI: Auditing Image Forgery Localization via Scene-Disjoint Evaluation and XAI Faithfulness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Task: Forensic Segmentation](https://img.shields.io/badge/Task-Forensic_Segmentation-darkblue.svg)]()
[![IEEE Access](https://img.shields.io/badge/Published-IEEE_Access-green.svg)]()

> **Official PyTorch implementation** of the paper  
> *"Beyond Accuracy: Auditing Image Forgery Localization via Scene-Disjoint Evaluation and XAI Faithfulness"*  
> Aoudi, S., Saleel, A. P., Al Barghuthi, N., & Alamir, O. (2026). TBD.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Hardware and Software Requirements](#hardware-and-software-requirements)
5. [Installation](#installation)
6. [Dataset Setup](#dataset-setup)
7. [Experiment Pipeline](#experiment-pipeline)
   - [Phase 1: Data Preparation](#phase-1-data-preparation)
   - [Phase 2: Model Training](#phase-2-model-training)
   - [Phase 3: Forensic Evaluation](#phase-3-forensic-evaluation)
   - [Phase 4: XAI Generation](#phase-4-xai-generation)
   - [Phase 5: Paper Statistics Pipeline](#phase-5-paper-statistics-pipeline)
   - [Phase 6: Supporting Analyses](#phase-6-supporting-analyses)
   - [Phase 7: Manuscript Figures and Tables](#phase-7-manuscript-figures-and-tables)
8. [Reproducibility](#reproducibility)
9. [Expected Outputs and Results](#expected-outputs-and-results)
10. [Known Issues and Constraints](#known-issues-and-constraints)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)

---

## Overview

This repository provides the complete implementation of **DF2023-XAI**, a forensic evaluation framework that exposes a critical weakness in standard image forgery localization benchmarks: high pixel-level accuracy does not guarantee that a model has learned semantically meaningful manipulation traces.

We term this the **Metric Trap** — a systemic evaluation failure where models achieve competitive leaderboard performance by exploiting spurious background correlations rather than learning intrinsic manipulation artifacts. To address this, we propose two interlocking validity constraints:

1. **Scene-Disjoint Evaluation**: A strict data partitioning protocol that eliminates background leakage and supervisory conflict by ensuring no source scene appears in more than one partition.

2. **XAI Faithfulness Audit**: A quantitative audit of model attribution using Grad-CAM++ and Integrated Gradients, with Saliency Total Variation (TV) as a spatial coherence proxy to distinguish semantically coherent "Blob" attribution from fragmented "Wireframe" boundary-fixation.

We benchmark two architectures under these constraints on the 1M-image DF2023 dataset:
- **SegFormer-B2** — hierarchical Vision Transformer
- **U-Net (ResNet-34)** — standard CNN baseline

The results reveal a **Metric Paradox**: the CNN achieves marginally higher pixel-level precision, but the XAI audit confirms it functions primarily as a boundary detector. The Transformer exhibits 20% lower Saliency TV, indicating structurally coherent attribution aligned with the manipulated object as a whole.

---

## Key Results

### Segmentation Metrics — Scene-Disjoint Test Set (100,470 images, mean ± std across 3 seeds)

| Model | Precision | Recall | F1-Score | IoU | FPR ↓ | Saliency TV ↓ |
|---|---|---|---|---|---|---|
| U-Net (ResNet-34) | 0.9271 ± 0.0046 | 0.9089 ± 0.0040 | 0.9093 ± 0.0015 | 0.8669 ± 0.0014 | 0.0109 ± 0.0011 | 0.0119 ± 0.0003 |
| SegFormer-B2 | 0.9181 ± 0.0009 | 0.9079 ± 0.0029 | 0.9054 ± 0.0012 | 0.8582 ± 0.0014 | 0.0126 ± 0.0005 | 0.0095 ± 0.0001 |
| Δ | −0.90% | −0.10% | −0.39% | −0.87% | +0.0017 | **−20.0%** |

Saliency TV: 95% CI via bootstrap (B=10,000) over 301,410 pooled images.  
U-Net: [0.01186, 0.01188] · SegFormer-B2: [0.00949, 0.00950]

### Directional Consistency (TV)

| Seed | TV(U-Net) | TV(SegFormer) | Δ | Result |
|---|---|---|---|---|
| 1337 | 0.01156 | 0.00953 | +17.6% | PASS ✓ |
| 2027 | 0.01216 | 0.00956 | +21.4% | PASS ✓ |
| 3141 | 0.01191 | 0.00940 | +21.1% | PASS ✓ |

### Parameter Randomization Sanity Check

| Model | Random TV | Trained TV | Δ | Result |
|---|---|---|---|---|
| U-Net | 0.0431 | 0.0119 | +263% | SENSITIVE ✓ |
| SegFormer-B2 | 0.0712 | 0.0095 | +650% | SENSITIVE ✓ |

### Frequency Experiment (Enhancement Subset)

| Model | IoU | Mean TV |
|---|---|---|
| SegFormer-B2 RGB-only | 0.7756 | 0.008896 |
| SegFormer-B2 SRM+RGB | 0.7967 (+2.7%) | 0.008571 (−3.66%) |

---

## Repository Structure

```
df2023-xai/
├── configs/
│   ├── phase1.yaml                      # Data symlink and manifest config
│   ├── train_segformer_b2_full.yaml     # CANONICAL SegFormer-B2 training config
│   ├── train_segformer_b2_random.yaml   # Ablation: random-split training
│   ├── train_unet_r34_full.yaml         # CANONICAL U-Net-R34 training config
│   ├── forensic_eval.yaml               # Scene-disjoint evaluation (6 models)
│   ├── forensic_eval_random.yaml        # Ablation evaluation (random splits)
│   └── xai_gen.yaml                     # XAI generation (8 samples, 2 models)
├── data/
│   ├── manifests/
│   │   ├── df2023_manifest.csv          # Combined train+val manifest
│   │   └── splits/
│   │       ├── train_split.csv          # 803,965 images, 800,000 scenes
│   │       ├── val_split.csv            # 100,514 images, 100,000 scenes
│   │       ├── test_split.csv           # 100,470 images, 100,000 scenes
│   │       ├── train_split_random.csv   # Ablation: random split
│   │       ├── val_split_random.csv
│   │       └── test_split_random.csv
│   ├── train/images/                    # → symlink to COCO_V15 train
│   ├── train/masks/                     # → symlink to COCO_V15_GT train
│   ├── val/images/                      # → symlink to COCO_V15 val
│   └── val/masks/                       # → symlink to COCO_V15_GT val
├── scripts/
│   ├── prepare_data.py                  # Unified data preparation pipeline
│   │                                    #   subcommands: add-scene-ids, scene-splits,
│   │                                    #                random-splits, all
│   ├── paper_stats.py                   # Unified paper statistics pipeline (Steps 1–5)
│   │                                    #   subcommands: verify-checkpoints, compute-tv,
│   │                                    #                bootstrap-ci, metric-std,
│   │                                    #                directional-check, run-all
│   ├── run_analyses.py                  # Unified supporting analyses pipeline
│   │                                    #   subcommands: per-sample-iou, sanity-check,
│   │                                    #                coherence-check, freq-experiment,
│   │                                    #                run-all
│   ├── generate_manuscript.py           # Unified manuscript figure/table generator
│   │                                    #   subcommands: figure3, table-dataset,
│   │                                    #                table-hyperparams, all
│   ├── run_paper_stats.sh               # Shell wrapper → paper_stats.py run-all
│   ├── run_additions.sh                 # Shell wrapper → run_analyses.py run-all
│   └── run_post_step2.sh                # Watcher: polls for TV arrays, triggers Steps 3–5
├── src/df2023xai/
│   ├── cli/                             # Installed entry points (dfx-*)
│   │   ├── symlink_data.py              # dfx-symlink-data
│   │   ├── build_manifest.py            # dfx-build-manifest
│   │   ├── run_train.py                 # dfx-train
│   │   ├── run_xai.py                   # dfx-xai
│   │   ├── run_forensic_eval.py         # dfx-forensic
│   │   └── split_scenes.py              # dfx-split-scenes
│   ├── data/                            # Dataset, manifest, integrity checks
│   ├── eval/                            # Metrics and batch evaluation
│   ├── models/                          # SegFormer-B2 and U-Net wrappers
│   ├── train/                           # Training loop and hybrid loss
│   ├── utils/                           # Logging, paths, seeds, registry
│   └── xai/                             # Grad-CAM++, IG, Attention Rollout
├── outputs/                             # Training and evaluation artifacts (gitignored)
├── docs/                                # MkDocs documentation
├── tests/                               # Pytest suite
├── model_selection/                     # Architecture selection criteria and report
├── pyproject.toml                       # Canonical dependency specification
└── Makefile                             # Convenience targets
```

---

## Hardware and Software Requirements

### Hardware

| Component | Tested Configuration | Minimum Recommended |
|---|---|---|
| GPU | NVIDIA L40-12Q, 12 GB vGPU | 12 GB VRAM (use `--chunk-size 5` for TV computation) |
| CPU | Any modern multi-core | 4+ cores |
| RAM | 32 GB | 16 GB |
| Disk | ~500 GB free | ~500 GB for dataset + outputs |

> **⚠️ vGPU Constraint (CRITICAL):** If running on an NVIDIA vGPU profile (e.g., L40-12Q), do **NOT** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This causes an immediate `RuntimeError: CUDA driver error: operation not supported` because `cuMemCreate` is unsupported on this vGPU profile. Omit this variable entirely.

### Software

| Package | Tested Version | Required |
|---|---|---|
| Python | 3.10.12 | ≥ 3.10 |
| CUDA | 12.8 | ≥ 11.8 |
| cuDNN | 9.1.9 | Compatible with CUDA |
| PyTorch | 2.2+ | ≥ 2.2 |
| torchvision | 0.17+ | ≥ 0.17 |
| timm | 1.0.22 | ≥ 0.9.16 |
| segmentation-models-pytorch | 0.5.0 | ≥ 0.3.3 |
| captum | 0.8.0 | ≥ 0.7 |
| numpy | 1.26.4 | ≥ 1.24 |
| pandas | 2.3.3 | ≥ 2.1 |

See `pyproject.toml` for the complete dependency specification.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sameraoudi/df2023-xai.git
cd df2023-xai

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Install all dependencies (base + models + XAI)
pip install -e .[dev,models,xai]

# 4. (Optional) Install documentation dependencies
pip install -e .[docs]

# 5. (Optional) Install pre-commit hooks
pre-commit install --install-hooks

# 6. Verify installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import df2023xai; print('Package installed successfully')"
```

---

## Dataset Setup

### About DF2023

**DF2023 (V15)** is a large-scale image forgery localization dataset comprising 1,005,000 image–mask pairs across four manipulation types: `copy_move`, `splicing`, `inpainting`, and `enhancement`.

- **Image format:** JPEG
- **Mask format:** PNG binary (0 = authentic, 255 = forged)
- **Naming convention:** `COCO_DF_{MethodCode}_{SceneID}.jpg`
  - Example: `COCO_DF_C102B00000_00505270.jpg` → `scene_id = 00505270`
- **Total disk size:** Approximately 300–400 GB
- **Dataset availability:** https://zenodo.org/records/7326540

### Expected Raw Directory Layout

```
/path/to/DF2023_train/DF2023_V15_train/
    COCO_V15/         ← training images (JPEG)
    COCO_V15_GT/      ← training masks (PNG, binary 0/255)

/path/to/DF2023_val/DF2023_V15_val/
    COCO_V15/         ← validation images (JPEG)
    COCO_V15_GT/      ← validation masks (PNG, binary 0/255)
```

### Data Preparation Steps

#### Step 1a — Create Symlinks

```bash
dfx-symlink-data run \
  --train-images "/path/to/DF2023_V15_train/COCO_V15" \
  --train-masks  "/path/to/DF2023_V15_train/COCO_V15_GT" \
  --val-images   "/path/to/DF2023_V15_val/COCO_V15" \
  --val-masks    "/path/to/DF2023_V15_val/COCO_V15_GT" \
  --dest data
```

**Outputs:** `data/train/images/`, `data/train/masks/`, `data/val/images/`, `data/val/masks/` (read-only symlinks)

#### Step 1b — Build Master Manifest

```bash
dfx-build-manifest run \
  --images     data/train/images \
  --masks      data/train/masks \
  --images-val data/val/images \
  --masks-val  data/val/masks \
  --out        data/manifests/df2023_v15_manifest.csv
```

**Output:** `data/manifests/df2023_v15_manifest.csv`

#### Steps 1c–1e — Data Preparation (unified)

```bash
# Run all three preparation steps in sequence
python scripts/prepare_data.py all \
  --manifest data/manifests/df2023_v15_manifest.csv \
  --seed 1337

# Or run each step individually:

# Step 1c — Add scene IDs to manifest (in-place)
python scripts/prepare_data.py add-scene-ids \
  --manifest data/manifests/df2023_v15_manifest.csv

# Step 1d — Generate scene-disjoint splits (CANONICAL)
python scripts/prepare_data.py scene-splits \
  --manifest data/manifests/df2023_v15_manifest.csv \
  --seed 1337

# Step 1e — Generate random splits (ablation only)
python scripts/prepare_data.py random-splits \
  --manifest data/manifests/df2023_v15_manifest.csv \
  --seed 42
```

Run `python scripts/prepare_data.py --help` for all available flags.

**Scene-split outputs:**
- `data/manifests/splits/train_split.csv` — 803,965 images, 800,000 unique scenes
- `data/manifests/splits/val_split.csv` — 100,514 images, 100,000 unique scenes
- `data/manifests/splits/test_split.csv` — 100,470 images, 100,000 unique scenes

**Split distribution by manipulation type:**

| Split | Images | Scenes | copy_move | enhancement | inpainting | splicing |
|---|---|---|---|---|---|---|
| Train | 803,965 | 800,000 | 241,047 | 160,955 | 80,487 | 321,476 |
| Val | 100,514 | 100,000 | 30,170 | 19,954 | 10,010 | 40,366 |
| Test | 100,470 | 100,000 | 30,283 | 20,091 | 10,003 | 40,158 |

---

## Experiment Pipeline

### Phase 2: Model Training

Both architectures use a **Hybrid Loss** (Weighted Cross-Entropy + Soft Dice), AdamW optimizer, Cosine Annealing scheduler, and identical augmentations (HFlip, Rot90). AMP is disabled on both — float32 is used exclusively.

> **⚠️ Required environment variable for SegFormer determinism:**
> ```bash
> export CUBLAS_WORKSPACE_CONFIG=:4096:8
> ```
> This must be set before training SegFormer-B2. It is set automatically by `run_paper_stats.sh`.

#### Train SegFormer-B2 (3 seeds)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for SEED in 1337 2027 3141; do
  SEED=$SEED python -m df2023xai.cli.run_train train \
    --config configs/train_segformer_b2_full.yaml
done
```

Or using make (seed 1337 only): `make train.segformer_b2_full`

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5.0 × 10⁻⁶ |
| Weight decay | 0.05 |
| Batch size (per GPU) | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Total steps | 160,000 |
| Warmup steps | 5,000 |
| Gradient clip | 1.0 |
| Precision | float32 (AMP disabled) |
| Loss weights | CE [0.2, 0.8] + Dice (λ = 0.5 each) |
| Image size | 512 × 512 |
| Augmentation | HFlip (p=0.5), Rot90 (p=0.5) |
| Early stop patience | 15 |
| Min LR | 1.0 × 10⁻⁷ |

**Outputs:** `outputs/segformer_b2_v2/seed{SEED}/best.pt`, `last.pt`, `config_train.json`, `stdout.log`  
**Runtime:** ~2–3 days per seed on L40-12Q (12 GB vGPU)

#### Train U-Net ResNet-34 (3 seeds)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for SEED in 1337 2027 3141; do
  SEED=$SEED python -m df2023xai.cli.run_train train \
    --config configs/train_unet_r34_full.yaml
done
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.01 |
| Batch size (per GPU) | 8 |
| Gradient accumulation | 4 (effective batch = 32) |
| Total steps | 100,000 |
| Warmup steps | 1,000 |
| Gradient clip | 1.0 |
| Precision | float32 (AMP disabled) |
| Loss weights | CE [0.2, 0.8] + Dice (λ = 0.5 each) |
| Image size | 512 × 512 |
| Augmentation | HFlip (p=0.5), Rot90 (p=0.5) |
| Early stop patience | 20 |
| Min LR | 1 × 10⁻⁶ |

**Outputs:** `outputs/unet_r34_v2/seed{SEED}/best.pt`, `last.pt`, `stdout.log`  
**Runtime:** ~1–2 days per seed

#### Ablation: Random-Split Training (SegFormer only)

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=1337 python -m df2023xai.cli.run_train train \
  --config configs/train_segformer_b2_random.yaml
```

---

### Phase 3: Forensic Evaluation

```bash
python -m df2023xai.cli.run_forensic_eval \
  --config configs/forensic_eval.yaml
```

Or: `make eval.forensic`

**Inputs:** 6 model checkpoint directories, `test_split.csv`  
**Outputs:**
- `outputs/eval_results/metrics_summary.csv`
- `outputs/eval_results/{model}_sample_{0-4}.png`
- `outputs/{model}/seed{seed}/test_metrics.json` — required by Phase 5

**Runtime:** ~20–30 minutes on GPU  
**Prediction threshold:** 0.5 (sigmoid output)

Ablation (random-split): `python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval_random.yaml`

---

### Phase 4: XAI Generation

```bash
python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml
```

Or: `make xai.run`

**Outputs:** `outputs/xai_audit/{model}_{method}_{idx}.png` (Grad-CAM++, IG, Attention Rollout × 8 images × 2 models)  
**Runtime:** ~10 minutes

---

### Phase 5: Paper Statistics Pipeline

Computes all quantitative values reported in the paper: Table IV metrics, Saliency TV, bootstrap CIs, and directional consistency.

> **Prerequisite:** All 6 `test_metrics.json` files must exist (produced by Phase 3).

#### Recommended: Single Command

```bash
bash scripts/run_paper_stats.sh              # runs all 5 steps
bash scripts/run_paper_stats.sh --skip-step2 # skip TV if arrays already exist
```

#### Step-by-Step

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Step 1 — Verify all 6 checkpoints exist
python scripts/paper_stats.py verify-checkpoints

# Step 2 — Compute per-image Saliency TV (long-running)
python scripts/paper_stats.py compute-tv
python scripts/paper_stats.py compute-tv --chunk-size 5        # for <12 GB VRAM
python scripts/paper_stats.py compute-tv --models segformer_b2_v2 --seeds 2027  # resume one
python scripts/paper_stats.py compute-tv --overwrite           # force recompute

# Step 3 — Bootstrap 95% CIs
python scripts/paper_stats.py bootstrap-ci

# Step 4 — Table IV (mean ± std across seeds)
python scripts/paper_stats.py metric-std

# Step 5 — Directional consistency check
python scripts/paper_stats.py directional-check
```

Run `python scripts/paper_stats.py --help` or `python scripts/paper_stats.py <subcommand> --help` for all flags.

**TV computation protocol** (exact pipeline used in the paper):
1. Generate Grad-CAM++ and IG attribution maps per image
2. Apply ReLU → per-map min-max normalization to [0,1] (ε = 10⁻⁷)
3. Resize to 512×512 via bilinear interpolation
4. No post-hoc smoothing applied
5. Compute isotropic discrete TV: `TV(S) = (1/HW) Σ sqrt((S[i+1,j]-S[i,j])² + (S[i,j+1]-S[i,j])²)`
6. Average over all 100,470 test images

**Key outputs:**

| Step | Output | Runtime |
|---|---|---|
| verify-checkpoints | Console report | < 1s |
| compute-tv | `outputs/tv_arrays/*_tv.npy` (6 files) | ~25h per combination (~150h total) |
| bootstrap-ci | `outputs/tv_arrays/bootstrap_results.json` | ~2–5 min |
| metric-std | Table IV to stdout | < 1s |
| directional-check | `outputs/tv_arrays/directional_check.json` | < 1s |

> **Resume logic:** `compute-tv` saves checkpoints every 500 images as `.tmp.npy` — safe to interrupt and resume at any time.

---

### Phase 6: Supporting Analyses

These analyses can run in parallel while `compute-tv` is processing. They address specific reviewer validation requirements.

#### Recommended: Single Command

```bash
bash scripts/run_additions.sh
```

#### Step-by-Step

```bash
# Per-sample IoU + Wilcoxon signed-rank test
python scripts/run_analyses.py per-sample-iou
python scripts/run_analyses.py per-sample-iou --wilcoxon-only  # if arrays exist

# Parameter randomization sanity check 
python scripts/run_analyses.py sanity-check
python scripts/run_analyses.py sanity-check --chunk-size 5     # for <12 GB VRAM
python scripts/run_analyses.py sanity-check --overwrite

# Fast coherence diagnostic (200 images)
python scripts/run_analyses.py coherence-check

# Frequency augmentation experiment — SRM 4-channel
python scripts/run_analyses.py freq-experiment
python scripts/run_analyses.py freq-experiment --skip-train    # if checkpoint exists
python scripts/run_analyses.py freq-experiment --skip-tv       # IoU only
```

Run `python scripts/run_analyses.py --help` or `python scripts/run_analyses.py <subcommand> --help` for all flags.

**Key outputs and expected results:**

| Analysis | Output | Expected Result | Runtime |
|---|---|---|---|
| per-sample-iou | `outputs/metrics/*_per_sample_iou.npy` (6 files) | Wilcoxon p ≈ 0 (U-Net > SegFormer) | ~2–4h |
| sanity-check | `outputs/tv_arrays/*_RANDOM_tv.npy` | U-Net +263%, SegFormer +650% above trained | ~25h per model |
| coherence-check | `outputs/coherence_v2_results.json` | Directional signal confirmed on 200 images | ~minutes |
| freq-experiment | `outputs/freq_experiment/` | IoU +2.7%, TV −3.66% on Enhancement subset | ~1h |

> **Resume logic:** `sanity-check` and `freq-experiment` save checkpoints as `.tmp.npy` — safe to interrupt and resume.

---

### Phase 7: Manuscript Figures and Tables

```bash
# Generate all manuscript outputs in sequence
python scripts/generate_manuscript.py all

# Or individually:
python scripts/generate_manuscript.py figure3           # Figure 3: 2×6 qualitative panel
python scripts/generate_manuscript.py table-dataset     # Table II: split distribution
python scripts/generate_manuscript.py table-hyperparams # Table III: hyperparameters
```

Additional flags:
```bash
python scripts/generate_manuscript.py figure3 --dpi 600 --out outputs/figure3_hires.png
python scripts/generate_manuscript.py table-dataset --format latex
python scripts/generate_manuscript.py table-hyperparams --format latex
```

Run `python scripts/generate_manuscript.py --help` for all flags.

> **Note:** `figure3` requires trained model checkpoints (Phase 2 outputs). `table-dataset` and `table-hyperparams` are CPU-only and can be run at any time after Phase 1.

---

## Reproducibility

### Fixed Random Seeds

| Purpose | Seed | Script / Flag |
|---|---|---|
| Model training — primary | 1337 | `SEED=1337 dfx-train` |
| Model training — replicate 2 | 2027 | `SEED=2027 dfx-train` |
| Model training — replicate 3 | 3141 | `SEED=3141 dfx-train` |
| Scene-disjoint split | 1337 | `prepare_data.py scene-splits --seed 1337` |
| Random split (ablation) | 42 | `prepare_data.py random-splits --seed 42` |
| Bootstrap CI | 42 | `paper_stats.py bootstrap-ci --seed 42` |
| Random model init (sanity check) | 42 | `run_analyses.py sanity-check` |
| Frequency experiment base model | 2027 | `run_analyses.py freq-experiment` |

### Determinism Checklist

Before running any experiment, verify:

- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported (SegFormer only)
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **NOT** set (L40 vGPU)
- [ ] AMP is disabled (`amp: false` in all configs)
- [ ] float32 used throughout (no mixed precision)
- [ ] Correct seed passed via `SEED=` environment variable for training

> **Environment safety:** Both `paper_stats.py` and `run_analyses.py` automatically warn at startup if `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is detected or if `CUBLAS_WORKSPACE_CONFIG` is missing.

### Normalization Robustness

The Saliency TV ranking (SegFormer-B2 < U-Net) is preserved under three alternative normalization schemes — global dataset-percentile, softmax, and per-map z-score — confirming it is not an artifact of the chosen normalization strategy.

---

## Expected Outputs and Results

After completing all phases, your `outputs/` directory should contain:

```
outputs/
├── segformer_b2_v2/seed{1337,2027,3141}/
│   ├── best.pt                       (~95 MB)
│   ├── config_train.json
│   ├── stdout.log
│   └── test_metrics.json
├── unet_r34_v2/seed{1337,2027,3141}/
│   ├── best.pt                       (~94 MB)
│   └── test_metrics.json
├── segformer_b2_random/seed1337/     (ablation only)
├── tv_arrays/
│   ├── *_tv.npy                      (6 files × 100,470 float32 values)
│   ├── *_RANDOM_tv.npy               (2 files — sanity check)
│   ├── bootstrap_results.json
│   └── directional_check.json
├── metrics/
│   └── *_per_sample_iou.npy          (6 files)
├── freq_experiment/
│   ├── checkpoints/{best,last}.pt
│   └── *.npy
├── eval_results/
│   ├── metrics_summary.csv
│   └── *.png
└── xai_audit/
    └── *.png
```

**Total disk usage (outputs only):** approximately 2–3 GB (excluding checkpoints: ~1.1 GB across 6 seeds)

---

## Known Issues and Constraints

| Issue | Impact | Resolution |
|---|---|---|
| `compute-tv` takes ~150h total | Long wall-clock time for full reproduction | Use `--skip-step2` if arrays already exist; use `--chunk-size 5` for smaller VRAM |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` crashes L40-12Q | Training/inference fails immediately | Do not set this variable on vGPU profiles; both `paper_stats.py` and `run_analyses.py` warn at startup if detected |
| Makefile has hardcoded absolute path | `make` targets fail on other machines | Change `ROOT:=/home/proj-samer/df2023-xai` to `ROOT:=$(shell pwd)` |

---

## Citation

If you use this code or the DF2023-XAI framework in your research, please cite:

```bibtex
@article{aoudi2026beyond,
  title     = {Beyond Accuracy: Auditing Image Forgery Localization via
               Scene-Disjoint Evaluation and {XAI} Faithfulness},
  author    = {Aoudi, Samer},
  journal   = {TBD},
  year      = {2026},
  publisher = {TBD}
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

**Dr. Samer Aoudi**  
Assistant Professor & Division Chair, Computer Information Science  
Higher Colleges of Technology (HCT), UAE  
📧 cybersecurity@sameraoudi.com  
🔗 ORCID: [0000-0003-3887-0119](https://orcid.org/0000-0003-3887-0119)
