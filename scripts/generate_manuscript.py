#!/usr/bin/env python
# ===============================================================================
# PROJECT      : DF2023-XAI (Explainable AI for Deepfake Detection)
# SCRIPT       : generate_manuscript.py
# VERSION      : 1.0.0
# DESCRIPTION  : Unified manuscript figure and table generator for DF2023-XAI.
# -------------------------------------------------------------------------------
# FUNCTIONALITY:
#     Combines three manuscript generation scripts into a single entry point:
#     (1) figure3          — Figure 3: 2×6 qualitative XAI analysis panel
#     (2) table-dataset    — Table II: dataset split distribution statistics
#     (3) table-hyperparams — Table III: training hyperparameters (both models)
#     (4) all              — generates all three in sequence
# USAGE:
#     python scripts/generate_manuscript.py figure3
#     python scripts/generate_manuscript.py figure3 --dpi 600 --out fig3.png
#     python scripts/generate_manuscript.py table-dataset [--format latex]
#     python scripts/generate_manuscript.py table-hyperparams [--format latex]
#     python scripts/generate_manuscript.py all
# INPUTS       :
#     - data/manifests/splits/test_split.csv          (figure3)
#     - outputs/{model}/seed{seed}/best.pt            (figure3)
#     - data/manifests/splits/{train,val,test}_split.csv (table-dataset)
#     - configs/train_segformer_b2_full.yaml          (table-hyperparams)
#     - configs/train_unet_r34_full.yaml              (table-hyperparams)
# OUTPUTS      :
#     - Figure_3_Qualitative_Analysis.png             (figure3, ≥300 DPI)
#     - Table II printed to stdout                    (table-dataset)
#     - Table III printed to stdout                   (table-hyperparams)
# AUTHOR       : Dr. Samer Aoudi
# AFFILIATION  : Higher Colleges of Technology (HCT), UAE
# ROLE         : Assistant Professor & Division Chair (CIS)
# EMAIL        : cybersecurity@sameraoudi.com
# ORCID        : 0000-0003-3887-0119
# CREATED      : 2026-03-20
# UPDATED      : 2026-03-20
# LICENSE      : MIT License
# CITATION     : If used in academic research, please cite:
#                Aoudi, S. et al. (2026). "Beyond Accuracy: Auditing Image
#                Forgery Localization via Scene-Disjoint Evaluation and
#                XAI Faithfulness." IEEE Access.
# DESIGN NOTES:
#     - table-hyperparams reads values directly from YAML config files —
#       no hyperparameter values are hardcoded in this script.
#     - figure3 enforces DPI ≥ 300 (IEEE Access minimum) and warns if
#       a lower value is requested via --dpi.
#     - all subcommands are independently runnable; run-all does not
#       require prior outputs from paper_stats.py or run_analyses.py
#       except figure3 which needs trained model checkpoints.
# DEPENDENCIES:
#     - Python >= 3.10
#     - torch >= 2.2      (figure3 only)
#     - matplotlib >= 3.8
#     - pandas
#     - omegaconf >= 2.3
# ===============================================================================

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: figure3
# ══════════════════════════════════════════════════════════════════════════════


def _generate_figure3(args: argparse.Namespace) -> None:
    """
    Figure 3 — 2×6 qualitative XAI analysis panel.
    Logic copied verbatim from generate_figure3.py; only globals replaced
    with args attributes.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from df2023xai.data.dataset import ForgerySegDataset
    from df2023xai.models.factory import load_model_from_dir

    # ── IEEE Access DPI guard ──────────────────────────────────────────────────
    if args.dpi < 300:
        print("[WARNING] DPI < 300 does not meet IEEE Access minimum requirements.")

    # ── Reconstruct module-level constants from args ───────────────────────────
    OUTPUT_DIR = args.outputs_dir
    IDS_TO_PLOT = args.sample_ids
    seed = args.seeds[0]  # figure3 uses a single checkpoint seed

    MODELS = {
        "segformer": {
            "name": "segformer_b2_v2",
            "path": f"outputs/segformer_b2_v2/seed{seed}",
            "display": "SegFormer-B2",
        },
        "unet": {
            "name": "unet_r34_v2",
            "path": f"outputs/unet_r34_v2/seed{seed}",
            "display": "U-Net-R34",
        },
    }

    # ── Nested helpers (verbatim from original, using closure over locals) ────

    def ensure_reference_files(idx, dataset):
        """Saves raw image and GT if missing."""
        img_path = f"{OUTPUT_DIR}/sample_{idx}_img.jpg"
        gt_path = f"{OUTPUT_DIR}/sample_{idx}_gt.png"

        # Always fetch data to return it, even if file exists
        image, mask = dataset[idx]

        # Denormalize Image for saving (assuming standard ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = image * std + mean
        img_np = img_tensor.permute(1, 2, 0).numpy().clip(0, 1)

        # Save Image
        if not os.path.exists(img_path):
            plt.imsave(img_path, img_np)
            print(f"[+] Recovered Image: {img_path}")

        # Save GT
        if not os.path.exists(gt_path):
            plt.imsave(gt_path, mask.squeeze(), cmap="gray")
            print(f"[+] Recovered GT: {gt_path}")

        return image.unsqueeze(0), img_path, gt_path

    def generate_prediction(idx, model, model_key, batch_img):
        """Runs inference and saves prediction if missing."""
        pred_path = f"{OUTPUT_DIR}/{MODELS[model_key]['name']}_pred_{idx}.png"

        if not os.path.exists(pred_path):
            with torch.no_grad():
                logits = model(batch_img.cuda())
                pred_mask = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

            plt.imsave(pred_path, pred_mask, cmap="gray")
            print(f"[+] Generated Prediction: {pred_path}")

        return pred_path

    # ── Main figure3 logic (verbatim from original main()) ────────────────────

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Dataset
    print("[*] Loading Dataset to recover images...")
    ds = ForgerySegDataset(manifest_csv=args.manifest, img_size=512, split="test")

    # 2. Load Models
    print("[*] Loading Models for inference...")
    loaded_models = {}
    for key, cfg in MODELS.items():
        try:
            loaded_models[key] = load_model_from_dir(cfg["path"]).eval().cuda()
        except Exception as e:
            print(f"[!] Warning: Could not load {key}. Error: {e}")

    # 3. Recover Files
    file_map: dict[int, dict[str, str | None]] = {idx: {} for idx in IDS_TO_PLOT}

    for idx in IDS_TO_PLOT:
        # Get/Save Input & GT
        batch_img, f_img, f_gt = ensure_reference_files(idx, ds)
        file_map[idx]["img"] = f_img
        file_map[idx]["gt"] = f_gt

        # Get/Save Predictions
        for key, model in loaded_models.items():
            f_pred = generate_prediction(idx, model, key, batch_img)
            file_map[idx][f"{key}_pred"] = f_pred

            # Map existing XAI files
            # Expected format: segformer_b2_v2_gradcampp_0.png
            xai_path = f"{OUTPUT_DIR}/{MODELS[key]['name']}_gradcampp_{idx}.png"
            if os.path.exists(xai_path):
                file_map[idx][f"{key}_xai"] = xai_path
            else:
                print(f"[!] Warning: XAI file missing: {xai_path}")
                file_map[idx][f"{key}_xai"] = None

    # 4. Plot Figure 3
    print("[*] Compositing Figure 3...")
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), dpi=args.dpi)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    cols = [
        "Input Image",
        "Ground Truth",
        "U-Net\nPrediction",
        "U-Net\nXAI",
        "SegFormer\nPrediction",
        "SegFormer\nXAI",
    ]

    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col, fontsize=11, fontweight="bold", pad=10)

    # Row Labels
    axes[0, 0].set_ylabel(
        f"Splicing Case\n(Sample {IDS_TO_PLOT[0]})", fontsize=11, fontweight="bold"
    )
    axes[1, 0].set_ylabel(
        f"Clever Hans Case\n(Sample {IDS_TO_PLOT[1]})", fontsize=11, fontweight="bold"
    )

    # Helper to load safely
    def load_safe(path):
        if path and os.path.exists(path):
            return mpimg.imread(path)
        return np.ones((512, 512, 3))  # White placeholder

    for r_idx, sample_id in enumerate(IDS_TO_PLOT):
        files = file_map[sample_id]

        # Order: Img, GT, Unet_Pred, Unet_XAI, Seg_Pred, Seg_XAI
        images = [
            files["img"],
            files["gt"],
            files.get("unet_pred"),
            files.get("unet_xai"),
            files.get("segformer_pred"),
            files.get("segformer_xai"),
        ]

        for c_idx, f_path in enumerate(images):
            axes[r_idx, c_idx].imshow(load_safe(f_path))
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

            # Colored Borders
            if c_idx in [2, 3]:  # U-Net
                for s in axes[r_idx, c_idx].spines.values():
                    s.set_edgecolor("orange")
                    s.set_linewidth(2)
            if c_idx in [4, 5]:  # SegFormer
                for s in axes[r_idx, c_idx].spines.values():
                    s.set_edgecolor("green")
                    s.set_linewidth(2)

    out_file = args.out
    plt.savefig(out_file, dpi=args.dpi, bbox_inches="tight")

    width_px = int(20 * args.dpi)
    height_px = int(7 * args.dpi)
    print(f"Figure 3 saved → {out_file} ({height_px}x{width_px} px, DPI: {args.dpi})")


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: table-dataset
# ══════════════════════════════════════════════════════════════════════════════


def _extract_scene_id(row: "pd.Series") -> str:
    """
    Parses scene ID from image_id string.
    Example: COCO_DF_I000B00000_00918198 -> 00918198
    Verbatim from generate_table_dataset_distribution.py.
    """
    # If the column exists, use it
    if "source_scene_id" in row:
        return row["source_scene_id"]
    if "scene_id" in row:
        return row["scene_id"]

    # Fallback: Split string by underscore and take the last part
    if "image_id" in row:
        return str(row["image_id"]).split("_")[-1]
    return "unknown"


def _generate_table_dataset(args: argparse.Namespace) -> None:
    """
    Table II — dataset split distribution statistics.
    Logic copied verbatim from generate_table_dataset_distribution.py;
    module-level paths replaced with args attributes.
    Adds --format and --out support beyond the original stdout-only output.
    """
    base_path = Path(args.splits_dir)
    files = {
        "Train": base_path / "train_split.csv",
        "Val": base_path / "val_split.csv",
        "Test": base_path / "test_split.csv",
    }

    total_imgs = 0
    stats: dict[str, dict] = {}

    # Collect Stats (verbatim from original)
    for split, p in files.items():
        if p.exists():
            try:
                df = pd.read_csv(p)

                # Create temporary scene column for counting
                df["scene_extracted"] = df.apply(_extract_scene_id, axis=1)

                n_imgs = len(df)
                n_scenes = df["scene_extracted"].nunique()

                stats[split] = {"imgs": n_imgs, "scenes": n_scenes}
                total_imgs += n_imgs

            except Exception as e:
                print(f"[ERROR] Reading {split}: {e}")
        else:
            print(f"[WARN] File not found: {p}")

    # Format output
    fmt = args.format
    if fmt == "text":
        lines = [
            f"{'Partition':<10} | {'# Images':<10} | {'# Scenes':<10} | {'% Images':<10}",
            "-" * 46,
        ]
        for split, data in stats.items():
            n_imgs = data["imgs"]
            n_scenes = data["scenes"]
            pct = (n_imgs / total_imgs) * 100 if total_imgs > 0 else 0
            lines.append(f"{split:<10} | {n_imgs:<10} | {n_scenes:<10} | {pct:<10.1f}%")
        lines.append("-" * 46)
        lines.append(f"{'Total':<10} | {total_imgs:<10} | {'-':<10} | 100.0%")
        table = "\n".join(lines)

    elif fmt == "csv":
        rows = ["Partition,# Images,# Scenes,% Images"]
        for split, data in stats.items():
            n_imgs = data["imgs"]
            n_scenes = data["scenes"]
            pct = (n_imgs / total_imgs) * 100 if total_imgs > 0 else 0
            rows.append(f"{split},{n_imgs},{n_scenes},{pct:.1f}")
        rows.append(f"Total,{total_imgs},-,100.0")
        table = "\n".join(rows)

    elif fmt == "latex":
        rows = [
            r"\begin{tabular}{lrrr}",
            r"\hline",
            r"Partition & \# Images & \# Scenes & \% Images \\",
            r"\hline",
        ]
        for split, data in stats.items():
            n_imgs = data["imgs"]
            n_scenes = data["scenes"]
            pct = (n_imgs / total_imgs) * 100 if total_imgs > 0 else 0
            rows.append(f"{split} & {n_imgs} & {n_scenes} & {pct:.1f}\\% \\\\")
        rows.append(r"\hline")
        rows.append(f"Total & {total_imgs} & -- & 100.0\\% \\\\")
        rows.append(r"\hline")
        rows.append(r"\end{tabular}")
        table = "\n".join(rows)

    else:
        print(f"[ERROR] Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"Table II written → {args.out}")
    else:
        print(table)


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: table-hyperparams
# ══════════════════════════════════════════════════════════════════════════════


def _generate_table_hyperparams(args: argparse.Namespace) -> None:
    """
    Table III — training hyperparameters for both models.
    Logic copied verbatim from generate_table_hyperparams.py for the
    SegFormer column; extends to a two-column comparison when --unet-config
    is provided (default). Values read directly from YAML — never hardcoded.
    """
    # ── Load SegFormer config (verbatim from original) ─────────────────────
    seg_cfg = None
    try:
        seg_cfg = OmegaConf.load(args.segformer_config)
        st = seg_cfg.train
        sd = seg_cfg.data
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # ── Load U-Net config (extended behaviour when --unet-config provided) ──
    unet_cfg = None
    if args.unet_config:
        try:
            unet_cfg = OmegaConf.load(args.unet_config)
        except Exception as e:
            print(f"Error loading U-Net config: {e}")
            sys.exit(1)

    # ── Build parameter rows ────────────────────────────────────────────────
    # "Optimizer" is hardcoded as AdamW (standard in loop.py — same as original)
    seg_vals = {
        "Optimizer": "AdamW",
        "Learning Rate": str(st.lr),
        "Weight Decay": str(st.weight_decay),
        "Batch Size": str(sd.batch_size),
        "Warmup Steps": str(st.warmup_steps),
        "Gradient Clip": str(st.grad_clip),
        "Precision (AMP)": str(st.amp),
        "Image Size": f"{sd.img_size}x{sd.img_size}",
    }

    unet_vals = {}
    if unet_cfg is not None:
        ut = unet_cfg.train
        ud = unet_cfg.data
        unet_vals = {
            "Optimizer": "AdamW",
            "Learning Rate": str(ut.lr),
            "Weight Decay": str(ut.weight_decay),
            "Batch Size": str(ud.batch_size),
            "Warmup Steps": str(ut.warmup_steps),
            "Gradient Clip": str(ut.grad_clip),
            "Precision (AMP)": str(ut.amp),
            "Image Size": f"{ud.img_size}x{ud.img_size}",
        }

    params = list(seg_vals.keys())
    fmt = args.format
    two_col = unet_cfg is not None

    # ── Format output ────────────────────────────────────────────────────────
    if fmt == "text":
        if two_col:
            header = f"{'Parameter':<25} | {'SegFormer-B2':<16} | {'U-Net-R34':<15}"
            sep = "-" * 62
            lines = [header, sep]
            for p in params:
                sv = seg_vals[p]
                uv = unet_vals.get(p, "")
                lines.append(f"{p:<25} | {sv:<16} | {uv:<15}")
            table = "\n".join(lines)
        else:
            # Single-column: verbatim original format
            lines = [
                f"{'Parameter':<25} | {'Value':<15}",
                "-" * 43,
            ]
            for p in params:
                sv = seg_vals[p]
                lines.append(f"{p:<25} | {sv:<15}")
            table = "\n".join(lines)

    elif fmt == "csv":
        if two_col:
            rows = ["Parameter,SegFormer-B2,U-Net-R34"]
            for p in params:
                rows.append(f"{p},{seg_vals[p]},{unet_vals.get(p, '')}")
        else:
            rows = ["Parameter,Value"]
            for p in params:
                rows.append(f"{p},{seg_vals[p]}")
        table = "\n".join(rows)

    elif fmt == "latex":
        if two_col:
            rows = [
                r"\begin{tabular}{lll}",
                r"\hline",
                r"Parameter & SegFormer-B2 & U-Net-R34 \\",
                r"\hline",
            ]
            for p in params:
                sv = seg_vals[p].replace("_", r"\_")
                uv = unet_vals.get(p, "").replace("_", r"\_")
                rows.append(f"{p} & {sv} & {uv} \\\\")
        else:
            rows = [
                r"\begin{tabular}{ll}",
                r"\hline",
                r"Parameter & Value \\",
                r"\hline",
            ]
            for p in params:
                sv = seg_vals[p].replace("_", r"\_")
                rows.append(f"{p} & {sv} \\\\")
        rows.append(r"\hline")
        rows.append(r"\end{tabular}")
        table = "\n".join(rows)

    else:
        print(f"[ERROR] Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"Table III written → {args.out}")
    else:
        print(table)


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: all
# ══════════════════════════════════════════════════════════════════════════════


def _cmd_all(args: argparse.Namespace) -> None:
    """Run all three generators in sequence."""

    print("[1/3] Generating Figure 3...")
    try:
        _generate_figure3(args)
    except Exception as e:
        print(f"[ERROR] figure3 failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("")
    print("[2/3] Generating Table II (dataset distribution)...")
    table_ds_args = argparse.Namespace(
        splits_dir=args.splits_dir,
        out=None,
        format=args.format,
    )
    try:
        _generate_table_dataset(table_ds_args)
    except Exception as e:
        print(f"[ERROR] table-dataset failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("")
    print("[3/3] Generating Table III (hyperparameters)...")
    table_hp_args = argparse.Namespace(
        segformer_config=args.segformer_config,
        unet_config=args.unet_config,
        out=None,
        format=args.format,
    )
    try:
        _generate_table_hyperparams(table_hp_args)
    except Exception as e:
        print(f"[ERROR] table-hyperparams failed: {e}", file=sys.stderr)
        sys.exit(1)

    table_ii_dest = "stdout"
    table_iii_dest = "stdout"

    print("")
    print("═" * 39)
    print(" MANUSCRIPT OUTPUTS — FINAL SUMMARY")
    print("═" * 39)
    print(f"Figure 3   → {args.out}")
    print(f"Table II   → {table_ii_dest}")
    print(f"Table III  → {table_iii_dest}")
    print("═" * 39)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _add_figure3_args(p: argparse.ArgumentParser) -> None:
    """Add figure3 arguments to parser p."""
    p.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/splits/test_split.csv",
        help="Path to test split CSV (default: data/manifests/splits/test_split.csv)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["segformer", "unet"],
        choices=["segformer", "unet"],
        help="Models to load for inference (default: segformer unet)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1337],
        help="Checkpoint seed(s); first value used for model path (default: 1337)",
    )
    p.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs/xai_audit",
        help="Directory for cached prediction/GT PNGs (default: outputs/xai_audit)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="Figure_3_Qualitative_Analysis.png",
        help="Output PNG path (default: Figure_3_Qualitative_Analysis.png)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI — IEEE Access minimum is 300 (default: 300)",
    )
    p.add_argument(
        "--sample-ids",
        nargs="+",
        type=int,
        default=[0, 3],
        dest="sample_ids",
        help="Test-set indices to plot as figure rows (default: 0 3)",
    )


def _add_table_dataset_args(p: argparse.ArgumentParser) -> None:
    """Add table-dataset arguments to parser p."""
    p.add_argument(
        "--splits-dir",
        type=str,
        default="data/manifests/splits",
        dest="splits_dir",
        help="Directory containing {train,val,test}_split.csv " "(default: data/manifests/splits)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write table to this file instead of stdout (default: stdout)",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["text", "csv", "latex"],
        default="text",
        help="Output format (default: text)",
    )


def _add_table_hyperparams_args(p: argparse.ArgumentParser) -> None:
    """Add table-hyperparams arguments to parser p."""
    p.add_argument(
        "--segformer-config",
        type=str,
        default="configs/train_segformer_b2_full.yaml",
        dest="segformer_config",
        help="SegFormer training config YAML " "(default: configs/train_segformer_b2_full.yaml)",
    )
    p.add_argument(
        "--unet-config",
        type=str,
        default="configs/train_unet_r34_full.yaml",
        dest="unet_config",
        help="U-Net training config YAML " "(default: configs/train_unet_r34_full.yaml)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write table to this file instead of stdout (default: stdout)",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["text", "csv", "latex"],
        default="text",
        help="Output format (default: text)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_manuscript.py",
        description="Unified manuscript figure and table generator for DF2023-XAI",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # ── figure3 ───────────────────────────────────────────────────────────────
    sp1 = sub.add_parser(
        "figure3",
        help="Figure 3: 2×6 qualitative XAI analysis panel (requires GPU + checkpoints)",
    )
    _add_figure3_args(sp1)

    # ── table-dataset ─────────────────────────────────────────────────────────
    sp2 = sub.add_parser(
        "table-dataset",
        help="Table II: dataset split distribution (images and scenes per partition)",
    )
    _add_table_dataset_args(sp2)

    # ── table-hyperparams ─────────────────────────────────────────────────────
    sp3 = sub.add_parser(
        "table-hyperparams",
        help="Table III: training hyperparameters read from YAML configs",
    )
    _add_table_hyperparams_args(sp3)

    # ── all ───────────────────────────────────────────────────────────────────
    sp4 = sub.add_parser(
        "all",
        help="Generate all three outputs in sequence: figure3 → table-dataset → table-hyperparams",
    )
    # Union of all flags from all three subcommands
    _add_figure3_args(sp4)
    # table-dataset flags (--splits-dir; skip --out since stdout in all)
    sp4.add_argument(
        "--splits-dir",
        type=str,
        default="data/manifests/splits",
        dest="splits_dir",
        help="Directory containing {train,val,test}_split.csv " "(default: data/manifests/splits)",
    )
    # table-hyperparams flags (skip --out since stdout in all)
    sp4.add_argument(
        "--segformer-config",
        type=str,
        default="configs/train_segformer_b2_full.yaml",
        dest="segformer_config",
        help="SegFormer training config YAML " "(default: configs/train_segformer_b2_full.yaml)",
    )
    sp4.add_argument(
        "--unet-config",
        type=str,
        default="configs/train_unet_r34_full.yaml",
        dest="unet_config",
        help="U-Net training config YAML " "(default: configs/train_unet_r34_full.yaml)",
    )
    # Shared table --format (applies to both tables)
    sp4.add_argument(
        "--format",
        type=str,
        choices=["text", "csv", "latex"],
        default="text",
        help="Output format for both tables (default: text)",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "figure3": _generate_figure3,
        "table-dataset": _generate_table_dataset,
        "table-hyperparams": _generate_table_hyperparams,
        "all": _cmd_all,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
