#!/usr/bin/env python
# ===============================================================================
# PROJECT      : DF2023-XAI (Explainable AI for Deepfake Detection)
# SCRIPT       : run_analyses.py
# VERSION      : 1.0.0
# DESCRIPTION  : Unified runner for reviewer-response analysis scripts.
# -------------------------------------------------------------------------------
# FUNCTIONALITY:
#     Combines four analysis scripts into a single entry point with subcommands:
#       per-sample-iou   — R2.C4: per-image IoU arrays + Wilcoxon signed-rank test
#       sanity-check     — R1.C2: parameter randomization sanity check (~150h GPU)
#       coherence-check  — Early directional TV check (200-image subset, seed2027)
#       freq-experiment  — R1.C3: SRM frequency-augmented SegFormer experiment
#       run-all          — Run all four in dependency order (as subprocesses)
#     Shared saliency pipeline helpers (compute_isotropic_tv, relu_minmax_norm,
#     resize_512) are extracted once. FP32/AMP distinction between coherence-check
#     (FP32, paper spec) and sanity-check (AMP) is preserved exactly.
# USAGE:
#     python scripts/run_analyses.py per-sample-iou [--seeds ...] [--overwrite]
#     python scripts/run_analyses.py sanity-check [--chunk-size 5] [--overwrite]
#     python scripts/run_analyses.py coherence-check [--chunk-size 5]
#     python scripts/run_analyses.py freq-experiment [--skip-train] [--skip-tv]
#     python scripts/run_analyses.py run-all [--chunk-size 5] [--overwrite]
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
#     - os.environ.pop("PYTORCH_CUDA_ALLOC_CONF") at import time (before torch)
#       prevents RuntimeError on L40-12Q vGPU (cuMemCreate not supported).
#     - coherence-check uses FP32 IG (no AMP) to match paper spec; sanity-check
#       uses AMP — intentionally different, preserved verbatim.
#     - freq-experiment evaluate_tv hardcodes chunk_size=5 for OOM safety.
#     - run-all executes each step as a subprocess for clean CUDA context isolation.
#     - freq-experiment depends on paper_stats.py compute-tv TV arrays (run
#       separately); run-all will error in freq-experiment if those are absent.
# DEPENDENCIES:
#     - Python >= 3.10
#     - torch >= 2.2, numpy >= 1.24, scipy >= 1.11, tqdm >= 4.66
#     - df2023xai (local package: data, models, train.losses, xai.gradcampp)
# ===============================================================================

import os

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # Must precede torch imports (L40-12Q vGPU)

import argparse  # noqa: E402
import csv  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from df2023xai.data.dataset import ForgerySegDataset  # noqa: E402
from df2023xai.models.factory import build_model, load_model_from_dir  # noqa: E402
from df2023xai.train.losses import HybridForensicLoss  # noqa: E402
from df2023xai.xai.gradcampp import _find_last_conv, gradcampp_or_fallback  # noqa: E402

# ── Module-level constants ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
TV_RES = 512
EPSILON = 1e-7
SAVE_EVERY = 500

# ── Shared saliency pipeline helpers ─────────────────────────────────────────


def compute_isotropic_tv(S: np.ndarray) -> float:
    """Isotropic TV with zero-padding on last row/col — exact step2 formula."""
    S_t = torch.as_tensor(S, dtype=torch.float32)
    dy = torch.zeros_like(S_t)
    dx = torch.zeros_like(S_t)
    dy[:-1, :] = S_t[1:, :] - S_t[:-1, :]
    dy[-1, :] = -S_t[-1, :]
    dx[:, :-1] = S_t[:, 1:] - S_t[:, :-1]
    dx[:, -1] = -S_t[:, -1]
    return torch.sqrt(dy**2 + dx**2).mean().item()


def relu_minmax_norm(t: torch.Tensor) -> torch.Tensor:
    t = torch.relu(t)
    return (t - t.min()) / (t.max() - t.min() + EPSILON)


def resize_512(t: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=(TV_RES, TV_RES),
        mode="bilinear",
        align_corners=False,
    ).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: per-sample-iou
# ══════════════════════════════════════════════════════════════════════════════

_PSIOU_SEEDS = [1337, 2027, 3141]
_PSIOU_MODELS = {
    "unet_r34_v2": "outputs/unet_r34_v2",
    "segformer_b2_v2": "outputs/segformer_b2_v2",
}
_PSIOU_MODEL_SHORT = {
    "unet_r34_v2": "unet_r34",
    "segformer_b2_v2": "segformer_b2",
}
_PSIOU_MODEL_ARCH = {
    "unet_r34_v2": ("unet_r34", 1),
    "segformer_b2_v2": ("segformer_b2", 1),
}
_PSIOU_TEST_CSV = Path("data/manifests/splits/test_split.csv")
_PSIOU_METRICS_DIR = Path("outputs/metrics")
_PSIOU_LOG_FILE = Path("logs/post_step2.log")


def _psiou_robust_load_model(model_name: str, model_dir: Path) -> torch.nn.Module:
    cfg_path = model_dir / "config_train.json"
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "last.pt"

    if cfg_path.exists() and ckpt_path.exists():
        return load_model_from_dir(str(model_dir))

    print(f"  [WARN] config_train.json absent — hard-coded arch for '{model_name}'")
    arch_name, num_classes = _PSIOU_MODEL_ARCH[model_name]
    model = build_model(arch_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"  [OK] Loaded {ckpt_path.name}")
    return model


def _psiou_binary_iou(pred: np.ndarray, mask: np.ndarray) -> float:
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return float(inter / (union + EPSILON))


def _psiou_predict_binary(
    model: torch.nn.Module, img_gpu: torch.Tensor, num_classes: int
) -> np.ndarray:
    with torch.no_grad():
        logits = model(img_gpu.unsqueeze(0))
    if num_classes == 1:
        prob = torch.sigmoid(logits[0, 0])
        return (prob > 0.5).cpu().numpy()
    else:
        return (logits[0].argmax(0) == 1).cpu().numpy()


def _psiou_process_seed(model_name: str, seed: int, overwrite: bool) -> np.ndarray:
    short = _PSIOU_MODEL_SHORT[model_name]
    out_path = _PSIOU_METRICS_DIR / f"{short}_seed{seed}_per_sample_iou.npy"
    tmp_path = _PSIOU_METRICS_DIR / f"{short}_seed{seed}_per_sample_iou.tmp.npy"

    if out_path.exists() and not overwrite:
        arr = np.load(out_path)
        print(
            f"[SKIP] {out_path.name}  n={len(arr):,}  "
            f"mean={arr.mean():.6f}  — use --overwrite to redo"
        )
        return arr

    model_dir = Path(_PSIOU_MODELS[model_name]) / f"seed{seed}"
    print(f"\n[{model_name}  seed={seed}]  Loading model …")
    model = _psiou_robust_load_model(model_name, model_dir)
    model.eval().to(DEVICE)

    full_dataset = ForgerySegDataset(
        manifest_csv=str(_PSIOU_TEST_CSV), img_size=IMG_SIZE, split="test"
    )
    N = len(full_dataset)
    ious = np.full(N, np.nan, dtype=np.float32)
    _, num_classes = _PSIOU_MODEL_ARCH[model_name]

    start_idx = 0
    if tmp_path.exists() and not overwrite:
        tmp = np.load(tmp_path)
        if len(tmp) == N:
            done = int(np.sum(~np.isnan(tmp)))
            if done > 0:
                ious = tmp
                start_idx = done
                print(f"  [RESUME] {done:,}/{N:,} done — skipping to {start_idx}")

    remaining = list(range(start_idx, N))
    if not remaining:
        print("  Already complete — nothing to do.")
        return ious

    loader = DataLoader(
        Subset(full_dataset, remaining),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    print(
        f"  {len(remaining):,} images remaining  |  device={DEVICE}  "
        f"threshold=0.5  (inference only, no gradients)"
    )

    pbar = tqdm(total=len(remaining), unit="img", desc=f"  {short}/seed{seed}")

    for batch_idx, (img, mask) in enumerate(loader):
        global_idx = start_idx + batch_idx
        img_gpu = img.squeeze(0).to(DEVICE)
        mask_np = mask.squeeze(0).numpy().astype(bool)

        pred_np = _psiou_predict_binary(model, img_gpu, num_classes)
        ious[global_idx] = _psiou_binary_iou(pred_np, mask_np)

        pbar.update(1)

        if (batch_idx + 1) % SAVE_EVERY == 0:
            np.save(tmp_path, ious)

    pbar.close()
    np.save(out_path, ious)
    if tmp_path.exists():
        tmp_path.unlink()
    print(
        f"  Saved → {out_path}  " f"n={len(ious):,}  mean={ious.mean():.6f}  std={ious.std():.6f}"
    )
    return ious


def _psiou_run_wilcoxon(seeds: list[int]) -> None:
    from scipy.stats import wilcoxon

    unet_arrays, seg_arrays = [], []
    for seed in seeds:
        p_u = _PSIOU_METRICS_DIR / f"unet_r34_seed{seed}_per_sample_iou.npy"
        p_s = _PSIOU_METRICS_DIR / f"segformer_b2_seed{seed}_per_sample_iou.npy"
        if not p_u.exists() or not p_s.exists():
            print(f"  [SKIP Wilcoxon] Missing arrays for seed {seed}")
            return
        unet_arrays.append(np.load(p_u).astype(np.float64))
        seg_arrays.append(np.load(p_s).astype(np.float64))

    unet_mean_iou = np.mean(np.stack(unet_arrays, axis=0), axis=0)
    seg_mean_iou = np.mean(np.stack(seg_arrays, axis=0), axis=0)

    stat, pval = wilcoxon(seg_mean_iou, unet_mean_iou, alternative="greater")
    significant = pval < 0.05
    verdict = "SIGNIFICANT at p<0.05 ✓" if significant else "NOT significant at p<0.05 ✗"

    lines = [
        "",
        "=== WILCOXON TEST (R2.C4) ===",
        f"U-Net mean per-sample IoU:     {unet_mean_iou.mean():.6f}",
        f"SegFormer mean per-sample IoU: {seg_mean_iou.mean():.6f}",
        f"Wilcoxon statistic:            {stat:.2f}",
        f"Wilcoxon p-value (one-sided,   H1: SegFormer > U-Net): {pval:.6e}",
        f"Result: {verdict}",
    ]
    report = "\n".join(lines)
    print(report)

    _PSIOU_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PSIOU_LOG_FILE, "a") as fh:
        fh.write(report + "\n")
    print(f"\n  Report appended → {_PSIOU_LOG_FILE}")


def cmd_per_sample_iou(args: argparse.Namespace) -> None:
    _PSIOU_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE == "cuda" and not args.wilcoxon_only:
        torch.backends.cudnn.benchmark = True
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        print("  [CUDA] benchmark=True  (expandable_segments disabled for vGPU compat)")

    if not args.wilcoxon_only:
        for model_name in args.models:
            if model_name not in _PSIOU_MODELS:
                print(f"[WARN] Unknown model '{model_name}', skipping.")
                continue
            for seed in args.seeds:
                _psiou_process_seed(model_name, seed, overwrite=args.overwrite)

        print("\n" + "=" * 60)
        print("STEP 2b COMPLETE — Per-Sample IoU")
        for model_name in args.models:
            short = _PSIOU_MODEL_SHORT[model_name]
            for seed in args.seeds:
                p = _PSIOU_METRICS_DIR / f"{short}_seed{seed}_per_sample_iou.npy"
                if p.exists():
                    a = np.load(p)
                    print(
                        f"  {short:15s}  seed={seed}  "
                        f"n={len(a):,}  mean={a.mean():.6f}  std={a.std():.6f}"
                    )
        print("=" * 60)

    all_present = all(
        (_PSIOU_METRICS_DIR / f"{_PSIOU_MODEL_SHORT[m]}_seed{s}_per_sample_iou.npy").exists()
        for m in _PSIOU_MODELS
        for s in args.seeds
    )
    if all_present:
        _psiou_run_wilcoxon(args.seeds)
    else:
        missing = [
            f"{_PSIOU_MODEL_SHORT[m]}_seed{s}_per_sample_iou.npy"
            for m in _PSIOU_MODELS
            for s in args.seeds
            if not (
                _PSIOU_METRICS_DIR / f"{_PSIOU_MODEL_SHORT[m]}_seed{s}_per_sample_iou.npy"
            ).exists()
        ]
        print(f"\n[NOTE] Wilcoxon test deferred — {len(missing)} array(s) not yet present:")
        for f in missing:
            print(f"       outputs/metrics/{f}")
        print("       Re-run this script once all inference passes complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: sanity-check
# ══════════════════════════════════════════════════════════════════════════════

_SANITY_RANDOM_MODELS = {
    "unet_r34": ("unet_r34", 1, "outputs/tv_arrays/unet_r34_RANDOM_tv.npy"),
    "segformer_b2": ("segformer_b2", 1, "outputs/tv_arrays/segformer_b2_RANDOM_tv.npy"),
}
_SANITY_TRAINED_TV_MEANS = {
    "unet_r34": 0.0222,  # mean(0.02241, 0.02254, 0.02169)
    "segformer_b2": 0.0205,  # mean(0.02095, 0.02046, 0.02013)
}
_SANITY_TEST_CSV = Path("data/manifests/splits/test_split.csv")
_SANITY_TV_OUT_DIR = Path("outputs/tv_arrays")
_SANITY_LOG_FILE = Path("logs/post_step2.log")
_SANITY_IG_STEPS = 50
_SANITY_IG_CHUNK_SIZE = 25
_SANITY_TORCH_SEED = 42
_SANITY_SENSITIVITY_THRESHOLD = 0.10


def _sanity_ig_chunked(
    model: torch.nn.Module,
    image: torch.Tensor,
    steps: int = _SANITY_IG_STEPS,
    chunk_size: int = _SANITY_IG_CHUNK_SIZE,
) -> torch.Tensor:
    """Riemann-sum IG with AMP, identical to step2. Returns (H, W) CPU tensor in [0, 1]."""
    x = image.unsqueeze(0)
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)

    grad_accum = torch.zeros_like(x)

    for start in range(0, steps, chunk_size):
        chunk_a = alphas[start : start + chunk_size]
        c = len(chunk_a)

        scaled = baseline + chunk_a.view(c, 1, 1, 1) * (x - baseline)
        scaled = scaled.detach().requires_grad_(True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(scaled)

        if logits.ndim == 4:
            score = (
                (logits[:, 1:2] if logits.shape[1] > 1 else logits[:, 0:1])
                .mean(dim=(1, 2, 3))
                .sum()
            )
        else:
            score = (logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]).sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        grad_accum += scaled.grad.detach().sum(dim=0, keepdim=True)
        del scaled, logits, score

    avg_grads = grad_accum / steps
    attr = ((x - baseline) * avg_grads).abs().sum(dim=1).squeeze(0)
    attr_cpu = attr.detach().cpu()
    return attr_cpu / (attr_cpu.max() + EPSILON)


def _sanity_process_random_model(
    model_key: str,
    arch_name: str,
    num_classes: int,
    out_path: Path,
    overwrite: bool,
    chunk_size: int,
) -> np.ndarray:
    tmp_path = out_path.with_suffix(".tmp.npy")

    if out_path.exists() and not overwrite:
        arr = np.load(out_path)
        print(
            f"[SKIP] {out_path}  n={len(arr):,}  mean={arr.mean():.6f}"
            f"  — use --overwrite to redo"
        )
        return arr

    print(
        f"\n[{model_key}  RANDOM]  Building untrained model "
        f"(arch={arch_name}, torch_seed={_SANITY_TORCH_SEED}) …"
    )
    torch.manual_seed(_SANITY_TORCH_SEED)
    model = build_model(arch_name, num_classes=num_classes, pretrained=False)
    model.eval().to(DEVICE)
    print(f"  [OK] Random weights set (seed={_SANITY_TORCH_SEED})")

    full_dataset = ForgerySegDataset(
        manifest_csv=str(_SANITY_TEST_CSV), img_size=IMG_SIZE, split="test"
    )
    N = len(full_dataset)
    tvs = np.full(N, np.nan, dtype=np.float32)

    start_idx = 0
    if tmp_path.exists() and not overwrite:
        tmp = np.load(tmp_path)
        if len(tmp) == N:
            done = int(np.sum(~np.isnan(tmp)))
            if done > 0:
                tvs = tmp
                start_idx = done
                print(f"  [RESUME] {done:,}/{N:,} done — skipping to {start_idx}")

    remaining = list(range(start_idx, N))
    if not remaining:
        print("  Already complete — nothing to do.")
        return tvs

    loader = DataLoader(
        Subset(full_dataset, remaining),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    n_chunks = -(-_SANITY_IG_STEPS // chunk_size)
    print(
        f"  {len(remaining):,} images  |  device={DEVICE}  "
        f"IG_STEPS={_SANITY_IG_STEPS}  chunk={chunk_size}×{n_chunks}"
    )

    pbar = tqdm(total=len(remaining), unit="img", desc=f"  {model_key}/RANDOM")

    for batch_idx, (img, _) in enumerate(loader):
        global_idx = start_idx + batch_idx
        img_gpu = img.squeeze(0).to(DEVICE)

        gcam_raw = gradcampp_or_fallback(model, img_gpu)
        gcam_norm = relu_minmax_norm(gcam_raw.detach().cpu())
        gcam_512 = resize_512(gcam_norm)

        ig_raw = _sanity_ig_chunked(model, img_gpu, steps=_SANITY_IG_STEPS, chunk_size=chunk_size)
        ig_norm = relu_minmax_norm(ig_raw.cpu())
        ig_512 = resize_512(ig_norm)

        S = ((gcam_512 + ig_512) / 2.0).numpy()
        tvs[global_idx] = compute_isotropic_tv(S)

        pbar.update(1)

        if (batch_idx + 1) % SAVE_EVERY == 0:
            np.save(tmp_path, tvs)

    pbar.close()
    np.save(out_path, tvs)
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"  Saved → {out_path}  mean={tvs.mean():.6f}  std={tvs.std():.6f}")
    return tvs


def _sanity_print_and_log_report(random_results: dict, trained_means: dict) -> None:
    lines = [
        "",
        "=== SANITY CHECK (R1.C2) ===",
    ]

    arch_labels = {
        "unet_r34": "U-Net    ",
        "segformer_b2": "SegFormer",
    }

    for arch_key, label in arch_labels.items():
        random_tv = random_results.get(arch_key)
        trained_tv = trained_means.get(arch_key)

        if random_tv is None:
            lines.append(f"{label} RANDOM TV: (not computed)")
            continue

        random_str = f"{random_tv:.6f}"
        trained_str = f"{trained_tv:.6f}" if trained_tv is not None else "(pending)"

        if trained_tv is not None:
            rel_diff = abs(random_tv - trained_tv) / (trained_tv + 1e-12)
            sensitive = rel_diff >= _SANITY_SENSITIVITY_THRESHOLD
            verdict = "SENSITIVE ✓" if sensitive else "NOT SENSITIVE ✗"
            diff_str = f"  (|Δ|/trained = {rel_diff:.1%})"
        else:
            verdict = "(trained TV pending — re-run after step2 completes)"
            diff_str = ""

        lines.append(
            f"{label} RANDOM TV: {random_str} | TRAINED TV: {trained_str}" f" → {verdict}{diff_str}"
        )

    report = "\n".join(lines)
    print(report)

    _SANITY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SANITY_LOG_FILE, "a") as fh:
        fh.write(report + "\n")
    print(f"\n  Report appended → {_SANITY_LOG_FILE}")


def cmd_sanity_check(args: argparse.Namespace) -> None:
    global DEVICE
    if args.device is not None:
        DEVICE = args.device

    _SANITY_TV_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [DEVICE] {DEVICE}")
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  [CUDA] benchmark=True  (expandable_segments disabled for vGPU compat)")

    random_results = {}

    for model_key in args.models:
        arch_name, num_classes, out_str = _SANITY_RANDOM_MODELS[model_key]
        out_path = Path(out_str)
        tvs = _sanity_process_random_model(
            model_key,
            arch_name,
            num_classes,
            out_path,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
        )
        random_results[model_key] = float(tvs.mean())

    print("\n" + "=" * 60)
    print("RANDOMIZATION SANITY CHECK COMPLETE")
    for model_key, _mean_tv in random_results.items():
        out_path = Path(_SANITY_RANDOM_MODELS[model_key][2])
        arr = np.load(out_path)
        print(f"  {model_key:20s}  n={len(arr):,}  " f"mean={arr.mean():.6f}  std={arr.std():.6f}")
    print("=" * 60)

    _sanity_print_and_log_report(random_results, dict(_SANITY_TRAINED_TV_MEANS))


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: coherence-check
# ══════════════════════════════════════════════════════════════════════════════

_COHERENCE_MODELS = {
    "unet_r34": {
        "arch": "unet_r34",
        "dir": Path("outputs/unet_r34_v2/seed2027"),
        "seed": 2027,
        "num_cls": 1,
    },
    "segformer_b2": {
        "arch": "segformer_b2",
        "dir": Path("outputs/segformer_b2_v2/seed2027"),
        "seed": 2027,
        "num_cls": 1,
    },
}
_COHERENCE_MODEL_ARCH = {
    "unet_r34": ("unet_r34", 1),
    "segformer_b2": ("segformer_b2", 1),
}
_COHERENCE_TEST_CSV = Path("data/manifests/splits/test_split.csv")
_COHERENCE_N_SUBSET = 200
_COHERENCE_IG_STEPS = 50
_COHERENCE_IG_CHUNK = 25
_COHERENCE_OLD_UNET_TV = 0.0224
_COHERENCE_OLD_SEG_TV = 0.0209


def _coherence_robust_load_model(model_key: str, model_dir: Path) -> torch.nn.Module:
    cfg_path = model_dir / "config_train.json"
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "last.pt"

    if cfg_path.exists() and ckpt_path.exists():
        return load_model_from_dir(str(model_dir))

    print(f"  [WARN] config_train.json absent — using hard-coded arch for '{model_key}'")
    arch_name, num_classes = _COHERENCE_MODEL_ARCH[model_key]
    model = build_model(arch_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"  [OK] Loaded {ckpt_path}")
    return model


def _coherence_ig(
    model: torch.nn.Module,
    image: torch.Tensor,
    steps: int = _COHERENCE_IG_STEPS,
    chunk_size: int = _COHERENCE_IG_CHUNK,
) -> torch.Tensor:
    """Riemann-sum IG. FP32, no AMP — intentional paper spec. Returns (H, W) CPU [0, 1]."""
    x = image.unsqueeze(0)
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)

    grad_accum = torch.zeros_like(x)

    for start in range(0, steps, chunk_size):
        chunk_a = alphas[start : start + chunk_size]
        c = len(chunk_a)

        scaled = baseline + chunk_a.view(c, 1, 1, 1) * (x - baseline)
        scaled = scaled.detach().requires_grad_(True)

        # FP32 — no autocast
        logits = model(scaled)

        if logits.ndim == 4:
            score = (
                (logits[:, 1:2] if logits.shape[1] > 1 else logits[:, 0:1])
                .mean(dim=(1, 2, 3))
                .sum()
            )
        else:
            score = (logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]).sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        grad_accum += scaled.grad.detach().sum(dim=0, keepdim=True)
        del scaled, logits, score

    avg_grads = grad_accum / steps
    attr = ((x - baseline) * avg_grads).abs().sum(dim=1).squeeze(0)
    attr_cpu = attr.detach().cpu()
    return attr_cpu / (attr_cpu.max() + EPSILON)


def _coherence_evaluate_model(model_key: str, cfg: dict, chunk_size: int) -> float:
    print(f"\n[{model_key}  seed={cfg['seed']}]  Loading model ...")
    model = _coherence_robust_load_model(model_key, cfg["dir"])
    model.eval().to(DEVICE)

    full_ds = ForgerySegDataset(
        manifest_csv=str(_COHERENCE_TEST_CSV), img_size=IMG_SIZE, split="test"
    )
    indices = torch.linspace(0, len(full_ds) - 1, steps=_COHERENCE_N_SUBSET).long()
    subset = Subset(full_ds, indices.tolist())
    loader = DataLoader(subset, batch_size=1, num_workers=2, shuffle=False)

    n_chunks = -(-_COHERENCE_IG_STEPS // chunk_size)
    print(
        f"  {_COHERENCE_N_SUBSET} images  |  device={DEVICE}  IG_STEPS={_COHERENCE_IG_STEPS}"
        f"  chunk={chunk_size}x{n_chunks}"
    )

    tvs = []
    pbar = tqdm(loader, unit="img", desc=f"  {model_key}")

    for img, _ in pbar:
        img_gpu = img.squeeze(0).to(DEVICE)

        gcam_raw = gradcampp_or_fallback(model, img_gpu)
        gcam_norm = relu_minmax_norm(gcam_raw.detach().cpu())
        gcam_512 = resize_512(gcam_norm)

        ig_raw = _coherence_ig(model, img_gpu, chunk_size=chunk_size)
        ig_norm = relu_minmax_norm(ig_raw.cpu())
        ig_512 = resize_512(ig_norm)

        S = ((gcam_512 + ig_512) / 2.0).numpy()
        tv = compute_isotropic_tv(S)
        tvs.append(tv)

    mean_tv = float(np.mean(tvs))
    std_tv = float(np.std(tvs))
    print(f"  mean TV = {mean_tv:.6f}  std = {std_tv:.6f}  (n={len(tvs)})")
    return mean_tv


def cmd_coherence_check(args: argparse.Namespace) -> None:
    import json

    chunk_size = args.chunk_size

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  [CUDA] benchmark=True  device={DEVICE}  chunk_size={chunk_size}")
    else:
        print(f"  [DEVICE] {DEVICE}  chunk_size={chunk_size}")

    results_path = Path("outputs/coherence_v2_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"  [RESUME] Loaded existing results from {results_path}: {results}")

    for model_key in args.models:
        if model_key in results:
            print(
                f"  [SKIP] {model_key} already computed (mean TV={results[model_key]:.6f}). "
                "Delete outputs/coherence_v2_results.json to rerun."
            )
            continue
        cfg = _COHERENCE_MODELS[model_key]
        results[model_key] = _coherence_evaluate_model(model_key, cfg, chunk_size=chunk_size)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  [SAVED] {results_path}")

    if "unet_r34" not in results or "segformer_b2" not in results:
        print(f"\n  [PARTIAL] Only have results for: {list(results.keys())}")
        print("  Rerun with remaining models to get the final comparison.")
        return

    unet_tv = results["unet_r34"]
    seg_tv = results["segformer_b2"]
    delta = seg_tv - unet_tv
    pct = 100.0 * delta / unet_tv if unet_tv > 0 else float("nan")
    confirmed = seg_tv < unet_tv

    print("\n" + "=" * 64)
    print("=== EARLY DIRECTIONAL CHECK (200-image subset, step2 protocol) ===")
    print(f"U-Net     mean TV (isotropic, GradCAM++⊕IG): {unet_tv:.6f}")
    print(f"SegFormer mean TV (isotropic, GradCAM++⊕IG): {seg_tv:.6f}")
    print(f"Delta: {delta:+.6f} ({pct:+.1f}%)")
    direction = "CONFIRMED" if confirmed else "NOT CONFIRMED"
    print(f"Direction: SegFormer < U-Net -> {direction}")
    print()
    print("Cross-check against old protocol (from xai_coherence_summary.csv):")
    print(f"U-Net     mean TV (anisotropic, InputGrad):   {_COHERENCE_OLD_UNET_TV:.4f}")
    print(f"SegFormer mean TV (anisotropic, InputGrad):   {_COHERENCE_OLD_SEG_TV:.4f}")
    print("Note: values are NOT comparable -- different method and formula.")
    print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: freq-experiment
# ══════════════════════════════════════════════════════════════════════════════

_FREQ_BEST_SEG_DIR = Path("outputs/segformer_b2_v2/seed2027")
_FREQ_TRAIN_CSV = Path("data/manifests/splits/train_split.csv")
_FREQ_VAL_CSV = Path("data/manifests/splits/val_split.csv")
_FREQ_TEST_CSV = Path("data/manifests/splits/test_split.csv")
_FREQ_TV_DIR = Path("outputs/tv_arrays")
_FREQ_METRICS_DIR = Path("outputs/metrics")
_FREQ_OUT_DIR = Path("outputs/freq_experiment")
_FREQ_CKPT_DIR = _FREQ_OUT_DIR / "checkpoints"
_FREQ_MANIFEST_DIR = _FREQ_OUT_DIR / "manifests"
_FREQ_LOG_FILE = Path("logs/post_step2.log")

_FREQ_BATCH_SIZE_TRAIN = 4
_FREQ_BATCH_SIZE_EVAL = 8
_FREQ_FINE_TUNE_STEPS = 5000
_FREQ_VAL_EVERY = 500
_FREQ_VAL_SAMPLE = 2000
_FREQ_LR = 5e-6
_FREQ_WEIGHT_DECAY = 0.05
_FREQ_GRAD_CLIP = 1.0
_FREQ_IG_STEPS = 50
_FREQ_IG_CHUNK_SIZE = 25  # may be overridden via global in cmd_freq_experiment
_FREQ_SEG_SEEDS = [1337, 2027, 3141]


def _freq_srm_kernels_30() -> torch.Tensor:
    """
    30 fixed SRM-inspired high-pass residual filters (3×3).
    All kernels sum to zero (proper residual property).
    Based on Fridrich & Kodovsky 2012 (Rich Models for Steganalysis).
    Returns: (30, 1, 3, 3) float32 tensor.
    """
    raw = [
        # Group 1 — First-order axis-aligned (4)
        [0, 0, 0, -1, 1, 0, 0, 0, 0],  # H forward
        [0, 0, 0, 0, 1, -1, 0, 0, 0],  # H backward
        [0, -1, 0, 0, 1, 0, 0, 0, 0],  # V forward
        [0, 0, 0, 0, 1, 0, 0, -1, 0],  # V backward
        # Group 2 — First-order diagonal (4)
        [-1, 0, 0, 0, 1, 0, 0, 0, 0],  # D+ forward
        [0, 0, 0, 0, 1, 0, 0, 0, -1],  # D+ backward
        [0, 0, -1, 0, 1, 0, 0, 0, 0],  # D- forward
        [0, 0, 0, 0, 1, 0, -1, 0, 0],  # D- backward
        # Group 3 — Second-order axis-aligned (4)
        [0, 0, 0, 1, -2, 1, 0, 0, 0],  # H 2nd order
        [0, 1, 0, 0, -2, 0, 0, 1, 0],  # V 2nd order
        [1, 0, 0, 0, -2, 0, 0, 0, 1],  # D+ 2nd order
        [0, 0, 1, 0, -2, 0, 1, 0, 0],  # D- 2nd order
        # Group 4 — Second-order single-row / single-column (4)
        [1, -2, 1, 0, 0, 0, 0, 0, 0],  # H top row
        [0, 0, 0, 0, 0, 0, 1, -2, 1],  # H bottom row
        [1, 0, 0, -2, 0, 0, 1, 0, 0],  # V left col
        [0, 0, 1, 0, 0, -2, 0, 0, 1],  # V right col
        # Group 5 — Laplacian variants (4)
        [0, -1, 0, -1, 4, -1, 0, -1, 0],  # L4  (4-neighbor)
        [-1, 0, -1, 0, 4, 0, -1, 0, -1],  # L4d (diagonal)
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],  # L8  (8-neighbor)
        [1, -2, 1, -2, 4, -2, 1, -2, 1],  # 2D second-order (DoG)
        # Group 6 — Mixed 2D composites (4)
        [-1, 2, -1, 0, 0, 0, 1, -2, 1],  # Cross-row 2nd
        [-1, 0, 1, 2, 0, -2, -1, 0, 1],  # Sobel-residual H
        [1, 0, -1, 0, 0, 0, -1, 0, 1],  # Sobel-residual D
        [0, 1, -1, 0, 0, 0, 0, -1, 1],  # Corner residual
        # Group 7 — Corner 2nd-order composites (4)
        [-1, 1, 0, 1, -1, 0, 0, 0, 0],  # TL corner
        [0, 1, -1, 0, -1, 1, 0, 0, 0],  # TR corner
        [0, 0, 0, 1, -1, 0, -1, 1, 0],  # BL corner
        [0, 0, 0, 0, -1, 1, 0, 1, -1],  # BR corner
        # Group 8 — Cross-direction composites (2)
        [0, -1, 0, 1, 0, -1, 0, 1, 0],  # Cross V-H
        [1, 0, -1, -1, 0, 1, 0, 0, 0],  # Edge-cross
    ]
    assert len(raw) == 30, f"Expected 30 kernels, got {len(raw)}"
    for i, k in enumerate(raw):
        assert sum(k) == 0, f"Kernel {i + 1} does not sum to zero: sum={sum(k)}"
    return torch.tensor(raw, dtype=torch.float32).view(30, 1, 3, 3)


class SRMLayer(nn.Module):
    """
    Fixed (non-trainable) SRM residual channel extractor.
    Input:  (B, 3, H, W) RGB in [0, 1]
    Output: (B, 1, H, W) averaged residual, clipped to [-1, 1]
    """

    def __init__(self, clip: float = 3.0):
        super().__init__()
        self.register_buffer(
            "rgb_to_gray",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
        )
        self.register_buffer("srm_weight", _freq_srm_kernels_30())
        self.clip = clip
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = (x * self.rgb_to_gray).sum(dim=1, keepdim=True)
        residuals = F.conv2d(gray, self.srm_weight, padding=1)
        residuals = residuals.clamp(-self.clip, self.clip) / self.clip
        return residuals.mean(dim=1, keepdim=True)


def _freq_build_srm_segformer(base_ckpt_path: Path) -> tuple[nn.Module, SRMLayer]:
    model = build_model("segformer_b2", num_classes=1, pretrained=False)

    ckpt = torch.load(str(base_ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"  [OK] Base weights loaded from {base_ckpt_path}")

    old_proj = model.encoder.patch_embed1.proj
    new_proj = nn.Conv2d(
        in_channels=4,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=(old_proj.bias is not None),
    )
    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight.clone()
        nn.init.kaiming_normal_(new_proj.weight[:, 3:4, :, :], mode="fan_out", nonlinearity="relu")
        new_proj.weight[:, 3:4, :, :] *= 0.01
        if old_proj.bias is not None:
            new_proj.bias.data = old_proj.bias.clone()
    model.encoder.patch_embed1.proj = new_proj
    print("  [OK] patch_embed1.proj expanded: 3→64  →  4→64 channels")

    srm_layer = SRMLayer()
    return model, srm_layer


def _freq_make_enhancement_csv(src_csv: Path, dst_csv: Path) -> int:
    if dst_csv.exists():
        with open(dst_csv) as f:
            return sum(1 for _ in f) - 1
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(src_csv, newline="") as fin, open(dst_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        count = 0
        for row in reader:
            if row.get("manip_type") == "enhancement":
                writer.writerow(row)
                count += 1
    print(f"  [CSV] {dst_csv.name}: {count:,} enhancement rows")
    return count


def _freq_get_enhancement_test_indices(test_csv: Path) -> list[int]:
    indices = []
    with open(test_csv, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row.get("manip_type") == "enhancement":
                indices.append(i)
    return indices


def _freq_compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float().view(-1)
    tgts = targets.float().view(-1)
    inter = (preds * tgts).sum()
    union = preds.sum() + tgts.sum() - inter
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2 * inter + 1e-7) / (preds.sum() + tgts.sum() + 1e-7)
    return iou.item(), dice.item()


def _freq_ig_4ch(
    model: nn.Module,
    srm_layer: SRMLayer,
    image: torch.Tensor,
    steps: int = _FREQ_IG_STEPS,
    chunk_size: int = _FREQ_IG_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Riemann-sum IG on the 4-channel SRM+RGB input.
    Baseline = zeros for all 4 channels. Since SRM is linear,
    SRM(α·x) = α·SRM(x), so scaling x4 = [x, SRM(x)] by α is correct.
    Returns (H, W) CPU tensor.
    """
    x = image.unsqueeze(0)
    with torch.no_grad():
        srm_ch = srm_layer(x)
    x4 = torch.cat([x, srm_ch], dim=1)
    baseline = torch.zeros_like(x4)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x4.device)

    grad_accum = torch.zeros_like(x4)

    for start in range(0, steps, chunk_size):
        chunk_a = alphas[start : start + chunk_size]
        c = len(chunk_a)
        scaled = baseline + chunk_a.view(c, 1, 1, 1) * (x4 - baseline)
        scaled = scaled.detach().requires_grad_(True)

        logits = model(scaled)
        score = (logits[:, 0:1].mean(dim=(1, 2, 3))).sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        grad_accum += scaled.grad.detach().sum(dim=0, keepdim=True)
        del scaled, logits, score

    avg_grads = grad_accum / steps
    attr = ((x4 - baseline) * avg_grads).abs().sum(dim=1).squeeze(0)
    attr_cpu = attr.detach().cpu()
    return attr_cpu / (attr_cpu.max() + EPSILON)


def _freq_gradcampp_4ch(
    model: nn.Module,
    srm_layer: SRMLayer,
    image: torch.Tensor,
) -> torch.Tensor:
    """GradCAM++ on the 4-channel SRM+RGB model. Returns (H, W) heatmap."""
    model.eval()
    with torch.no_grad():
        srm_ch = srm_layer(image.unsqueeze(0))
    x4 = torch.cat([image.unsqueeze(0), srm_ch], dim=1)

    target = _find_last_conv(model)
    if target is None:
        x4 = x4.requires_grad_(True)
        logits = model(x4)
        score = logits[:, 0:1].mean()
        model.zero_grad(set_to_none=True)
        score.backward()
        cam = x4.grad[0].abs().mean(0)
        return (cam / (cam.max() + 1e-8)).detach()

    feats = {}

    def _capture(module, inp, out):
        # Detach activation and re-attach as fresh leaf — cuts backward graph
        # here so score.backward() only flows through the tiny segmentation
        # head, not the full transformer. Reduces CUDA peak from ~9.74 GB to
        # ~2 GB on SegFormer-B2 at 512×512.
        A = out.detach().requires_grad_(True)
        feats["A"] = A
        return A

    h1 = target.register_forward_hook(_capture)

    logits = model(x4)
    A = feats["A"]
    score = logits[:, 0:1].mean()
    score.backward()

    G = A.grad.detach()
    A = A.detach()
    weights = (G**2 / (2 * G**2 + (A * G).sum(dim=(2, 3), keepdim=True) + 1e-8)).sum(
        dim=(2, 3), keepdim=True
    )
    cam = (weights * A).sum(dim=1)[0]
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    h1.remove()
    return cam.detach()


def _freq_validate_srm(
    model: nn.Module,
    srm_layer: SRMLayer,
    loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    model.eval()
    srm_layer.eval()
    total_iou = 0.0
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            srm_ch = srm_layer(images)
            x4 = torch.cat([images, srm_ch], dim=1)
            logits = model(x4)
            iou, dice = _freq_compute_binary_metrics(logits, masks)
            total_iou += iou
            total_dice += dice
            n += 1
    return total_iou / n, total_dice / n


def _freq_fine_tune(
    model: nn.Module,
    srm_layer: SRMLayer,
    train_csv: Path,
    val_csv: Path,
    ckpt_dir: Path,
    device: str,
) -> float:
    criterion = HybridForensicLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=_FREQ_LR, weight_decay=_FREQ_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_FREQ_FINE_TUNE_STEPS)

    train_ds = ForgerySegDataset(
        manifest_csv=str(train_csv),
        img_size=IMG_SIZE,
        split="train",
        aug_cfg={"hflip": True, "rot90": True},
    )
    val_ds = ForgerySegDataset(manifest_csv=str(val_csv), img_size=IMG_SIZE, split="test")
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(val_ds), size=min(_FREQ_VAL_SAMPLE, len(val_ds)), replace=False)
    val_sub = Subset(val_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=_FREQ_BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=_FREQ_BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    print(f"\n  Train: {len(train_ds):,}  Val sample: {len(val_sub):,}")
    print(
        f"  Steps: {_FREQ_FINE_TUNE_STEPS}  lr={_FREQ_LR}  wd={_FREQ_WEIGHT_DECAY}  "
        f"grad_clip={_FREQ_GRAD_CLIP}  bs={_FREQ_BATCH_SIZE_TRAIN}"
    )

    best_dice = 0.0
    train_iter = iter(train_loader)
    model.train()
    srm_layer.eval()

    pbar = tqdm(total=_FREQ_FINE_TUNE_STEPS, desc="  Fine-tune SRM+SegFormer")

    for step in range(1, _FREQ_FINE_TUNE_STEPS + 1):
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)

        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            srm_ch = srm_layer(images)
        x4 = torch.cat([images, srm_ch], dim=1)
        logits = model(x4)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        if _FREQ_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _FREQ_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        pbar.update(1)
        pbar.set_description(f"  loss={loss.item():.4f}")

        if step % _FREQ_VAL_EVERY == 0:
            val_iou, val_dice = _freq_validate_srm(model, srm_layer, val_loader, device)
            model.train()
            srm_layer.eval()
            tqdm.write(f"  step={step:5d}  val_iou={val_iou:.4f}  val_dice={val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_iou": val_iou,
                        "val_dice": val_dice,
                    },
                    ckpt_dir / "best.pt",
                )
                tqdm.write(f"    [+] Best saved  dice={val_dice:.4f}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                },
                ckpt_dir / "last.pt",
            )

    pbar.close()
    print(f"\n  Fine-tuning done. Best val dice: {best_dice:.4f}")
    return best_dice


def _freq_evaluate_iou(
    model: nn.Module,
    srm_layer: SRMLayer,
    test_csv: Path,
    device: str,
) -> tuple[np.ndarray, float, float]:
    ds = ForgerySegDataset(manifest_csv=str(test_csv), img_size=IMG_SIZE, split="test")
    loader = DataLoader(
        ds,
        batch_size=_FREQ_BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )
    model.eval()
    srm_layer.eval()
    ious = []
    dices = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  Eval IoU", unit="batch"):
            images = images.to(device)
            masks = masks.to(device)
            srm_ch = srm_layer(images)
            x4 = torch.cat([images, srm_ch], dim=1)
            logits = model(x4)
            for b in range(images.size(0)):
                iou, dice = _freq_compute_binary_metrics(logits[b : b + 1], masks[b : b + 1])
                ious.append(iou)
                dices.append(dice)

    return np.array(ious, dtype=np.float32), float(np.mean(ious)), float(np.mean(dices))


def _freq_evaluate_tv(
    model: nn.Module,
    srm_layer: SRMLayer,
    test_csv: Path,
    out_path: Path,
    device: str,
    overwrite: bool = False,
) -> np.ndarray:
    """Compute per-image Saliency TV on Enhancement test set. Resumes from .tmp.npy."""
    tmp_path = out_path.with_suffix(".tmp.npy")

    ds = ForgerySegDataset(manifest_csv=str(test_csv), img_size=IMG_SIZE, split="test")
    N = len(ds)
    tvs = np.full(N, np.nan, dtype=np.float32)

    start_idx = 0
    if tmp_path.exists() and not overwrite:
        tmp = np.load(tmp_path)
        if len(tmp) == N:
            done = int(np.sum(~np.isnan(tmp)))
            if done > 0:
                tvs = tmp
                start_idx = done
                print(f"  [RESUME TV] {done:,}/{N:,}")

    remaining = list(range(start_idx, N))
    if not remaining:
        print("  TV already complete.")
        return tvs

    loader = DataLoader(
        Subset(ds, remaining),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    model.eval()
    srm_layer.eval()
    # Disable parameter gradients for the entire TV loop: GradCAM++ and IG
    # only need activation/input gradients, not parameter gradients.  With
    # requires_grad=True on all 25M params, PyTorch retains ALL intermediate
    # activations for parameter-gradient computation, inflating backward
    # memory from ~2 GB to ~9.7 GB on SegFormer-B2 at 512×512.
    model.requires_grad_(False)

    pbar = tqdm(total=len(remaining), unit="img", desc="  SRM TV")

    for batch_idx, (img, _) in enumerate(loader):
        global_idx = start_idx + batch_idx
        img_gpu = img.squeeze(0).to(device)

        gcam_raw = _freq_gradcampp_4ch(model, srm_layer, img_gpu)
        gcam_norm = relu_minmax_norm(gcam_raw.detach().cpu())
        gcam_512 = resize_512(gcam_norm)

        ig_raw = _freq_ig_4ch(model, srm_layer, img_gpu, chunk_size=5)  # hardcoded for OOM safety
        ig_norm = relu_minmax_norm(ig_raw.cpu())
        ig_512 = resize_512(ig_norm)

        S = ((gcam_512 + ig_512) / 2.0).numpy()
        tvs[global_idx] = compute_isotropic_tv(S)

        pbar.update(1)
        if (batch_idx + 1) % SAVE_EVERY == 0:
            np.save(tmp_path, tvs)

    pbar.close()
    np.save(out_path, tvs)
    if tmp_path.exists():
        tmp_path.unlink()
    return tvs


def _freq_extract_rgb_enhancement_tv(test_csv: Path, out_path: Path) -> np.ndarray:
    if out_path.exists():
        return np.load(out_path)

    indices = _freq_get_enhancement_test_indices(test_csv)
    seed_arrays = []
    for seed in _FREQ_SEG_SEEDS:
        p = _FREQ_TV_DIR / f"segformer_b2_v2_seed{seed}_tv.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing step2 TV array: {p}\n" "Phase C must run after step2 completes."
            )
        arr = np.load(p).astype(np.float64)
        seed_arrays.append(arr[indices])

    rgb_tv = np.mean(np.stack(seed_arrays, axis=0), axis=0).astype(np.float32)
    np.save(out_path, rgb_tv)
    print(f"  Saved RGB baseline TV → {out_path}  n={len(rgb_tv):,}  " f"mean={rgb_tv.mean():.6f}")
    return rgb_tv


def _freq_extract_rgb_enhancement_iou(test_csv: Path) -> float:
    indices = _freq_get_enhancement_test_indices(test_csv)
    seed_means = []
    for seed in _FREQ_SEG_SEEDS:
        p = _FREQ_METRICS_DIR / f"segformer_b2_seed{seed}_per_sample_iou.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing step2b IoU array: {p}")
        arr = np.load(p).astype(np.float64)
        seed_means.append(float(arr[indices].mean()))
    return float(np.mean(seed_means))


def _freq_append_report(
    rgb_iou: float,
    rgb_tv: float,
    srm_iou: float,
    srm_tv: float,
) -> None:
    delta = srm_tv - rgb_tv
    delta_pct = 100.0 * delta / (rgb_tv + 1e-12)

    if abs(delta_pct) < 2.0:
        direction = "UNCHANGED"
    elif delta < 0:
        direction = "DECREASED"
    else:
        direction = "INCREASED"

    interpretation = (
        "DOES reduce attribution fragmentation"
        if delta < 0
        else "DOES NOT reduce attribution fragmentation"
    )

    lines = [
        "",
        "=== FREQUENCY EXPERIMENT (R1.C3) ===",
        f"Enhancement subset — RGB-only SegFormer:  " f"IoU={rgb_iou:.4f}  |  mean TV={rgb_tv:.6f}",
        f"Enhancement subset — SRM+RGB SegFormer:   " f"IoU={srm_iou:.4f}  |  mean TV={srm_tv:.6f}",
        f"TV delta: {delta:+.6f} ({delta_pct:+.2f}%) — {direction}",
        f"Interpretation: frequency input {interpretation} on Enhancement forgeries",
    ]
    report = "\n".join(lines)
    print(report)

    _FREQ_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_FREQ_LOG_FILE, "a") as fh:
        fh.write(report + "\n")
    print(f"\n  Report appended → {_FREQ_LOG_FILE}")


def cmd_freq_experiment(args: argparse.Namespace) -> None:
    global _FREQ_IG_CHUNK_SIZE
    _FREQ_IG_CHUNK_SIZE = args.chunk_size

    _FREQ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FREQ_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _FREQ_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  [CUDA] benchmark=True  (expandable_segments disabled for vGPU compat)")
    print(f"  [DEVICE] {DEVICE}")

    enh_train_csv = _FREQ_MANIFEST_DIR / "enhancement_train.csv"
    enh_val_csv = _FREQ_MANIFEST_DIR / "enhancement_val.csv"
    enh_test_csv = _FREQ_MANIFEST_DIR / "enhancement_test.csv"

    n_train = _freq_make_enhancement_csv(_FREQ_TRAIN_CSV, enh_train_csv)
    n_val = _freq_make_enhancement_csv(_FREQ_VAL_CSV, enh_val_csv)
    n_test = _freq_make_enhancement_csv(_FREQ_TEST_CSV, enh_test_csv)
    print(f"  Enhancement split: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    base_ckpt = _FREQ_BEST_SEG_DIR / "best.pt"
    print(f"\n  Building SRM+SegFormer from {base_ckpt} …")
    model, srm_layer = _freq_build_srm_segformer(base_ckpt)

    if args.skip_train:
        saved_ckpt = _FREQ_CKPT_DIR / "best.pt"
        if not saved_ckpt.exists():
            raise FileNotFoundError(
                f"--skip-train set but {saved_ckpt} does not exist. "
                "Run without --skip-train first."
            )
        ckpt = torch.load(str(saved_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"  [SKIP-TRAIN] Loaded {saved_ckpt}  "
            f"(val_iou={ckpt.get('val_iou', 0):.4f}  "
            f"val_dice={ckpt.get('val_dice', 0):.4f})"
        )
    else:
        print(f"\n  Fine-tuning for {_FREQ_FINE_TUNE_STEPS} steps on Enhancement train …")
        model.to(DEVICE)
        srm_layer.to(DEVICE)
        _freq_fine_tune(model, srm_layer, enh_train_csv, enh_val_csv, _FREQ_CKPT_DIR, DEVICE)
        best_state = torch.load(
            str(_FREQ_CKPT_DIR / "best.pt"), map_location="cpu", weights_only=False
        )
        model.load_state_dict(best_state["model_state_dict"])
        print("  [OK] Reloaded best.pt for evaluation")

    model.to(DEVICE)
    srm_layer.to(DEVICE)
    model.eval()
    srm_layer.eval()

    print(f"\n  Evaluating IoU on Enhancement test ({n_test:,} images) …")
    srm_iou_arr, mean_srm_iou, mean_srm_f1 = _freq_evaluate_iou(
        model, srm_layer, enh_test_csv, DEVICE
    )
    iou_out = _FREQ_OUT_DIR / "segformer_b2_srm_iou_enhancement.npy"
    if args.overwrite or not iou_out.exists():
        np.save(iou_out, srm_iou_arr)
    print(f"  SRM+RGB  IoU={mean_srm_iou:.4f}  F1={mean_srm_f1:.4f}")
    print(f"  Saved → {iou_out}")

    print("\n  Extracting RGB-only baseline IoU (from step2b arrays) …")
    mean_rgb_iou = _freq_extract_rgb_enhancement_iou(_FREQ_TEST_CSV)
    print(f"  RGB-only IoU={mean_rgb_iou:.4f}  (mean over 3 seeds, Enhancement subset)")

    rgb_tv_out = _FREQ_OUT_DIR / "segformer_b2_rgb_tv_enhancement.npy"
    print("\n  Extracting RGB-only baseline TV (from step2 arrays) …")
    rgb_tv_arr = _freq_extract_rgb_enhancement_tv(_FREQ_TEST_CSV, rgb_tv_out)
    mean_rgb_tv = float(rgb_tv_arr.mean())
    print(f"  RGB-only mean TV={mean_rgb_tv:.6f}  (Enhancement subset, 3-seed mean)")

    if not args.skip_tv:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        srm_tv_out = _FREQ_OUT_DIR / "segformer_b2_srm_tv_enhancement.npy"
        print(f"\n  Computing SRM Saliency TV on Enhancement test ({n_test:,} images) …")
        srm_tv_arr = _freq_evaluate_tv(
            model,
            srm_layer,
            enh_test_csv,
            srm_tv_out,
            DEVICE,
            overwrite=args.overwrite,
        )
        mean_srm_tv = float(srm_tv_arr.mean())
        print(f"  SRM+RGB  mean TV={mean_srm_tv:.6f}")
        print(f"  Saved → {srm_tv_out}")
    else:
        print("\n  [--skip-tv] Saliency TV evaluation skipped.")
        mean_srm_tv = float("nan")

    print("\n" + "=" * 60)
    print("FREQUENCY EXPERIMENT COMPLETE")
    print(f"  RGB-only SegFormer:  IoU={mean_rgb_iou:.4f}  TV={mean_rgb_tv:.6f}")
    print(f"  SRM+RGB  SegFormer:  IoU={mean_srm_iou:.4f}  TV={mean_srm_tv:.6f}")
    print("=" * 60)

    if not args.skip_tv:
        _freq_append_report(mean_rgb_iou, mean_rgb_tv, mean_srm_iou, mean_srm_tv)


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMAND: run-all
# ══════════════════════════════════════════════════════════════════════════════


def _run_subprocess(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n[RUN-ALL] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ABORT] Subprocess exited with code {result.returncode}.")
        sys.exit(result.returncode)


def cmd_run_all(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    chunk_args = ["--chunk-size", str(args.chunk_size)]
    overwrite_args = ["--overwrite"] if args.overwrite else []

    # Step 1: coherence-check — fast diagnostic (~minutes), run first
    print("\n" + "▶" * 3 + "  RUN-ALL STEP 1/4: coherence-check (200-image diagnostic) …")
    _run_subprocess(script, ["coherence-check"] + chunk_args)

    # Step 2: per-sample-iou — full test-set inference (hours)
    print("\n" + "▶" * 3 + "  RUN-ALL STEP 2/4: per-sample-iou (full test set) …")
    seed_args = ["--seeds"] + [str(s) for s in args.seeds]
    _run_subprocess(script, ["per-sample-iou"] + overwrite_args + seed_args)

    # Step 3: sanity-check — random model saliency TV (~150h)
    print("\n" + "▶" * 3 + "  RUN-ALL STEP 3/4: sanity-check (R1.C2 randomization) …")
    _run_subprocess(script, ["sanity-check"] + chunk_args + overwrite_args)

    # Step 4: freq-experiment — SRM fine-tune + TV (depends on steps 2+3 and
    # paper_stats.py compute-tv outputs; will error if TV arrays absent)
    print("\n" + "▶" * 3 + "  RUN-ALL STEP 4/4: freq-experiment (R1.C3 SRM augmentation) …")
    freq_args = list(overwrite_args)
    if args.skip_train:
        freq_args.append("--skip-train")
    if args.skip_tv:
        freq_args.append("--skip-tv")
    _run_subprocess(script, ["freq-experiment"] + freq_args)

    print("\n✓ All run-all steps completed successfully.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_analyses.py",
        description=(
            "Unified runner for reviewer-response analysis scripts " "(R1.C2, R1.C3, R2.C4)"
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # ── per-sample-iou ────────────────────────────────────────────────────────
    sp1 = sub.add_parser(
        "per-sample-iou",
        help="R2.C4: per-image IoU arrays + one-sided Wilcoxon signed-rank test",
    )
    sp1.add_argument("--seeds", nargs="+", type=int, default=_PSIOU_SEEDS)
    sp1.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=list(_PSIOU_MODELS.keys()),
        choices=list(_PSIOU_MODELS.keys()),
    )
    sp1.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if output .npy files exist",
    )
    sp1.add_argument(
        "--wilcoxon-only",
        action="store_true",
        help="Skip inference; only run Wilcoxon if all 6 arrays exist",
    )

    # ── sanity-check ──────────────────────────────────────────────────────────
    sp2 = sub.add_parser(
        "sanity-check",
        help="R1.C2: parameter randomization sanity check (~150h GPU)",
    )
    sp2.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if output .npy files exist",
    )
    sp2.add_argument(
        "--chunk-size",
        type=int,
        default=_SANITY_IG_CHUNK_SIZE,
        help=f"IG steps per micro-batch (default {_SANITY_IG_CHUNK_SIZE}). Use 5 if CUDA OOM.",
    )
    sp2.add_argument(
        "--models",
        nargs="+",
        default=list(_SANITY_RANDOM_MODELS.keys()),
        choices=list(_SANITY_RANDOM_MODELS.keys()),
        help="Which random models to process",
    )
    sp2.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device: 'cpu' or 'cuda' (default: auto-detect)",
    )

    # ── coherence-check ───────────────────────────────────────────────────────
    sp3 = sub.add_parser(
        "coherence-check",
        help="Early directional TV check: 200-image subset, seed2027, FP32 IG (~minutes)",
    )
    sp3.add_argument(
        "--chunk-size",
        type=int,
        default=_COHERENCE_IG_CHUNK,
        help=f"IG steps per micro-batch (default {_COHERENCE_IG_CHUNK}). Use 1 for SegFormer OOM.",
    )
    sp3.add_argument(
        "--models",
        nargs="+",
        default=list(_COHERENCE_MODELS.keys()),
        choices=list(_COHERENCE_MODELS.keys()),
        help="Which models to evaluate (default: all)",
    )

    # ── freq-experiment ───────────────────────────────────────────────────────
    sp4 = sub.add_parser(
        "freq-experiment",
        help="R1.C3: SRM+RGB SegFormer fine-tuning and TV comparison on Enhancement subset",
    )
    sp4.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip fine-tuning; load existing best.pt from checkpoints dir",
    )
    sp4.add_argument(
        "--skip-tv",
        action="store_true",
        help="Skip Saliency TV evaluation (IoU only)",
    )
    sp4.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output arrays",
    )
    sp4.add_argument(
        "--chunk-size",
        type=int,
        default=_FREQ_IG_CHUNK_SIZE,
        help=f"IG micro-batch size for fine-tuning path (default {_FREQ_IG_CHUNK_SIZE})",
    )

    # ── run-all ───────────────────────────────────────────────────────────────
    sp5 = sub.add_parser(
        "run-all",
        help=(
            "Run all four analyses in dependency order as subprocesses: "
            "coherence-check → per-sample-iou → sanity-check → freq-experiment"
        ),
    )
    sp5.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="IG micro-batch size forwarded to coherence-check and sanity-check (default 25)",
    )
    sp5.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward --overwrite to all subcommands",
    )
    sp5.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=_PSIOU_SEEDS,
        help="Seeds forwarded to per-sample-iou",
    )
    sp5.add_argument(
        "--skip-train",
        action="store_true",
        help="Forward --skip-train to freq-experiment",
    )
    sp5.add_argument(
        "--skip-tv",
        action="store_true",
        help="Forward --skip-tv to freq-experiment",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "per-sample-iou": cmd_per_sample_iou,
        "sanity-check": cmd_sanity_check,
        "coherence-check": cmd_coherence_check,
        "freq-experiment": cmd_freq_experiment,
        "run-all": cmd_run_all,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
