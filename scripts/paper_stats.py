#!/usr/bin/env python
# ===============================================================================
# PROJECT      : DF2023-XAI (Explainable AI for Deepfake Detection)
# SCRIPT       : paper_stats.py
# VERSION      : 1.0.0
# DESCRIPTION  : Unified paper statistics pipeline — Steps 1–5 as subcommands.
# -------------------------------------------------------------------------------
# SUBCOMMANDS:
#     verify-checkpoints  Step 1: verify all 6 model checkpoints exist
#     compute-tv          Step 2: compute per-image saliency TV (GPU, resumable)
#     bootstrap-ci        Step 3: percentile bootstrap 95% CI on saliency TV
#     metric-std          Step 4: Table IV mean±std across 3 seeds
#     directional-check   Step 5: verify TV(U-Net) > TV(SegFormer) for all seeds
#     run-all             Orchestrate Steps 1–5 (Step 2 runs as subprocess)
# USAGE:
#     python scripts/paper_stats.py verify-checkpoints
#     python scripts/paper_stats.py compute-tv [--chunk-size 5] [--overwrite]
#     python scripts/paper_stats.py bootstrap-ci [--B 10000]
#     python scripts/paper_stats.py metric-std [--unet-metrics 'P,R,F1,IoU,FPR' ...]
#     python scripts/paper_stats.py directional-check
#     python scripts/paper_stats.py run-all [--skip-step2] [--chunk-size 10]
#     bash scripts/run_paper_stats.sh [--skip-step2]
# INPUTS       :
#     - outputs/unet_r34_v2/seed{1337,2027,3141}/best.pt
#     - outputs/segformer_b2_v2/seed{1337,2027,3141}/best.pt
#     - data/manifests/splits/test_split.csv        (compute-tv)
#     - outputs/tv_arrays/*_tv.npy                  (bootstrap-ci, directional-check)
#     - outputs/{model}/seed{seed}/test_metrics.json (metric-std)
# OUTPUTS      :
#     - outputs/tv_arrays/{model}_seed{seed}_tv.npy
#     - outputs/tv_arrays/bootstrap_results.json
#     - outputs/tv_arrays/directional_check.json
#     - Manuscript-ready summary table to stdout
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
#     - PYTORCH_CUDA_ALLOC_CONF is popped at module top before any torch import:
#       cuMemCreate is unsupported on L40-12Q vGPU (expandable_segments crash).
#     - compute-tv always runs as a subprocess inside run-all to guarantee a
#       clean CUDA context; the parent process never initialises PyTorch.
#     - bootstrap_ci() is reproduced verbatim from the paper specification
#       (B=10,000, alpha=0.05, seed=42, percentile method) — do NOT alter.
#     - Steps 3/4/5 call implementation functions directly (no subprocess).
# DEPENDENCIES:
#     - Python >= 3.10
#     - numpy >= 1.24  (all subcommands except compute-tv)
#     - torch >= 2.2, tqdm >= 4.66, df2023xai  (compute-tv only)
# ===============================================================================

import os

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # must precede any torch import

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Shared constants ───────────────────────────────────────────────────────────
SEEDS = [1337, 2027, 3141]

# Step 1 / Step 4: short key → full directory name
MODEL_KEYS: dict[str, str] = {
    "unet": "unet_r34_v2",
    "segformer": "segformer_b2_v2",
}

# Step 2: full model name → output base directory
TV_MODELS: dict[str, str] = {
    "unet_r34_v2": "outputs/unet_r34_v2",
    "segformer_b2_v2": "outputs/segformer_b2_v2",
}

# Step 3 / Step 5: ordered list of full names
MODEL_NAMES: list[str] = list(TV_MODELS.keys())

TV_DIR = Path("outputs/tv_arrays")
OUTPUTS_ROOT = Path("outputs")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — verify-checkpoints
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = ["best.pt"]
OPTIONAL_FILES = [
    "config_train.json",
    "last.pt",
    "test_metrics.json",
    "forensic_metrics.json",
    "metrics.json",
    "results.json",
]


def _check_seed_dir(model_key: str, model_name: str, seed: int) -> dict:
    seed_dir = OUTPUTS_ROOT / model_name / f"seed{seed}"
    status: dict = {
        "model_key": model_key,
        "model_name": model_name,
        "seed": seed,
        "seed_dir": str(seed_dir),
        "dir_exists": seed_dir.is_dir(),
        "required": {},
        "optional": {},
        "metric_file": None,
    }

    for fname in REQUIRED_FILES:
        fpath = seed_dir / fname
        status["required"][fname] = fpath.exists()  # type: ignore[index]

    for fname in OPTIONAL_FILES:
        fpath = seed_dir / fname
        found = fpath.exists()
        status["optional"][fname] = found  # type: ignore[index]
        if found and fname.endswith(".json") and "metric" in fname or "result" in fname:
            status["metric_file"] = str(fpath)

    cfg_path = seed_dir / "config_train.json"
    if cfg_path.exists():
        with contextlib.suppress(Exception):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            status["arch_from_config"] = cfg.get("model", cfg.get("arch", "—"))

    return status


def _print_checkpoint_report(all_status: list[dict]) -> int:
    WIDTH = 72
    missing_total = 0

    print("=" * WIDTH)
    print("STEP 1 — CHECKPOINT VERIFICATION REPORT")
    print("=" * WIDTH)

    for s in all_status:
        print(f"\n  Model : {s['model_name']}  |  Seed : {s['seed']}")
        print(f"  Dir   : {s['seed_dir']}")
        print(f"  Exists: {'✓' if s['dir_exists'] else '✗  ← DIRECTORY MISSING'}")

        if not s["dir_exists"]:
            missing_total += len(REQUIRED_FILES)
            print(f"  ACTION NEEDED: Run training for {s['model_name']} seed {s['seed']} first.")
            continue

        for fname, found in s["required"].items():
            tag = "✓" if found else "✗  ← MISSING (required)"
            if not found:
                missing_total += 1
            print(f"    [{tag}] {fname}")

        if s["metric_file"]:
            print(f"    [✓] metric file : {Path(s['metric_file']).name}")
        else:
            print("    [!] No metric JSON found.  Step 4 will need re-running")
            print("        run_forensic_eval for this seed, or adjust OPTIONAL_FILES above.")

        if "arch_from_config" in s:
            print(f"    [i] arch in config_train.json : {s['arch_from_config']}")

    print("\n" + "=" * WIDTH)
    if missing_total == 0:
        print("OVERALL STATUS: ALL REQUIRED FILES FOUND ✓")
        print("\nNext step: python scripts/paper_stats.py compute-tv")
    else:
        print(f"OVERALL STATUS: {missing_total} REQUIRED FILE(S) MISSING ✗")
        print("\nACTION: Re-run training / evaluation for the seeds flagged above.")
        print("  SegFormer:")
        print("    export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        print("    SEED=<seed> python -m df2023xai.cli.run_train \\")
        print("      --config configs/train_segformer_b2_full.yaml train")
        print("  U-Net:")
        print("    SEED=<seed> python -m df2023xai.cli.run_train \\")
        print("      --config configs/train_unet_r34_full.yaml train")
    print("=" * WIDTH)

    return missing_total


def _verify_checkpoints(args: argparse.Namespace) -> bool:
    models_to_check = {k: v for k, v in MODEL_KEYS.items() if k in args.models}
    all_status = []
    for model_key, model_name in models_to_check.items():
        for seed in args.seeds:
            all_status.append(_check_seed_dir(model_key, model_name, seed))
    missing = _print_checkpoint_report(all_status)
    return missing == 0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — compute-tv
# torch / df2023xai are imported lazily to avoid CUDA init in the parent process
# ══════════════════════════════════════════════════════════════════════════════

_IMG_SIZE = 512
_TV_RES = 512
_IG_STEPS = 50  # paper spec — do NOT reduce
_IG_CHUNK_SIZE = 25  # steps per micro-batch; use 5 if CUDA OOM
_EPSILON = 1e-7
_SAVE_EVERY = 500  # flush .tmp.npy checkpoint every N images

_TV_MODEL_ARCH: dict[str, tuple[str, int]] = {
    "unet_r34_v2": ("unet_r34", 1),
    "segformer_b2_v2": ("segformer_b2", 1),
}


def _compute_tv(args: argparse.Namespace) -> None:  # noqa: PLR0912
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from df2023xai.data.dataset import ForgerySegDataset
    from df2023xai.models.factory import build_model, load_model_from_dir
    from df2023xai.xai.gradcampp import gradcampp_or_fallback

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TV_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        print("  [CUDA] benchmark=True  (expandable_segments disabled for vGPU compat)")

    # ── Inner helpers (capture DEVICE and module imports) ─────────────────────

    def _relu_minmax_norm(t: torch.Tensor) -> torch.Tensor:
        t = torch.relu(t)
        return (t - t.min()) / (t.max() - t.min() + _EPSILON)

    def _resize_512(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            t.unsqueeze(0).unsqueeze(0),
            size=(_TV_RES, _TV_RES),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    def _compute_isotropic_tv(S: np.ndarray) -> float:
        S_t = torch.as_tensor(S, dtype=torch.float32)
        dy = torch.zeros_like(S_t)
        dx = torch.zeros_like(S_t)
        dy[:-1, :] = S_t[1:, :] - S_t[:-1, :]
        dy[-1, :] = -S_t[-1, :]
        dx[:, :-1] = S_t[:, 1:] - S_t[:, :-1]
        dx[:, -1] = -S_t[:, -1]
        return torch.sqrt(dy**2 + dx**2).mean().item()

    def _integrated_gradients_chunked(
        model: torch.nn.Module,
        image: torch.Tensor,
        steps: int = _IG_STEPS,
        chunk_size: int = _IG_CHUNK_SIZE,
    ) -> torch.Tensor:
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
        return attr_cpu / (attr_cpu.max() + _EPSILON)

    def _robust_load_model(model_name: str, model_dir: Path) -> torch.nn.Module:
        cfg_path = model_dir / "config_train.json"
        ckpt_path = model_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "last.pt"

        if cfg_path.exists() and ckpt_path.exists():
            return load_model_from_dir(str(model_dir))

        print(f"  [WARN] config_train.json absent — hard-coded arch for '{model_name}'")
        arch_name, num_classes = _TV_MODEL_ARCH[model_name]
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

    def _process_seed(model_name: str, seed: int, overwrite: bool, chunk_size: int) -> np.ndarray:
        model_dir = Path(TV_MODELS[model_name]) / f"seed{seed}"
        out_path = TV_DIR / f"{model_name}_seed{seed}_tv.npy"
        tmp_path = TV_DIR / f"{model_name}_seed{seed}_tv.tmp.npy"

        print(f"\n[{model_name}  seed={seed}]  Loading model …")
        model = _robust_load_model(model_name, model_dir)
        model.eval().to(DEVICE)

        full_dataset = ForgerySegDataset(
            manifest_csv=str(Path("data/manifests/splits/test_split.csv")),
            img_size=_IMG_SIZE,
            split="test",
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
                    print(f"  [RESUME] {done:,}/{N:,} done — skipping to index {start_idx}")

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

        n_chunks = -(-_IG_STEPS // chunk_size)
        print(
            f"  {len(remaining):,} images remaining  |  device={DEVICE}  "
            f"IG_STEPS={_IG_STEPS}  chunk={chunk_size}×{n_chunks}"
        )

        pbar = tqdm(total=len(remaining), unit="img", desc=f"  {model_name}/seed{seed}")
        for batch_idx, (img, _) in enumerate(loader):
            global_idx = start_idx + batch_idx
            img_gpu = img.squeeze(0).to(DEVICE)

            gcam_raw = gradcampp_or_fallback(model, img_gpu)
            gcam_norm = _relu_minmax_norm(gcam_raw.detach().cpu())
            gcam_512 = _resize_512(gcam_norm)

            ig_raw = _integrated_gradients_chunked(
                model, img_gpu, steps=_IG_STEPS, chunk_size=chunk_size
            )
            ig_norm = _relu_minmax_norm(ig_raw.cpu())
            ig_512 = _resize_512(ig_norm)

            S = ((gcam_512 + ig_512) / 2.0).numpy()
            tvs[global_idx] = _compute_isotropic_tv(S)
            pbar.update(1)

            if (batch_idx + 1) % _SAVE_EVERY == 0:
                np.save(tmp_path, tvs)

        pbar.close()
        np.save(out_path, tvs)
        if tmp_path.exists():
            tmp_path.unlink()
        return tvs

    # ── Main compute-tv loop ──────────────────────────────────────────────────
    for model_name in args.models:
        if model_name not in TV_MODELS:
            print(f"[WARN] Unknown model '{model_name}', skipping.")
            continue
        for seed in args.seeds:
            out_path = TV_DIR / f"{model_name}_seed{seed}_tv.npy"
            if out_path.exists() and not args.overwrite:
                arr = np.load(out_path)
                print(
                    f"[SKIP] {out_path}  n={len(arr):,}  "
                    f"mean={arr.mean():.6f}  — use --overwrite to redo"
                )
                continue
            tvs = _process_seed(model_name, seed, args.overwrite, args.chunk_size)
            print(
                f"  Saved {out_path}  n={len(tvs):,}  "
                f"mean={tvs.mean():.6f}  std={tvs.std():.6f}"
            )

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    for model_name in args.models:
        for seed in args.seeds:
            p = TV_DIR / f"{model_name}_seed{seed}_tv.npy"
            if p.exists():
                a = np.load(p)
                print(
                    f"  {model_name:20s}  seed={seed}  "
                    f"n={len(a):,}  mean={a.mean():.6f}  std={a.std():.6f}"
                )
    print("Next: python scripts/paper_stats.py bootstrap-ci")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — bootstrap-ci
# ══════════════════════════════════════════════════════════════════════════════

# Bootstrap parameters — fixed by paper specification; do NOT alter
_B_ITERS = 10_000
_ALPHA = 0.05
_BOOT_SEED = 42


def bootstrap_ci(
    tv_array: np.ndarray,
    B: int = _B_ITERS,
    alpha: float = _ALPHA,
    seed: int = _BOOT_SEED,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap 95% CI for the mean of tv_array.
    Exact paper specification — do NOT alter.

    Parameters
    ----------
    tv_array : 1-D float array of per-image TV values
    B        : number of bootstrap resamples
    alpha    : significance level  (0.05 → 95% CI)
    seed     : RNG seed for reproducibility

    Returns
    -------
    (mean, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    means = [rng.choice(tv_array, size=len(tv_array), replace=True).mean() for _ in range(B)]
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(tv_array), lower, upper


def _load_tv_array(model_name: str, seed: int) -> np.ndarray:
    path = TV_DIR / f"{model_name}_seed{seed}_tv.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"TV array not found: {path}\n"
            "Run Step 2 first:  python scripts/paper_stats.py compute-tv"
        )
    arr = np.load(path)
    assert arr.ndim == 1 and arr.dtype in (
        np.float32,
        np.float64,
    ), f"Unexpected array shape/dtype for {path}: {arr.shape}, {arr.dtype}"
    return arr.astype(np.float64)


def _bootstrap_ci(args: argparse.Namespace) -> dict:
    all_results: dict = {}
    WIDTH = 72

    print("=" * WIDTH)
    print("STEP 3 — BOOTSTRAP 95% CI ON SALIENCY TV")
    print(f"         B={args.B:,}  |  alpha={_ALPHA}  |  RNG seed={_BOOT_SEED}")
    print("=" * WIDTH)

    for model_name in args.models:
        all_results[model_name] = {}
        seed_means = []

        print(f"\n  Model: {model_name}")
        print(
            f"  {'Seed':>6}  {'N':>8}  {'Mean TV':>12}  "
            f"{'95% CI Lower':>14}  {'95% CI Upper':>14}"
        )
        print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*14}")

        for seed in args.seeds:
            arr = _load_tv_array(model_name, seed)
            mean, lo, hi = bootstrap_ci(arr, B=args.B, alpha=_ALPHA, seed=_BOOT_SEED)

            all_results[model_name][str(seed)] = {
                "n": int(len(arr)),
                "mean_tv": float(mean),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
            }
            seed_means.append(mean)
            print(f"  {seed:>6}  {len(arr):>8,}  {mean:>12.6f}  {lo:>14.6f}  {hi:>14.6f}")

        all_tv = np.concatenate([_load_tv_array(model_name, s) for s in args.seeds])
        pooled_mean, pooled_lo, pooled_hi = bootstrap_ci(
            all_tv, B=args.B, alpha=_ALPHA, seed=_BOOT_SEED
        )
        all_results[model_name]["pooled"] = {
            "n": int(len(all_tv)),
            "mean_tv": float(pooled_mean),
            "ci_lower": float(pooled_lo),
            "ci_upper": float(pooled_hi),
        }
        std_across_seeds = float(np.std(seed_means, ddof=1))
        all_results[model_name]["cross_seed_std"] = std_across_seeds

        print(f"\n  Pooled ({len(all_tv):,} images, all seeds):")
        print(f"    Mean TV = {pooled_mean:.6f}  " f"[95% CI: {pooled_lo:.6f}, {pooled_hi:.6f}]")
        print(f"    Cross-seed std (ddof=1) = {std_across_seeds:.6f}")

    TV_DIR.mkdir(parents=True, exist_ok=True)
    results_json = TV_DIR / "bootstrap_results.json"
    with open(results_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n  Results saved → {results_json}")

    print("\n" + "=" * WIDTH)
    print("PAPER-READY VALUES (pooled across all seeds per architecture):")
    print("=" * WIDTH)
    for model_name in args.models:
        r = all_results[model_name]["pooled"]
        label = "U-Net" if "unet" in model_name else "SegFormer"
        print(
            f"  {label:10s}: Saliency TV: {r['mean_tv']:.4f} "
            f"[95% CI: {r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        )

    print("\nNext step: python scripts/paper_stats.py metric-std")
    print("=" * WIDTH)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — metric-std
# ══════════════════════════════════════════════════════════════════════════════

METRIC_FILE_CANDIDATES = [
    "test_metrics.json",
    "forensic_metrics.json",
    "metrics.json",
    "results.json",
    "eval_results.json",
]

METRIC_ALIASES: dict[str, list[str]] = {
    "Precision": ["precision", "prec", "positive_predictive_value", "ppv"],
    "Recall": ["recall", "sensitivity", "true_positive_rate", "tpr", "rec"],
    "F1": ["f1", "dice", "f1_score", "f1score", "pixel_f1", "dice_score"],
    "IoU": ["iou", "jaccard", "intersection_over_union", "miou", "mean_iou"],
    "FPR": ["fpr", "false_positive_rate", "fall_out"],
}


def _find_metric_file(model_name: str, seed: int) -> Path | None:
    seed_dir = OUTPUTS_ROOT / model_name / f"seed{seed}"
    for fname in METRIC_FILE_CANDIDATES:
        p = seed_dir / fname
        if p.exists():
            return p
    return None


def _extract_value(data: dict, aliases: list[str]) -> float | None:
    lower_data = {k.lower(): v for k, v in data.items()}
    for alias in aliases:
        v = lower_data.get(alias.lower())
        if v is not None:
            return float(v)
    for sub in data.values():
        if isinstance(sub, dict):
            lower_sub = {k.lower(): vv for k, vv in sub.items()}
            for alias in aliases:
                v = lower_sub.get(alias.lower())
                if v is not None:
                    return float(v)
    return None


def _load_seed_metrics(model_name: str, seed: int) -> dict[str, float] | None:
    path = _find_metric_file(model_name, seed)
    if path is None:
        return None

    with open(path) as fh:
        data = json.load(fh)

    result: dict[str, float] = {}
    for metric, aliases in METRIC_ALIASES.items():
        val = _extract_value(data, aliases)
        if val is None:
            print(f"  [WARN] Could not find '{metric}' in {path}")
            print(f"         Tried aliases: {aliases}")
            print(f"         Actual keys:   {list(data.keys())}")
            return None
        result[metric] = val
    return result


def _parse_manual_override(s: str) -> dict[str, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 comma-separated values (Prec,Rec,F1,IoU,FPR), got: {s}")
    keys = list(METRIC_ALIASES.keys())
    return dict(zip(keys, parts, strict=False))


def _compute_std_table(
    model_key: str,
    model_name: str,
    seed_metrics: list[dict[str, float] | None],
    seeds: list[int],
) -> dict[str, dict[str, float]] | None:
    if any(m is None for m in seed_metrics):
        missing = [seeds[i] for i, m in enumerate(seed_metrics) if m is None]
        print(f"\n  [ERROR] Missing metric data for {model_name} seeds: {missing}")
        print(
            "          Run:  python -m df2023xai.cli.run_forensic_eval "
            "--config configs/forensic_eval.yaml"
        )
        print(
            f"          Or pass values manually with --{model_key}-metrics  "
            f"'P,R,F1,IoU,FPR'  (one per seed)"
        )
        return None

    result: dict[str, dict[str, float]] = {}
    for metric in METRIC_ALIASES:
        vals = [m[metric] for m in seed_metrics if m is not None]  # type: ignore[index]
        result[metric] = {
            "values": vals,  # type: ignore[dict-item]
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
        }
    return result


def _print_metric_results(all_results: dict) -> None:
    WIDTH = 72
    print("\n" + "=" * WIDTH)
    print("STEP 4 — TABLE IV METRICS  (mean ± std  |  ddof=1 across 3 seeds)")
    print("=" * WIDTH)

    for _model_key, (model_name, res) in all_results.items():
        if res is None:
            print(f"\n  {model_name}: INCOMPLETE — see errors above")
            continue
        label = "U-Net" if "unet" in model_name else "SegFormer"
        print(f"\n  {label}  ({model_name})")
        print(f"  {'Metric':12s}  {'Seed values':>36s}  {'Mean':>8s}  {'±Std':>8s}")
        print(f"  {'-'*12}  {'-'*36}  {'-'*8}  {'-'*8}")
        for metric, stats in res.items():
            vals_str = ", ".join(f"{v:.4f}" for v in stats["values"])
            print(
                f"  {metric:12s}  {vals_str:>36s}  " f"{stats['mean']:>8.4f}  {stats['std']:>8.4f}"
            )


def _metric_std(args: argparse.Namespace) -> dict:
    seeds = args.seeds
    all_results: dict = {}

    for model_key, model_name in MODEL_KEYS.items():
        override_arg = args.unet_metrics if model_key == "unet" else args.seg_metrics

        if override_arg is not None:
            if len(override_arg) != len(seeds):
                print(
                    f"[ERROR] --{model_key}-metrics requires {len(seeds)} values "
                    f"(one per seed), got {len(override_arg)}"
                )
                sys.exit(1)
            seed_metrics: list[dict[str, float] | None] = [
                _parse_manual_override(s) for s in override_arg
            ]
        else:
            seed_metrics = [_load_seed_metrics(model_name, s) for s in seeds]

        res = _compute_std_table(model_key, model_name, seed_metrics, seeds)
        all_results[model_key] = (model_name, res)

    _print_metric_results(all_results)

    WIDTH = 72
    print("\n" + "=" * WIDTH)
    print("PAPER-READY (Table IV — insert into manuscript):")
    print("=" * WIDTH)
    for _model_key, (model_name, res) in all_results.items():
        if res is None:
            continue
        label = "U-Net" if "unet" in model_name else "SegFormer"
        parts = []
        for metric in ["Precision", "Recall", "F1", "IoU", "FPR"]:
            parts.append(f"{metric}: {res[metric]['mean']:.4f} ± {res[metric]['std']:.4f}")
        print(f"  {label:10s}: {' | '.join(parts)}")

    print("\nNext step: python scripts/paper_stats.py directional-check")
    print("=" * WIDTH)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — directional-check
# ══════════════════════════════════════════════════════════════════════════════


def _directional_check(args: argparse.Namespace) -> tuple[bool, list[dict]]:
    WIDTH = 72
    check_results: list[dict] = []

    print("=" * WIDTH)
    print("STEP 5 — DIRECTIONAL CONSISTENCY CHECK")
    print(f"         Hypothesis: TV({args.unet_name}) > TV({args.seg_name}) for all seeds")
    print("=" * WIDTH)
    print(
        f"\n  {'Seed':>6}  {'TV(U-Net)':>12}  {'TV(SegFormer)':>14}  "
        f"{'Δ (U-U)':>10}  {'Δ/U-Net %':>10}  {'Result':>8}"
    )
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")

    all_pass = True
    unet_means = []
    seg_means = []

    for seed in args.seeds:
        tv_unet = _load_tv_array(args.unet_name, seed)
        tv_seg = _load_tv_array(args.seg_name, seed)

        mu_unet = tv_unet.mean()
        mu_seg = tv_seg.mean()
        delta = mu_unet - mu_seg
        delta_rel = 100.0 * delta / mu_unet if mu_unet > 0 else float("nan")

        passed = mu_unet > mu_seg
        verdict = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False

        print(
            f"  {seed:>6}  {mu_unet:>12.6f}  {mu_seg:>14.6f}  "
            f"{delta:>+10.6f}  {delta_rel:>9.2f}%  {verdict:>8}"
        )

        unet_means.append(mu_unet)
        seg_means.append(mu_seg)
        check_results.append(
            {
                "seed": seed,
                "tv_unet": float(mu_unet),
                "tv_segformer": float(mu_seg),
                "delta": float(delta),
                "delta_rel_%": float(delta_rel),
                "pass": bool(passed),
            }
        )

    print("\n" + "-" * WIDTH)
    print(
        f"  Overall cross-seed mean TV:  U-Net={np.mean(unet_means):.6f}  "
        f"SegFormer={np.mean(seg_means):.6f}"
    )
    print(
        f"  Consistency across seeds:    "
        f"{'ALL SEEDS PASS ✓' if all_pass else 'ONE OR MORE SEEDS FAIL ✗'}"
    )
    print("=" * WIDTH)

    print("\n=== DIRECTIONAL CONSISTENCY CHECK ===")
    for r in check_results:
        verdict = "PASS" if r["pass"] else "FAIL"
        print(
            f"Seed {r['seed']:>4}: TV(U-Net)={r['tv_unet']:.4f} > "
            f"TV(SegFormer)={r['tv_segformer']:.4f} → {verdict}"
        )

    if not all_pass:
        print(
            "\n[WARNING] One or more seeds FAILED the directional check.\n"
            "  Possible causes:\n"
            "  1. The TV arrays were computed with different pipeline settings.\n"
            "  2. A training seed produced an outlier checkpoint.\n"
            "  3. The target-layer selection differs between architectures.\n"
            "  Inspect the TV arrays manually before reporting."
        )

    out = {
        "hypothesis": f"TV({args.unet_name}) > TV({args.seg_name}) for each seed",
        "all_pass": all_pass,
        "per_seed": check_results,
    }
    out_path = TV_DIR / "directional_check.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("\nAll steps complete.  See run-all for the manuscript-ready summary.")
    print("=" * WIDTH)

    return all_pass, check_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — run-all  (orchestrates Steps 1–5)
# ══════════════════════════════════════════════════════════════════════════════


def _print_final_table(
    boot_results: dict,
    metric_results: dict,
    dir_results: list[dict],
) -> None:
    WIDTH = 78
    print("\n\n" + "█" * WIDTH)
    print("  FINAL SUMMARY — PASTE INTO MANUSCRIPT")
    print("█" * WIDTH)

    print("\n=== TABLE IV VALUES ===")
    for model_key, model_name in MODEL_KEYS.items():
        label = "U-Net    " if "unet" in model_name else "SegFormer"

        mres = metric_results.get(model_key, (model_name, None))[1]
        if mres is not None:
            parts = []
            for m in ["Precision", "Recall", "F1", "IoU", "FPR"]:
                parts.append(f"{m}: {mres[m]['mean']:.4f} ± {mres[m]['std']:.4f}")
            metric_str = " | ".join(parts)
        else:
            metric_str = "(metric data unavailable — run metric-std with --Xnet-metrics)"

        pooled = boot_results.get(model_name, {}).get("pooled", {})
        if pooled:
            tv_str = (
                f"Saliency TV: {pooled['mean_tv']:.4f} "
                f"[95% CI: {pooled['ci_lower']:.4f}, {pooled['ci_upper']:.4f}]"
            )
        else:
            tv_str = "Saliency TV: (run bootstrap-ci)"

        print(f"{label}: {metric_str} | {tv_str}")

    print("\n=== DIRECTIONAL CONSISTENCY CHECK ===")
    for r in dir_results:
        verdict = "PASS" if r["pass"] else "FAIL"
        print(
            f"Seed {r['seed']:>4}: "
            f"TV(U-Net)={r['tv_unet']:.4f} > "
            f"TV(SegFormer)={r['tv_segformer']:.4f} → {verdict}"
        )

    all_pass = all(r["pass"] for r in dir_results)
    print(
        f"\nOverall directional consistency: "
        f"{'ALL PASS ✓' if all_pass else 'FAIL ✗ — CHECK STEP 5 OUTPUT'}"
    )
    print("\n" + "█" * WIDTH)


def _run_all(args: argparse.Namespace) -> None:
    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("\n" + "▶" * 3 + "  STEP 1: Verifying checkpoints …")
    step1_args = argparse.Namespace(seeds=args.seeds, models=list(MODEL_KEYS.keys()))
    ckpts_ok = _verify_checkpoints(step1_args)
    if not ckpts_ok:
        print("\n[ABORT] Step 1 failed: missing checkpoints.  See output above.")
        sys.exit(1)

    # ── Step 2 (subprocess for clean CUDA context) ────────────────────────────
    if args.skip_step2:
        print("\n▶▶▶  STEP 2: Skipped (--skip-step2 flag set)")
        missing = [
            str(TV_DIR / f"{m}_seed{s}_tv.npy")
            for m in MODEL_NAMES
            for s in args.seeds
            if not (TV_DIR / f"{m}_seed{s}_tv.npy").exists()
        ]
        if missing:
            print("[ERROR] --skip-step2 was set but the following TV arrays are missing:")
            for m in missing:
                print(f"         {m}")
            print("Remove --skip-step2 to re-compute.")
            sys.exit(1)
    else:
        print("\n▶▶▶  STEP 2: Computing Saliency TV for all images …")
        print("     Running as subprocess for a clean CUDA context (crash-safe).")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "compute-tv",
            "--chunk-size",
            str(args.chunk_size),
            "--seeds",
        ] + [str(s) for s in args.seeds]
        print(f"     CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n[ABORT] Step 2 exited with code {result.returncode}.")
            sys.exit(result.returncode)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("\n▶▶▶  STEP 3: Bootstrap CIs …")
    step3_args = argparse.Namespace(seeds=args.seeds, models=MODEL_NAMES, B=_B_ITERS)
    boot_results = _bootstrap_ci(step3_args)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    print("\n▶▶▶  STEP 4: Computing metric std …")
    step4_args = argparse.Namespace(
        seeds=args.seeds,
        unet_metrics=args.unet_metrics,
        seg_metrics=args.seg_metrics,
    )
    metric_results = _metric_std(step4_args)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    print("\n▶▶▶  STEP 5: Directional consistency check …")
    step5_args = argparse.Namespace(
        seeds=args.seeds,
        unet_name="unet_r34_v2",
        seg_name="segformer_b2_v2",
    )
    all_pass, dir_results = _directional_check(step5_args)

    # ── Final table ───────────────────────────────────────────────────────────
    _print_final_table(boot_results, metric_results, dir_results)

    sys.exit(0 if all_pass else 2)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_stats.py",
        description="DF2023-XAI paper statistics pipeline (Steps 1–5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/paper_stats.py verify-checkpoints\n"
            "  python scripts/paper_stats.py compute-tv --chunk-size 5\n"
            "  python scripts/paper_stats.py bootstrap-ci\n"
            "  python scripts/paper_stats.py metric-std\n"
            "  python scripts/paper_stats.py directional-check\n"
            "  python scripts/paper_stats.py run-all --skip-step2\n"
            "  bash scripts/run_paper_stats.sh --skip-step2\n"
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # ── verify-checkpoints ────────────────────────────────────────────────────
    p1 = sub.add_parser("verify-checkpoints", help="Step 1: verify all 6 model checkpoints exist")
    p1.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p1.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=list(MODEL_KEYS.keys()),
        metavar="MODEL",
        help=f"Short model keys to check (default: all): {list(MODEL_KEYS.keys())}",
    )

    # ── compute-tv ────────────────────────────────────────────────────────────
    p2 = sub.add_parser("compute-tv", help="Step 2: compute per-image saliency TV (GPU, resumable)")
    p2.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p2.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=MODEL_NAMES,
        metavar="MODEL",
        help=f"Full model names to process (default: all): {MODEL_NAMES}",
    )
    p2.add_argument("--overwrite", action="store_true", help="Re-compute even if output exists")
    p2.add_argument(
        "--chunk-size",
        type=int,
        default=_IG_CHUNK_SIZE,
        metavar="N",
        help=f"IG steps per micro-batch (default {_IG_CHUNK_SIZE}). Use 5 if CUDA OOM.",
    )

    # ── bootstrap-ci ─────────────────────────────────────────────────────────
    p3 = sub.add_parser("bootstrap-ci", help="Step 3: percentile bootstrap 95%% CI on saliency TV")
    p3.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p3.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=MODEL_NAMES,
        metavar="MODEL",
        help=f"Full model names to process (default: all): {MODEL_NAMES}",
    )
    p3.add_argument(
        "--B",
        type=int,
        default=_B_ITERS,
        metavar="N",
        help=f"Bootstrap resamples (default {_B_ITERS:,})",
    )

    # ── metric-std ────────────────────────────────────────────────────────────
    p4 = sub.add_parser("metric-std", help="Step 4: Table IV mean±std across 3 seeds")
    p4.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p4.add_argument(
        "--unet-metrics",
        nargs="+",
        type=str,
        default=None,
        metavar="'P,R,F1,IoU,FPR'",
        help="Manual override: one 'P,R,F1,IoU,FPR' string per seed (bypasses JSON lookup)",
    )
    p4.add_argument(
        "--seg-metrics",
        nargs="+",
        type=str,
        default=None,
        metavar="'P,R,F1,IoU,FPR'",
        help="Manual override: one 'P,R,F1,IoU,FPR' string per seed (bypasses JSON lookup)",
    )

    # ── directional-check ─────────────────────────────────────────────────────
    p5 = sub.add_parser(
        "directional-check", help="Step 5: verify TV(U-Net) > TV(SegFormer) for all seeds"
    )
    p5.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p5.add_argument("--unet-name", type=str, default="unet_r34_v2", metavar="NAME")
    p5.add_argument("--seg-name", type=str, default="segformer_b2_v2", metavar="NAME")

    # ── run-all ───────────────────────────────────────────────────────────────
    p6 = sub.add_parser("run-all", help="Orchestrate Steps 1–5 and print manuscript-ready summary")
    p6.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip Step 2 TV computation (use existing .npy files)",
    )
    p6.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        metavar="N",
        help="IG micro-batch size for Step 2 (default 10; use 5 if OOM)",
    )
    p6.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p6.add_argument(
        "--unet-metrics",
        nargs="+",
        type=str,
        default=None,
        metavar="'P,R,F1,IoU,FPR'",
    )
    p6.add_argument(
        "--seg-metrics",
        nargs="+",
        type=str,
        default=None,
        metavar="'P,R,F1,IoU,FPR'",
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.subcommand == "verify-checkpoints":
        ok = _verify_checkpoints(args)
        sys.exit(0 if ok else 1)
    elif args.subcommand == "compute-tv":
        _compute_tv(args)
    elif args.subcommand == "bootstrap-ci":
        _bootstrap_ci(args)
    elif args.subcommand == "metric-std":
        _metric_std(args)
    elif args.subcommand == "directional-check":
        all_pass, _ = _directional_check(args)
        sys.exit(0 if all_pass else 1)
    elif args.subcommand == "run-all":
        _run_all(args)


if __name__ == "__main__":
    main()
