#!/usr/bin/env python3
"""
About: 
    This script uses SVD to compress images and shows the effect of the number of
    singular values on reconstruction error and energy captured.

    By default it uses the Olivetti faces dataset from scikit-learn (may download on
    first run), but you can also run fully offline with a synthetic dataset.
Install dependencies:
    python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
Usage:
    python svd_image_analysis.py --output-dir output
    python svd_image_analysis.py --dataset synthetic --output-dir output --no-timestamp
Outputs:
    - <dataset>_curves_img<idx>_<timestamp>.png: Average rank-k reconstruction error and energy curves
    - <dataset>_rank_k_img<idx>_<timestamp>.png: Rank-k approximations panel for the selected image
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib

# Non-GUI backend; saves PNGs
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from image_compression import compute_avg_energy_frac, svd_reconstruct  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-rank image compression via SVD")
    p.add_argument(
        "--dataset",
        choices=("olivetti", "synthetic"),
        default="olivetti",
        help="Dataset source. 'olivetti' may download on first run.",
    )
    p.add_argument("--output-dir", default="output", help="Directory to save PNG outputs")
    p.add_argument("--img-idx", type=int, default=33, help="Image index for the rank-k panel")
    p.add_argument("--k-max", type=int, default=30, help="Maximum k for curves (plots k=1..k-max)")
    p.add_argument(
        "--k-panel",
        default="5,10,20,30",
        help="Comma-separated k values for the rank-k panel (e.g. '5,10,20,30')",
    )
    p.add_argument("--no-timestamp", action="store_true", help="Use stable output filenames")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic dataset")
    p.add_argument(
        "--n-synthetic",
        type=int,
        default=200,
        help="Number of synthetic images (only used when --dataset synthetic)",
    )
    return p.parse_args(argv)


def _resolve_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_images(dataset: str, seed: int, n_synthetic: int) -> np.ndarray:
    if dataset == "olivetti":
        try:
            from sklearn import datasets  # noqa: E402
        except ImportError:
            print("scikit-learn is required. Install with: pip install scikit-learn")
            sys.exit(1)

        print("Fetching Olivetti faces dataset...")
        data = datasets.fetch_olivetti_faces()
        return data.images  # (400, 64, 64)

    rng = np.random.default_rng(seed)
    # Values in [0, 1], shape similar to Olivetti for comparable plots.
    return rng.random((n_synthetic, 64, 64), dtype=np.float32)


def main():
    args = _parse_args(sys.argv[1:])
    images = _load_images(args.dataset, seed=args.seed, n_synthetic=args.n_synthetic)

    print(f"Loaded dataset '{args.dataset}': {images.shape[0]} images of shape {images.shape[1:]}.\n")
    out_dir = _resolve_output_dir(args.output_dir)

    suffix = "" if args.no_timestamp else f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Part (a) curves k=1..30
    print(f"Computing average L1 error and energy curves for k = 1..{args.k_max} ...")
    if args.k_max < 1:
        raise ValueError("--k-max must be >= 1")
    k_max_allowed = min(images.shape[1], images.shape[2])
    if args.k_max > k_max_allowed:
        raise ValueError(f"--k-max must be <= {k_max_allowed} for image shape {images.shape[1:]}")
    k_curve = np.arange(1, args.k_max + 1)
    avg_errors, avg_energy_frac = compute_avg_energy_frac(k_curve, images)
    print("Curves computed.")

    # Part (b) image idx 33, k = 5, 10, 20, 30 (notebook)
    k_vals = [int(x.strip()) for x in str(args.k_panel).split(",") if x.strip() != ""]
    if len(k_vals) == 0:
        raise ValueError("--k-panel produced no k values (expected comma-separated ints)")

    img_idx = int(args.img_idx)
    if not (0 <= img_idx < images.shape[0]):
        raise ValueError(f"--img-idx must be in [0, {images.shape[0]-1}], got {img_idx}")

    print(f"Building rank-k panel for image idx {img_idx} with k = {k_vals} ...")
    M = images[img_idx]
    m, n = M.shape
    full_size = m * n
    _, avg_energy_frac_k = compute_avg_energy_frac(k_vals, images)
    print("Panel computed.")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(k_curve, avg_errors, "o-", color="tab:blue")
    ax1.set_xlabel("k (number of singular values)")
    ax1.set_ylabel("mean L1 error", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(k_curve, np.array(avg_energy_frac) * 100, "s--", color="C1", alpha=0.8)
    ax2.set_ylabel("energy captured (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    
    plt.title(f"Average rank-k reconstruction error and energy ({args.dataset})")
    fig.tight_layout()

    curves_path = out_dir / f"{args.dataset}_curves_img{img_idx}{suffix}.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved curves to: {curves_path}")

    fig, axes = plt.subplots(1, 1 + len(k_vals), figsize=(3 * (1 + len(k_vals)), 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(M, cmap="gray")
    axes[0].set_title(f"Original\n{full_size} values")
    axes[0].axis("off")

    for i, k in enumerate(k_vals):
        A = svd_reconstruct(M, k)
        axes[i + 1].imshow(A, cmap="gray")
        rank_k_size = k * (m + n + 1)
        pct = 100 * rank_k_size / full_size
        axes[i + 1].set_title(
            f"Rank-{k}\n{rank_k_size} values ({pct:.0f}% of original)\n"
            f"avg_energy_frac: {100 * avg_energy_frac_k[i]:.4f}%"
        )
        axes[i + 1].axis("off")

    plt.suptitle("Rank-k approximations. Storage < original only when k < mn/(m+n+1).")
    plt.tight_layout()

    fig.text(
        0.5,
        -0.02,
        "Notice: energy captured (%) is not the same as visual quality.",
        ha="center",
        va="center",
        fontsize=9,
        color="tab:orange",
    )

    rankk_path = out_dir / f"{args.dataset}_rank_k_img{img_idx}{suffix}.png"
    plt.savefig(rankk_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rank-k panel to: {rankk_path}")


if __name__ == "__main__":
    main()
