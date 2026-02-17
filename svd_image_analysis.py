#!/usr/bin/env python3
"""
About: 
    This script uses svd to compress images and shows the effect of the number of singular values on the reconstruction error and energy captured.
    It uses the Olivetti faces dataset from scikit-learn.
Install dependencies:
    python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
Usage:
    python svd_image_analysis.py
Outputs:
    - olivetti_curves_img33_[time]: Average rank-k reconstruction error and energy curves for image idx 33
    - olivetti_rankk_img33_[time].png: Rank-k approximations for image idx 33 with k = 5, 10, 20, 30
"""

import os
import sys
from datetime import datetime
import matplotlib

# Non-GUI backend; saves PNGs
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.linalg import svd  # noqa: E402

try:
    from sklearn import datasets  # noqa: E402
except ImportError:
    print("scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)


def svd_reconstruct(M, k):
    """Best rank-k approximation to M (notebook logic)."""
    U, s, Vh = svd(M, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vhk = Vh[:k, :]
    return Uk @ np.diag(sk) @ Vhk


def l1_error(M, A):
    """Mean L1 error (per pixel)."""
    return np.abs(M - A).sum() / M.size


# Below method computes the average L1 error and energy fraction for a given list of k values and images.
def compute_avg_energy_frac(k_vals, images):
    """
    For each k, compute over `images`:
    - mean L1 reconstruction error for rank-k
    - mean fraction of total energy (Frobenius^2) in top-k singular values.
    """
    errs, fracs_out = [], []
    for k in k_vals:
        errors = [l1_error(M, svd_reconstruct(M, k)) for M in images]
        errs.append(np.mean(errors))
        fracs = []
        for M in images:
            _, s, _ = svd(M, full_matrices=False)
            total = (s ** 2).sum()
            fracs.append((s[:k] ** 2).sum() / total if total > 0 else 0)
        fracs_out.append(np.mean(fracs))
    return errs, fracs_out


def main():
    print("Fetching Olivetti faces dataset...")
    data = datasets.fetch_olivetti_faces()
    images = data.images  # (400, 64, 64)
    print(f"Loaded dataset: {images.shape[0]} images of shape {images.shape[1:]}.\n")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Part (a) curves k=1..30
    print("Computing average L1 error and energy curves for k = 1..30 ...")
    k_curve = np.arange(1, 31)
    avg_errors, avg_energy_frac = compute_avg_energy_frac(k_curve, images)
    print("Curves computed.")

    # Part (b) image idx 33, k = 5, 10, 20, 30 (notebook)
    print("Building rank-k panel for image idx 33 with k = 5, 10, 20, 30 ...")
    img_idx = 33
    M = images[img_idx]
    m, n = M.shape
    full_size = m * n
    k_vals = np.array([5, 10, 20, 30])
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
    
    plt.title("Average rank-k reconstruction error and energy (Olivetti faces)")
    # Bottom notices
    ax1.text(
        0.5, -0.18,
        "Notice: average rank-k reconstruction error decreases as k increases.",
        ha="center", va="center", transform=ax1.transAxes, fontsize=9, color="tab:blue"
    )
    fig.tight_layout()

    curves_path = os.path.join(os.getcwd(), "output", f"olivetti_curves_img{img_idx}_{ts}.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved curves to: {curves_path}")

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
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

    ax2.text(
        0.5, -0.30,
        "Notice: % energy captured is not the same as visual quality.",
        ha="center", va="center", transform=ax2.transAxes, fontsize=9, color="tab:orange"
    )
    plt.suptitle("Rank-k approximations. Storage < original only when k < mn/(m+n+1).\n")
    # Notice below the photos
    plt.figtext(
        0.5,
        -0.08,
        "Notice: % energy captured is not the same as visual quality.",
        ha="center",
        va="center",
        fontsize=9,
        color="tab:orange",
    )
    plt.tight_layout()
    rankk_path = os.path.join(os.getcwd(), "output", f"olivetti_rank_k_img{img_idx}_{ts}.png")
    plt.savefig(rankk_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rank-k panel to: {rankk_path}")


if __name__ == "__main__":
    main()
