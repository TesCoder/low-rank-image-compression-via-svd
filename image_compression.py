from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.linalg import svd
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return a 2D float32 grayscale array in [0, 1].

    - RGB images are converted to grayscale.
    - Output shape is (H, W).
    """
    p = Path(path)
    with Image.open(p) as im:
        im_l = im.convert("L")
        arr = np.asarray(im_l, dtype=np.float32) / 255.0
    return arr


def svd_reconstruct(M: np.ndarray, k: int) -> np.ndarray:
    """Return the best rank-k approximation of a 2D matrix M via SVD."""
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}")
    m, n = M.shape
    k_max = min(m, n)
    if not (1 <= k <= k_max):
        raise ValueError(f"k must be in [1, {k_max}], got {k}")

    U, s, Vh = svd(M, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vhk = Vh[:k, :]
    return Uk @ np.diag(sk) @ Vhk


def l1_error(M: np.ndarray, A: np.ndarray) -> float:
    """Mean L1 reconstruction error (per element)."""
    if M.shape != A.shape:
        raise ValueError(f"shape mismatch: M={M.shape} vs A={A.shape}")
    return float(np.abs(M - A).mean())


def compute_avg_energy_frac(
    k_vals: Sequence[int],
    images: Iterable[np.ndarray],
) -> tuple[list[float], list[float]]:
    """
    For each k, compute over `images`:
    - mean L1 reconstruction error for rank-k
    - mean fraction of total energy (Frobenius^2) in top-k singular values
    """
    imgs = list(images)
    if len(imgs) == 0:
        raise ValueError("images must be a non-empty iterable of 2D arrays")

    k_list = [int(k) for k in k_vals]
    if len(k_list) == 0:
        raise ValueError("k_vals must be non-empty")

    # Work on unique ks for efficiency, but preserve input order in outputs.
    unique_ks = sorted(set(k_list))

    # Validate k bounds based on first image.
    first = imgs[0]
    if first.ndim != 2:
        raise ValueError(f"images must contain 2D arrays, got shape {first.shape}")
    k_max_allowed = min(first.shape)
    for k in unique_ks:
        if not (1 <= k <= k_max_allowed):
            raise ValueError(f"k must be in [1, {k_max_allowed}], got {k}")

    pos_by_k = {k: i for i, k in enumerate(unique_ks)}
    errs_sum = np.zeros(len(unique_ks), dtype=np.float64)
    fracs_sum = np.zeros(len(unique_ks), dtype=np.float64)

    k_set = set(unique_ks)
    k_max_needed = max(unique_ks)

    for M in imgs:
        if M.ndim != 2:
            raise ValueError(f"images must contain 2D arrays, got shape {M.shape}")

        U, s, Vh = svd(M, full_matrices=False)
        s2 = s**2
        total = float(s2.sum())
        cum_energy = np.cumsum(s2)

        # Incremental rank-1 updates avoid repeated full matrix multiplies.
        A = np.zeros_like(M, dtype=np.float64)
        for r in range(k_max_needed):
            A += float(s[r]) * np.outer(U[:, r], Vh[r, :])
            k = r + 1
            if k in k_set:
                i = pos_by_k[k]
                errs_sum[i] += float(np.abs(M - A).mean())
                fracs_sum[i] += float(cum_energy[k - 1] / total) if total > 0 else 0.0

    errs_unique = (errs_sum / len(imgs)).tolist()
    fracs_unique = (fracs_sum / len(imgs)).tolist()

    errs_out = [errs_unique[pos_by_k[k]] for k in k_list]
    fracs_out = [fracs_unique[pos_by_k[k]] for k in k_list]
    return errs_out, fracs_out

