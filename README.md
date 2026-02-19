# Low-rank image compression via SVD

Compress a grayscale image by keeping only the top-\(k\) singular values, then **visualize the trade-off** between:

- **reconstruction error**: mean L1 per pixel
- **energy captured**: fraction of \( \lVert M \rVert_F^2 \) explained by the top-\(k\) singular values

You pick a dataset (or bring your own image matrix), choose a rank budget, and this repo produces **ready-to-share PNGs** that make the SVD story immediately tangible.

- **Input examples**:
  - `--dataset olivetti` (real 64×64 faces; may download once)
  - `--dataset synthetic` (fully offline, deterministic)
  - `load_image("photo.jpg")` (your own image; converted to grayscale)

## Example output

![Average error + energy curves](screenshots/olivetti_curves_img33.png)

![Rank-k panel](screenshots/olivetti_rank_k_img33.png)

## Why this matters

SVD is the simplest “compression” narrative that connects linear algebra to something you can **see**:

- Increasing \(k\) reduces error, but with diminishing returns.
- “Energy captured” is a useful summary statistic, but it does **not** perfectly track perceptual quality — so this project generates both the curves **and** the side-by-side reconstructions.

## Who it’s for

- **Students**: build intuition for low-rank approximation with immediate visual feedback.
- **Engineers**: sanity-check how far low-rank compression can go before quality degrades.
- **Anyone writing a report**: generate stable figures you can embed in docs and slides.

## Demo flow (3 steps)

1. **You** choose a dataset and ranks (e.g. \(k=1..30\) for curves and \([5,10,20,30]\) for a panel).
2. **The CLI** computes rank-\(k\) reconstructions via SVD and aggregates metrics across the dataset.
3. **You get** two PNG artifacts in an output directory:
   - `<dataset>_curves_img<idx>[_timestamp].png`
   - `<dataset>_rank_k_img<idx>[_timestamp].png`

## Quickstart (fully offline)

Requirements: **Python 3.12+**

```bash
bash run.sh
```

This creates `./.venv/`, installs dependencies, runs the **offline** synthetic demo, and writes PNGs to `./output/`.

## Install (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the CLI

### Default (Olivetti faces; may download)

```bash
python svd_image_analysis.py --output-dir output
```

Notes:

- `--dataset olivetti` is the default and may download the dataset on first run.
- Filenames include a timestamp by default; use `--no-timestamp` for stable names.

### Fully offline + stable filenames (synthetic)

```bash
python svd_image_analysis.py --dataset synthetic --output-dir output --no-timestamp
```

### Generate the README screenshots

```bash
python svd_image_analysis.py --dataset olivetti --output-dir screenshots --no-timestamp
```

### Useful flags

- `--img-idx 33`: which image to use for the rank-\(k\) panel
- `--k-max 30`: maximum \(k\) for the curves (plots \(k=1..k_{\max}\))
- `--k-panel 5,10,20,30`: comma-separated ranks for the side-by-side panel
- `--n-synthetic 200`: number of synthetic images (offline dataset)
- `--seed 0`: RNG seed (offline dataset)

## Use as a small library

If you want to compress your own image file (or any 2D matrix), import the helpers in `image_compression.py`:

```python
from image_compression import load_image, svd_reconstruct, l1_error

img = load_image("photo.jpg")  # -> (H, W) float32 in [0, 1]
recon = svd_reconstruct(img, k=50)
print("L1 error:", l1_error(img, recon))
```

Tip: `load_image()` converts to grayscale. For color images, apply SVD per-channel.

## How it works (high level)

- **Reconstruction**: \(A_k = U_k \Sigma_k V_k^\top\) (the best rank-\(k\) approximation in a least-squares sense)
- **Error metric**: mean L1 difference \( \frac{1}{mn} \lVert M - A_k \rVert_1 \)
- **Energy captured**: \( \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2} \)

## Production-oriented details

- **Headless rendering**: the CLI uses a non-GUI Matplotlib backend and only writes files (safe for servers/CI).
- **Deterministic runs**: use `--dataset synthetic --seed 0 --no-timestamp`.
- **Guardrails**: arguments are validated (rank bounds, image index bounds), and the output directory is created if missing.

## Limitations

- The CLI runs on built-in datasets (`olivetti`, `synthetic`). For arbitrary images, use the importable helpers.
- `load_image()` converts to grayscale; for color images, apply SVD per-channel.
- Dense SVD is not optimized for very large images; consider truncated/randomized SVD if you scale this up.

### Architecture diagram

<img src="docs/architecture.svg" alt="Architecture diagram" width="900" />

## Storage intuition

For an \(m \times n\) image:

- **Full storage**: \(mn\) values
- **Rank-\(k\) SVD** (approx.): \(k(m+n+1)\) values (U, \(\Sigma\), V)

Low-rank storage is smaller than the original only when \(k < \frac{mn}{m+n+1}\).

## Project structure

```
.
├─ image_compression.py      # importable helpers
├─ svd_image_analysis.py     # CLI runner (writes PNGs)
├─ run.sh                    # offline quickstart
├─ requirements.txt
├─ screenshots/              # curated images embedded in README
└─ output/                   # local/generated outputs (gitignored)
```

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
