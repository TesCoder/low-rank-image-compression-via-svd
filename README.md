# Low-Rank-Image-Compression-via-SVD


This tool provides image compression analysis using Singular Value Decomposition (SVD). It utilizes the Olivetti faces dataset by default, but can be adapted for any image input. The analysis demonstrates the relationship between mean L1 reconstruction error, the amount of energy captured at different values of k, and the resulting visual quality.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# or single line
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

```

## Dependencies (from `.venv`)

```
contourpy==1.3.3
cycler==0.12.1
fonttools==4.61.1
joblib==1.5.3
kiwisolver==1.4.9
matplotlib==3.10.8
numpy==2.4.2
packaging==26.0
pillow==12.1.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.0
six==1.17.0
threadpoolctl==3.6.0
```

## Usage

- Core helpers (importable):
  - **`load_image(path)`** — Load grayscale (or RGB→grayscale) as 2D float array in `[0, 1]`.
  - **`svd_reconstruct(M, k)`** — Best rank-*k* approximation of `M`.
  - **`l1_error(M, A)`** — Mean L1 reconstruction error between `M` and `A`.
  - **`compute_avg_energy_frac(k_vals, images)`** — Avg L1 error + avg energy fraction over a list of images.

- Notebook-style runner (saves PNGs, no GUI):  
  ```bash
  python svd_image_analysis.py
  ```
  - Saves curves to `output/olivetti_curves_img33_<timestamp>.png`.
  - Saves rank-k panel to `output/olivetti_rankk_img33_<timestamp>.png` (defaults: image idx 33, k = 5,10,20,30).

- Quick example (programmatic):
  ```python
  from image_compression import load_image, svd_reconstruct, l1_error
  img = load_image("photo.jpg")
  reconstructed = svd_reconstruct(img, k=50)
  print("L1 error:", l1_error(img, reconstructed))
  ```
