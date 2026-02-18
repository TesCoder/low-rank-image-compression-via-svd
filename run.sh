#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python svd_image_analysis.py --dataset synthetic --output-dir output --no-timestamp

echo
echo "Done. Open the PNGs in ./output/"

