## Contributing

Thanks for taking the time to contribute.

### Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run tests

```bash
pytest
```

### Run the demo script

```bash
# Saves PNGs under ./output/
python svd_image_analysis.py

# Fully offline run with deterministic filenames
python svd_image_analysis.py --dataset synthetic --output-dir output --no-timestamp
```

### Guidelines

- Keep changes focused and documented.
- Add/adjust tests for any behavioral change.
- Avoid committing generated artifacts (use `output/` for local outputs; use `screenshots/` only for curated README images).

