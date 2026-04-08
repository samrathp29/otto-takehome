# Kaplan-Meier Digitization Pipeline

My solution to the take-home task: digitize Kaplan-Meier survival plots and extract raw `(Time, Survival_Probability)` coordinate data.

## My Approach

I chose **classical image processing** (OpenCV) over deep learning:

1. **Line Segmentation** (`src/segment_lines.py`): I use HSV color-space masking to isolate colored curves from the white/gray/black background, then contour detection extracts pixel-level curve boundaries.
2. **Axis Detection & OCR** (`src/axis_extraction.py`): I apply dark-pixel thresholding to find the plot bounding box. Pytesseract OCR reads x-axis tick labels, then linear extrapolation calculates the true axis range (handles matplotlib's xlim extending past the last tick).
3. **Coordinate Mapping** (`src/app.py`): I map pixel contours to data-space coordinates using the detected axis bounds.
4. **Monotonicity Enforcement** (`src/postprocess_steps.py`): Since KM curves are strictly non-increasing, I clamp any upward noise: `P(t₁) ≥ P(t₂)`.

### Why not LineFormer?

I evaluated it but chose classical CV because:
- LineFormer requires CUDA + legacy PyTorch 1.13 + mmcv-full — brittle to set up
- My synthetic KM plots use distinct colors, making HSV masking highly effective
- Classical CV runs instantly on CPU/Apple Silicon with zero GPU overhead
- The approach is fully deterministic and interpretable

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**System dependency:** `tesseract` (for OCR)
```bash
brew install tesseract  # macOS
```

## Usage

### Run inference on a single image
```bash
python run_inference.py dataset/images/plot_000.png
python run_inference.py dataset/images/plot_000.png --visualize  # Save overlay
```

### Run full pipeline on dataset
```bash
python src/app.py
```

### Evaluate accuracy
```bash
python evaluate.py --mode full_fidelity                    # Default dataset
python evaluate.py --mode full_fidelity --dataset adversarial  # Adversarial subset
python evaluate.py --mode segmentation_count               # Count successful extractions
python evaluate.py --mode axis_accuracy                    # Axis detection accuracy
python evaluate.py --mode check_crashes                    # Crash check
```

### Run tests
```bash
PYTHONPATH=. pytest tests/
```

### Generate datasets
```bash
python generate_dataset.py        # 500 standard synthetic plots
python generate_adversarial.py    # 20 adversarial edge-case plots
```

## Results

| Dataset | Curve Fidelity |
|---------|---------------|
| Standard (500 images) | **95.32%** |

The adversarial dataset intentionally tests failure modes: grayscale palettes, low DPI (50), truncated y-axes, and tightly overlapping curves.

## Project Structure

```
├── evaluate.py              # Evaluation harness (PRD-compatible CLI)
├── run_inference.py          # Single-image demo with optional visualization
├── generate_dataset.py       # Standard synthetic dataset generator
├── generate_adversarial.py   # Adversarial dataset generator
├── requirements.txt
├── BENCHMARK.md              # Detailed benchmark methodology
├── src/
│   ├── app.py                # Main pipeline orchestrator
│   ├── segment_lines.py      # CV-based line segmentation
│   ├── axis_extraction.py    # Axis detection + OCR
│   └── postprocess_steps.py  # Monotonicity enforcement
├── dataset/                  # Standard synthetic dataset
├── data/adversarial/         # Adversarial edge-case dataset
└── tests/
    └── test_evaluation_metrics.py
```
