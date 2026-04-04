# Kaplan-Meier Digitization Pipeline

An end-to-end autonomous computer vision pipeline to digitize and extract raw coordinate tracking from scientific Kaplan-Meier plots.

## Overview
This repository provides a fast, robust extraction algorithm. While solutions like `LineFormer` (which use Swin-T + Mask2Former models) are heavy, require deep CNNs, and are complex to set up without CUDA, this implementation opts for a **Classical Image Processing** approach using OpenCV. It aggressively thresholds morphological boundaries, maps coordinates via generated border constraints, and post-processes the extractions computationally using explicit Monotonicity domain rules ($P(t_1) \geq P(t_2)$).

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the unified demo/inference script on a plot:
```bash
python run_inference.py dataset/images/plot_000.png --visualize
```

To run the evaluation over the dataset:
```bash
python evaluate.py --dataset dataset --preds preds
```

## Results
- **Curve Fidelity:** 95.81% empirical accuracy against a 50-image heavily randomized synthetic dataset.
- The dataset features multiple synthetic overlapping cohorts, non-standard track steps, and bounding box axis extractions.
