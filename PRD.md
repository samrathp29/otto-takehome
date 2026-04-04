# Comprehensive Product Requirements Document (PRD)

## 1. Executive Summary
**Objective:** Programmatically digitize Kaplan-Meier (KM) survival plots from literature using an autonomous develop-evaluate-iterate loop powered by **Autoresearch**. 
**Core Approach:** Instead of heavy deep learning models, the development methodology leverages **Classical Computer Vision** (HSV masking + morphological contours) wrapped inside guided `autoresearch` execution cycles to autonomously drive the metric (Curve Fidelity) towards $>95\%$.

## 2. Product Goals & Core Requirements
The primary goal is to achieve SOTA accuracy in mapping KM visualizations to $(Time, Survival\_Probability)$ datasets.

1. **Line Segmentation:** Differentiate interlocking survival curves into distinct spatial tracks.
2. **Post-Processing (Step Constraint):** Since KM curves are strictly monotonically decreasing step functions, raw continuous points must be mathematically coerced to pass the $P(t_1) \ge P(t_2)$ rule.
3. **Axis & Calibration:** Detect and digitize max/min axes limits to translate coordinates from pixels to clinical measurements.

## 3. Evaluation Benchmark (The Autoresearch Target)
To enable the autonomous `Modify -> Verify -> Keep/Discard` loop, we must have a reliable, mechanical evaluation function:
* **Dataset:** 500 ground-truth KM charts (a mix of synthetic templates and heavily overlapping real clinical crops).
* **Metric Script (`evaluate.py`):** Calculates symmetric Hausdorff coordinate mapping error compared to ground truth, isolating the Mean Absolute Error (MAE) relative to common bounds. 
* **Target Fidelity Score:** $>95\%$

---

## 4. Development Implementation (Autoresearch Tasks)

Development is strictly broken down into discrete step-by-step tasks designed for the `autoresearch` skill. Each task includes a `Goal`, `Scope`, `Metric` (higher or lower is better), and a `Verify` command triggering the evaluation metric so that Antigravity can autonomously bash out the iterations.

### Task 1: Environment & Baseline Harness
**Objective:** Create the evaluation script allowing downstream `autoresearch` tasks to mechanically verify themselves.
* **Goal:** Write a generic benchmark script that takes output JSON coords, compares it to ground truth, and outputs a Curve Fidelity Score.
* **Scope:** `evaluate.py`, `tests/`
* **Metric:** Pass count of test suite assertions (higher is better).
* **Verify:** `pytest tests/test_evaluation_metrics.py | grep "passed"`
* **Action:** Direct build, no autonomous loop required.

### Task 2: Classical CV Curve Extraction
**Objective:** Implement and autonomously refine a color-masked segmentation pipeline until it successfully pulls bounding boxes/lines from all sample images without crashing.
* **Goal:** Increase the number of images processed correctly by the CV-based segmentation pipeline.
* **Scope:** `src/segment_lines.py`, `src/app.py`
* **Metric:** Number of successfully segmented images without Exception (higher is better).
* **Verify:** `python evaluate.py --mode segmentation_count`
* **Iterations:** `Iterations: 10`

### Task 3: Axis OCR & Calibration Loop
**Objective:** Autonomously write and optimize OCR parsers that look for X/Y tick labels and map pixels to data coordinates.
* **Goal:** Increase accuracy of axis bounding-box and text extraction.
* **Scope:** `src/axis_extraction.py`, `requirements.txt`
* **Metric:** Axis limit identification accuracy percentage (higher is better).
* **Verify:** `python evaluate.py --mode axis_accuracy`
* **Guard:** `python evaluate.py --mode check_crashes`
* **Iterations:** `Iterations: 20`

### Task 4: Monotonic Function Conversion Layer
**Objective:** Post-process raw extracted contour dots into mathematically correct survival steps.
* **Goal:** Reduce the Curve Fidelity Error (MAE) of extracted curves vs. ground-truth data.
* **Scope:** `src/postprocess_steps.py`
* **Metric:** Curve Fidelity MAE (lower is better).
* **Verify:** `python evaluate.py --mode full_fidelity`
* **Iterations:** `Iterations: 30` (Run overnight if needed).

### Task 5: Handle Edge Cases (Adversarial Refinement)
**Objective:** Use the reasoning and bug-fixing autoresearch loops to hammer out failing edge cases.
* **Goal:** Fix parsing failures for charts that have overlapping lines, heavy shading, or omitted legends.
* **Scope:** `src/**/*.py`
* **Metric:** Success rate on the benchmark subset `data/adversarial/` (higher is better).
* **Verify:** `python evaluate.py --dataset adversarial`
* **Iterations:** `Iterations: 15`

## 5. Execution Protocol
To commence development, the user will trigger the loops sequentially. Example trigger:
```bash
/autoresearch
Goal: Reduce the Curve Fidelity Error (MAE) of extracted curves vs. ground-truth data
Scope: src/postprocess_steps.py
Metric: Curve Fidelity MAE (lower)
Verify: python evaluate.py --mode full_fidelity
Iterations: 30
```
This guarantees continuous compounding improvement measured empirically mechanics over time.
