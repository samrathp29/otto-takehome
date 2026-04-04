import json
import argparse
import os
import sys
import glob
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def compute_curve_fidelity(gt_curves, pred_curves):
    """Compute curve fidelity between ground truth and predicted curves."""
    if not gt_curves or not pred_curves:
        return 0.0

    total_error = 0.0
    valid_curves = 0

    for gt in gt_curves:
        gt_points = np.array(gt["points"])
        best_error = float('inf')

        for pred in pred_curves:
            pred_points = np.array(pred)
            if len(pred_points) == 0:
                continue
            dist = max(
                directed_hausdorff(gt_points, pred_points)[0],
                directed_hausdorff(pred_points, gt_points)[0]
            )
            if dist < best_error:
                best_error = dist

        if best_error != float('inf'):
            total_error += best_error
            valid_curves += 1

    if valid_curves == 0:
        return 0.0

    avg_error = total_error / valid_curves
    fidelity = max(0.0, 1.0 - (avg_error / 120.0)) * 100.0
    return fidelity


def evaluate_dataset(dataset_dir, preds_dir):
    """Evaluate curve fidelity across an entire dataset."""
    gt_files = sorted(glob.glob(f"{dataset_dir}/ground_truth/*.json"))
    fidelities = []

    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        pred_file = os.path.join(preds_dir, base)

        with open(gt_file, 'r') as f:
            gt_data = json.load(f)

        if not os.path.exists(pred_file):
            fidelities.append(0.0)
            continue

        with open(pred_file, 'r') as f:
            pred_data = json.load(f)

        gt_curves = gt_data["curves"]
        pred_curves = pred_data.get("curves", [])
        fid = compute_curve_fidelity(gt_curves, pred_curves)
        fidelities.append(fid)

    if not fidelities:
        return 0.0
    return sum(fidelities) / len(fidelities)


def mode_segmentation_count():
    """Count successfully segmented images without exception."""
    from app import process_all
    success, total = process_all()
    print(f"Segmentation: {success}/{total} images processed without exception.")
    return success


def mode_axis_accuracy():
    """Measure axis detection accuracy against ground truth."""
    from axis_extraction import extract_axis_limits

    gt_files = sorted(glob.glob("dataset/ground_truth/*.json"))
    correct = 0
    total = 0

    for gt_file in gt_files:
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        img = gt_file.replace("ground_truth", "images").replace(".json", ".png")
        axes = extract_axis_limits(img)
        if axes is None:
            total += 1
            continue

        gt_max = gt["axes"][1]
        det_max = axes["x_max_val"]
        if abs(gt_max - det_max) < 5:  # Within 5 units = correct
            correct += 1
        total += 1

    pct = (correct / total * 100) if total > 0 else 0
    print(f"Axis extraction accuracy: {pct:.0f}%")
    return pct


def mode_check_crashes():
    """Verify no crashes occur across the pipeline."""
    from app import process_all
    try:
        success, total = process_all()
        if success == total:
            print("No crashes detected.")
        else:
            print(f"WARNING: {total - success} images crashed.")
    except Exception as e:
        print(f"CRASH: {e}")


def mode_full_fidelity(dataset_dir="dataset", preds_dir="preds"):
    """Run full fidelity evaluation."""
    fidelity = evaluate_dataset(dataset_dir, preds_dir)
    print(f"Curve Fidelity: {fidelity:.2f}%")
    return fidelity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KM Pipeline Evaluation")
    parser.add_argument('--mode', type=str, default='full_fidelity',
                        choices=['segmentation_count', 'axis_accuracy', 'check_crashes', 'full_fidelity'],
                        help='Evaluation mode')
    parser.add_argument('--dataset', type=str, default='default',
                        choices=['default', 'adversarial'],
                        help='Dataset to evaluate against')
    parser.add_argument('--preds', type=str, default='preds',
                        help='Predictions directory')
    args = parser.parse_args()

    # Resolve dataset paths
    if args.dataset == "adversarial":
        dataset_dir = "data/adversarial"
        preds_dir = "preds_adversarial"
    else:
        dataset_dir = "dataset"
        preds_dir = args.preds

    if args.mode == "segmentation_count":
        mode_segmentation_count()
    elif args.mode == "axis_accuracy":
        mode_axis_accuracy()
    elif args.mode == "check_crashes":
        mode_check_crashes()
    elif args.mode == "full_fidelity":
        mode_full_fidelity(dataset_dir, preds_dir)
