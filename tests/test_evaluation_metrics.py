import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from evaluate import compute_curve_fidelity


def test_perfect_match():
    """Identical curves should yield 100% fidelity."""
    gt = [{"points": [[0, 1], [10, 0.8], [20, 0.5]]}]
    pred = [[[0, 1], [10, 0.8], [20, 0.5]]]
    assert compute_curve_fidelity(gt, pred) == 100.0


def test_empty_inputs():
    """Empty inputs should yield 0% fidelity."""
    assert compute_curve_fidelity([], []) == 0.0
    assert compute_curve_fidelity([{"points": [[0, 1]]}], []) == 0.0
    assert compute_curve_fidelity([], [[[0, 1]]]) == 0.0


def test_high_error():
    """Very distant predictions should yield low fidelity."""
    gt = [{"points": [[0, 0], [1, 0]]}]
    pred = [[[0, 200], [1, 200]]]  # 200 units away
    fid = compute_curve_fidelity(gt, pred)
    assert fid == 0.0  # Error of 200 >> max_error of 120


def test_moderate_error():
    """Moderate error should yield partial fidelity."""
    gt = [{"points": [[0, 1.0], [50, 0.5]]}]
    pred = [[[0, 1.0], [50, 0.6]]]  # Slightly off on y
    fid = compute_curve_fidelity(gt, pred)
    assert 90 < fid < 100  # Should be high but not perfect
