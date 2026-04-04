import pytest
from evaluate import compute_curve_fidelity

def test_compute_curve_fidelity():
    gt = [{"points": [[0, 1], [1, 0]]}]
    extr = [[[0, 1], [1, 0]]]
    assert compute_curve_fidelity(gt, extr) == 100.0
    
    gt2 = [{"points": [[0, 1], [1, 0]]}]
    # Error of 120 pixels should yield 0.0, wait the math is 1.0 - (dist/120.0).
    extr2 = [[[0, 121], [1, 0]]]
    assert compute_curve_fidelity(gt2, extr2) == 0.0

def test_compute_curve_fidelity_empty():
    assert compute_curve_fidelity([], []) == 0.0
