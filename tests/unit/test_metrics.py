import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import compute_all_metrics, compute_fpr


def test_metrics_include_future_label():
    y_true = ["none", "current", "past", "Not Applicable", "future"]
    y_pred = ["none", "current", "past", "none", "past"]

    metrics = compute_all_metrics(y_true, y_pred, labels=sorted(set(y_true) | {"none", "current", "past", "Not Applicable"}))
    # Should include 'future' in labels, support counted
    assert "future" in metrics["labels"]
    assert metrics["true_distribution"]["future"] == 1


def test_fpr_definition_respects_negative_set():
    # negatives: none, Not Applicable
    # positives: current, past
    y_true = ["none", "Not Applicable", "current", "past"]
    y_pred = ["past", "none", "past", "current"]  # FP on first (none->past), TN on second

    fpr = compute_fpr(y_true, y_pred)
    # FP=1, TN=1 => FPR=0.5
    assert abs(fpr - 0.5) < 1e-6


