"""
Evaluation metrics for Drug StatusTime classification.

Focus on False Positive Rate (FPR) as primary metric.
"""

from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """
    Compute confusion matrix for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names in order
        
    Returns:
        Dict with confusion matrix and per-class counts
    """
    # Initialize confusion matrix
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    
    # Fill confusion matrix
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx.get(true, -1)
        pred_idx = label_to_idx.get(pred, -1)
        if true_idx >= 0 and pred_idx >= 0:
            matrix[true_idx, pred_idx] += 1
    
    return {
        'matrix': matrix,
        'labels': labels,
        'label_to_idx': label_to_idx
    }


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute overall accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy (0-1)
    """
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def compute_fpr(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute False Positive Rate (FPR) as defined in the paper.
    
    FPR = FP / (FP + TN)
    
    Per paper definition:
    - Negative class (no drug use): "none" or "Not Applicable"
    - Positive class (drug use): "current" or "past"
    - FPR computed ONLY on samples where ground truth is negative
    
    FP = predicted positive (current/past) when truth is negative (none/Not Applicable)
    TN = predicted negative (none/Not Applicable) when truth is negative (none/Not Applicable)
    
    This measures how often we incorrectly claim drug use when patient has none.
    Critical for safety: minimize false alarms about drug use.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        FPR value (0-1), or 0.0 if no negative samples
    """
    # Define negative and positive classes
    negative_classes = {'none', 'Not Applicable'}
    positive_classes = {'current', 'past'}
    
    fp = 0  # False Positives: predicted positive when truth is negative
    tn = 0  # True Negatives: predicted negative when truth is negative
    
    for true, pred in zip(y_true, y_pred):
        # Only consider samples where ground truth is negative
        if true in negative_classes:
            if pred in positive_classes:
                fp += 1  # Incorrectly predicted drug use
            elif pred in negative_classes:
                tn += 1  # Correctly predicted no drug use
    
    denominator = fp + tn
    return fp / denominator if denominator > 0 else 0.0


def compute_per_class_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """
    Compute precision, recall, F1 for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Dict mapping label to metrics dict
    """
    metrics = {}
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in y_true if t == label),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return metrics


def compute_all_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict:
    """
    Compute all metrics at once.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names (auto-detected if None)
        
    Returns:
        Dict with all metrics
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    # Overall metrics
    accuracy = compute_accuracy(y_true, y_pred)
    fpr = compute_fpr(y_true, y_pred)
    
    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred, labels)
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    
    # Label distribution
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    
    return {
        'accuracy': accuracy,
        'fpr': fpr,
        'per_class': per_class,
        'confusion_matrix': cm,
        'true_distribution': dict(true_dist),
        'pred_distribution': dict(pred_dist),
        'n_samples': len(y_true),
        'labels': labels
    }


def print_metrics_report(metrics: Dict):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Metrics dict from compute_all_metrics()
    """
    print("="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  FPR (False Positive Rate): {metrics['fpr']:.4f}")
    print(f"      → Measures: predicted drug use (current/past) when truth is no use (none/Not Applicable)")
    print(f"      → Lower is better (minimize false alarms about drug use)")
    print(f"  Total Samples: {metrics['n_samples']}")
    
    print(f"\nLabel Distribution (Ground Truth):")
    for label, count in sorted(metrics['true_distribution'].items()):
        pct = count / metrics['n_samples'] * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\nLabel Distribution (Predicted):")
    for label, count in sorted(metrics['pred_distribution'].items()):
        pct = count / metrics['n_samples'] * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Label':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print("-" * 50)
    for label in metrics['labels']:
        m = metrics['per_class'][label]
        print(f"{label:<20} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:6d}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']['matrix']
    labels = metrics['confusion_matrix']['labels']
    
    # Header
    header = "True \\ Pred".ljust(20)
    for label in labels:
        header += f"{label[:10]:>10}"
    print(header)
    print("-" * (20 + 10 * len(labels)))
    
    # Rows
    for i, true_label in enumerate(labels):
        row = true_label[:20].ljust(20)
        for j in range(len(labels)):
            row += f"{cm[i, j]:10d}"
        print(row)
    
    print("="*80)

