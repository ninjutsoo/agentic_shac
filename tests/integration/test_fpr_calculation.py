"""Test FPR calculation matches paper definition"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import compute_fpr

print("="*80)
print("Testing FPR Calculation (Paper Definition)")
print("="*80)

print("\nPaper Definition:")
print("  Negative class: 'none' or 'Not Applicable' (no drug use)")
print("  Positive class: 'current' or 'past' (drug use)")
print("  FPR = FP / (FP + TN)")
print("  FP = predicted positive when truth is negative")
print("  TN = predicted negative when truth is negative")
print("  Computed ONLY on negative ground truth samples\n")

# Test Case 1: Perfect predictions on negative samples
print("\n" + "="*80)
print("Test 1: Perfect predictions on negative samples")
print("="*80)
y_true = ['none', 'none', 'none', 'Not Applicable', 'Not Applicable']
y_pred = ['none', 'none', 'none', 'Not Applicable', 'Not Applicable']

print(f"Ground truth: {y_true}")
print(f"Predictions:  {y_pred}")
print(f"  → All negative samples correctly predicted as negative")

fpr = compute_fpr(y_true, y_pred)
print(f"\nFPR = {fpr:.4f}")
print(f"Expected: 0.0000 (no false positives)")
assert fpr == 0.0, f"Expected FPR=0, got {fpr}"
print("✅ PASS")

# Test Case 2: All false positives on negative samples
print("\n" + "="*80)
print("Test 2: All false positives on negative samples")
print("="*80)
y_true = ['none', 'none', 'none', 'Not Applicable']
y_pred = ['current', 'past', 'current', 'past']

print(f"Ground truth: {y_true}")
print(f"Predictions:  {y_pred}")
print(f"  → All negative samples incorrectly predicted as positive")

fpr = compute_fpr(y_true, y_pred)
print(f"\nFPR = {fpr:.4f}")
print(f"Expected: 1.0000 (all false positives)")
assert fpr == 1.0, f"Expected FPR=1, got {fpr}"
print("✅ PASS")

# Test Case 3: Mixed - 50% FPR
print("\n" + "="*80)
print("Test 3: Mixed predictions - 50% FPR")
print("="*80)
y_true = ['none', 'none', 'none', 'none']
y_pred = ['none', 'none', 'current', 'past']

print(f"Ground truth: {y_true}")
print(f"Predictions:  {y_pred}")
print(f"  → 2/4 negative samples predicted as positive (FP=2)")
print(f"  → 2/4 negative samples predicted as negative (TN=2)")

fpr = compute_fpr(y_true, y_pred)
print(f"\nFPR = FP/(FP+TN) = 2/(2+2) = {fpr:.4f}")
print(f"Expected: 0.5000")
assert abs(fpr - 0.5) < 0.001, f"Expected FPR=0.5, got {fpr}"
print("✅ PASS")

# Test Case 4: Mixed with positive samples (ignored in FPR)
print("\n" + "="*80)
print("Test 4: Mixed with positive ground truth (should be ignored)")
print("="*80)
y_true = ['none', 'none', 'current', 'past', 'current']
y_pred = ['current', 'none', 'current', 'past', 'none']

print(f"Ground truth: {y_true}")
print(f"Predictions:  {y_pred}")
print(f"  → Negative samples: none, none")
print(f"  → Positive samples: current, past, current (IGNORED in FPR)")

fpr = compute_fpr(y_true, y_pred)
print(f"\nAnalysis of negative samples only:")
print(f"  Sample 1: truth='none', pred='current' → FP")
print(f"  Sample 2: truth='none', pred='none' → TN")
print(f"  FPR = FP/(FP+TN) = 1/(1+1) = {fpr:.4f}")
print(f"Expected: 0.5000")
assert abs(fpr - 0.5) < 0.001, f"Expected FPR=0.5, got {fpr}"
print("✅ PASS")

# Test Case 5: Real-world scenario
print("\n" + "="*80)
print("Test 5: Real-world scenario from baseline test")
print("="*80)
y_true = ['none', 'none', 'none', 'none', 'current']
y_pred = ['current', 'past', 'current', 'current', 'current']

print(f"Ground truth: {y_true}")
print(f"Predictions:  {y_pred}")
print(f"  → 4 negative samples (none)")
print(f"  → 1 positive sample (current)")

fpr = compute_fpr(y_true, y_pred)
print(f"\nAnalysis of negative samples (4 total):")
print(f"  All 4 'none' samples predicted as positive (current/past)")
print(f"  FP = 4, TN = 0")
print(f"  FPR = 4/(4+0) = {fpr:.4f}")
print(f"Expected: 1.0000 (all negative samples misclassified)")
assert fpr == 1.0, f"Expected FPR=1.0, got {fpr}"
print("✅ PASS")

print("\n" + "="*80)
print("✅ ALL FPR TESTS PASSED!")
print("="*80)
print("\nFPR calculation correctly implements paper definition:")
print("  ✓ Only considers negative ground truth samples")
print("  ✓ FP = predicted positive when truth is negative")
print("  ✓ TN = predicted negative when truth is negative")
print("  ✓ FPR = FP / (FP + TN)")
print("="*80)

