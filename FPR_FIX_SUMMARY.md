# FPR Calculation Fix - Critical Correction

## 🔴 Problem Identified

Our FPR (False Positive Rate) calculation was **completely backwards** from the paper's definition.

## 📄 Paper Definition

From the paper:
> "Our primary evaluation metric is the false positive rate (FPR), defined as FP/(FP + TN), where FP represents false positives (predicted current/past use when ground truth was none/unknown) and TN represents true negatives (correctly predicted none/unknown)."

### Key Points:
1. **Classification**: Model outputs 3 labels (none, current, past)
2. **Evaluation grouping**: Binary risk measure
   - **Negative class** (no drug use): `none` + `unknown`/`Not Applicable`
   - **Positive class** (drug use): `current` + `past`
3. **FPR computed ONLY on negative ground truth samples**

## ❌ What Was Wrong

### Old (Incorrect) Implementation:
```python
def compute_fpr(y_true, y_pred, positive_class='none'):
    """
    WRONG: Treated 'none' as the positive class
    FP = Predicted 'none' when actually 'current'/'past'
    TN = Predicted 'current'/'past' when actually 'current'/'past'
    """
    fp = 0
    tn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == positive_class and true != positive_class:
            fp += 1
        elif pred != positive_class and true != positive_class:
            tn += 1
    return fp / (fp + tn)
```

**Problem**: This calculated "how often we predict 'none' incorrectly", which is the **opposite** of what we want!

## ✅ What's Fixed

### New (Correct) Implementation:
```python
def compute_fpr(y_true, y_pred):
    """
    CORRECT: Per paper definition
    Negative class: 'none' OR 'Not Applicable'
    Positive class: 'current' OR 'past'
    
    FP = predicted positive (current/past) when truth is negative (none/Not Applicable)
    TN = predicted negative (none/Not Applicable) when truth is negative (none/Not Applicable)
    
    Computed ONLY on samples where ground truth is negative
    """
    negative_classes = {'none', 'Not Applicable'}
    positive_classes = {'current', 'past'}
    
    fp = 0
    tn = 0
    
    for true, pred in zip(y_true, y_pred):
        # Only consider samples where ground truth is negative
        if true in negative_classes:
            if pred in positive_classes:
                fp += 1  # Incorrectly predicted drug use
            elif pred in negative_classes:
                tn += 1  # Correctly predicted no drug use
    
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0
```

## 📊 Example

### Scenario:
```python
y_true = ['none', 'none', 'none', 'none', 'current']
y_pred = ['current', 'past', 'current', 'current', 'current']
```

### Old (Wrong) FPR:
- Would calculate: how often we predict 'none' when truth is not 'none'
- FP = 0 (never predicted 'none')
- TN = 1 (predicted 'current' when truth was 'current')
- FPR = 0.0 ❌ **WRONG!**

### New (Correct) FPR:
- Only looks at negative samples (first 4 samples)
- FP = 4 (predicted current/past when truth was 'none')
- TN = 0 (never predicted 'none' when truth was 'none')
- FPR = 4/4 = 1.0 ✅ **CORRECT!**

## 🧪 Testing

Created comprehensive test suite: `tests/integration/test_fpr_calculation.py`

**All 5 tests PASS:**
1. ✅ Perfect predictions on negative samples (FPR = 0.0)
2. ✅ All false positives (FPR = 1.0)
3. ✅ Mixed 50% FPR (FPR = 0.5)
4. ✅ Mixed with positive samples ignored (FPR = 0.5)
5. ✅ Real-world scenario (FPR = 1.0)

## 📝 Files Updated

1. **src/evaluation/metrics.py**
   - Rewrote `compute_fpr()` with correct logic
   - Updated docstrings
   - Updated `print_metrics_report()` to explain FPR

2. **tests/integration/test_fpr_calculation.py** (NEW)
   - Comprehensive FPR unit tests
   - Validates against known expected values

3. **roadmap.md**
   - Added detailed "Evaluation Metrics" section
   - Explains binary risk grouping
   - Documents FPR calculation step-by-step

4. **notebooks/05_test_baseline_inference.ipynb**
   - Updated validation checklist
   - Added "Understanding FPR" section with formula

## 🎯 Clinical Interpretation

**Old (Wrong) FPR**: "How often do we miss drug use?"
- Not useful for safety

**New (Correct) FPR**: "How often do we falsely claim drug use when there is none?"
- **Critical for safety**: Minimize false alarms
- Measures over-prediction of drug use
- Target: <15% for clinical deployment

## 🔍 Impact

- **All future evaluations**: Will use correct metric
- **Baseline comparison**: Must re-run with correct FPR
- **Agentic system goal**: Reduce FPR (now correctly defined)
- **Paper alignment**: Now matches paper's exact definition

---

**Status**: ✅ FIXED and TESTED

**Verified**: All FPR tests pass with expected values
