"""
Integration test for Agentic Pipeline vs Baseline (Phase 3).

Tests agentic pipeline performance against baseline on a batch of samples.
Validates that agentic improves or maintains performance compared to baseline.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.pipeline import AgenticPipeline
from src.baselines.llama_single import LlamaSingleBaseline
from src.utils.preprocess import load_from_jsonl
from src.evaluation.metrics import compute_all_metrics, print_metrics_report
import yaml
import torch

print("=" * 80)
print("Testing Agentic Pipeline vs Baseline (Phase 3)")
print("=" * 80)

# 1. Load configs
print("\n1. Loading configs...")
baseline_config_path = project_root / 'configs' / 'baseline.yaml'
agentic_config_path = project_root / 'configs' / 'agentic.yaml'

with open(baseline_config_path, 'r') as f:
    baseline_config = yaml.safe_load(f)
with open(agentic_config_path, 'r') as f:
    agentic_config = yaml.safe_load(f)

print(f"   Baseline model: {baseline_config['model_name']}")
print(f"   Agentic model: {agentic_config['model_name']}")

# 2. Check GPU
print("\n2. GPU check...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# 3. Load test samples from dev set
print("\n3. Loading test samples...")
data_path = project_root / 'data' / 'processed' / 'dev.jsonl'

if not data_path.exists():
    print(f"   ⚠️  No dev data found at {data_path}")
    print("   Using hardcoded samples instead...")
    test_samples = [
        {
            'id': 'test_001',
            'text': 'Patient denies cocaine use. Assessment: No evidence of substance abuse.',
            'trigger_text': 'cocaine',
            'status_label': 'none'
        },
        {
            'id': 'test_002',
            'text': 'Patient reports daily cocaine use. Assessment: Active substance abuse. Plan: Refer to addiction services.',
            'trigger_text': 'cocaine',
            'status_label': 'current'
        },
        {
            'id': 'test_003',
            'text': 'Patient has history of cocaine use but has been clean for 2 years. Assessment: Past substance abuse, currently in remission.',
            'trigger_text': 'cocaine',
            'status_label': 'past'
        }
    ]
    print(f"   Using {len(test_samples)} hardcoded samples")
else:
    all_samples = load_from_jsonl(data_path)
    # Take 30 samples (average batch size)
    test_samples = all_samples[:30]
    print(f"   Loaded {len(test_samples)} samples from {data_path}")

# 4. Initialize baseline
print("\n4. Initializing baseline...")
baseline = LlamaSingleBaseline(baseline_config)
baseline.load_model()
print("   ✅ Baseline model loaded")

# 5. Run baseline inference
print("\n5. Running baseline inference...")
baseline_results = baseline.predict_batch(test_samples, show_progress=True)
print(f"   ✅ Baseline completed on {len(baseline_results)} samples")

# 6. Compute baseline metrics
print("\n6. Computing baseline metrics...")
baseline_y_true = [r.get('status_label', 'Not Applicable') for r in baseline_results]
baseline_y_pred = [r.get('pred_label', 'Not Applicable') for r in baseline_results]
base_labels = {'none', 'current', 'past', 'Not Applicable'}
labels = sorted(base_labels | set(baseline_y_true))
baseline_metrics = compute_all_metrics(baseline_y_true, baseline_y_pred, labels=labels)

print("   Baseline Metrics:")
print(f"      Accuracy: {baseline_metrics['accuracy']:.2%}")
print(f"      FPR: {baseline_metrics['fpr']:.2%}")
print(f"      Per-class F1: {baseline_metrics.get('per_class_f1', {})}")

# 7. Initialize agentic pipeline
print("\n7. Initializing agentic pipeline...")
agentic_pipeline = AgenticPipeline(agentic_config)
agentic_pipeline.setup_cache()
agentic_pipeline.load_agents()
print("   ✅ Agentic pipeline loaded")

# 8. Run agentic pipeline
print("\n8. Running agentic pipeline...")
agentic_results = agentic_pipeline.run_pipeline(test_samples)
print(f"   ✅ Agentic pipeline completed on {len(agentic_results)} samples")

# 9. Compute agentic metrics
print("\n9. Computing agentic metrics...")
agentic_y_true = [r.get('status_label', 'Not Applicable') for r in agentic_results]
agentic_y_pred = [r.get('final_label', 'Not Applicable') for r in agentic_results]
agentic_metrics = compute_all_metrics(agentic_y_true, agentic_y_pred, labels=labels)

print("   Agentic Metrics:")
print(f"      Accuracy: {agentic_metrics['accuracy']:.2%}")
print(f"      FPR: {agentic_metrics['fpr']:.2%}")
print(f"      Per-class F1: {agentic_metrics.get('per_class_f1', {})}")

# 10. Compare metrics
print("\n10. Comparing metrics...")
print("=" * 80)
print("COMPARISON: Baseline vs Agentic")
print("=" * 80)

# Accuracy comparison
acc_diff = agentic_metrics['accuracy'] - baseline_metrics['accuracy']
acc_pct_change = (acc_diff / baseline_metrics['accuracy'] * 100) if baseline_metrics['accuracy'] > 0 else 0
print(f"\nAccuracy:")
print(f"   Baseline: {baseline_metrics['accuracy']:.2%}")
print(f"   Agentic:  {agentic_metrics['accuracy']:.2%}")
print(f"   Difference: {acc_diff:+.2%} ({acc_pct_change:+.1f}%)")

# FPR comparison (primary metric)
fpr_diff = agentic_metrics['fpr'] - baseline_metrics['fpr']
fpr_pct_change = (fpr_diff / baseline_metrics['fpr'] * 100) if baseline_metrics['fpr'] > 0 else 0
print(f"\nFPR (False Positive Rate):")
print(f"   Baseline: {baseline_metrics['fpr']:.2%}")
print(f"   Agentic:  {agentic_metrics['fpr']:.2%}")
print(f"   Difference: {fpr_diff:+.2%} ({fpr_pct_change:+.1f}%)")

# 11. Validation assertions
print("\n11. Validating performance...")
all_valid = True

# Check accuracy is within acceptable range (within 5% of baseline)
acc_tolerance = 0.05
if abs(acc_diff) > acc_tolerance:
    print(f"   ⚠️  Accuracy difference ({acc_diff:.2%}) exceeds tolerance ({acc_tolerance:.2%})")
    # This is a warning, not a failure - agentic might trade accuracy for better FPR
else:
    print(f"   ✅ Accuracy within tolerance ({acc_diff:.2%} <= {acc_tolerance:.2%})")

# Check FPR improvement (primary goal - should be lower)
if fpr_diff > 0:
    print(f"   ❌ FPR increased by {fpr_diff:.2%} (agentic should improve FPR)")
    all_valid = False
elif fpr_diff < -0.01:  # At least 1% improvement
    print(f"   ✅ FPR improved by {abs(fpr_diff):.2%}")
elif fpr_diff < 0:
    print(f"   ✅ FPR improved by {abs(fpr_diff):.2%} (marginal but positive)")
else:
    print(f"   ⚠️  FPR unchanged (agentic should ideally improve FPR)")

# Check that both methods have valid predictions
baseline_valid = all(pred in labels for pred in baseline_y_pred)
agentic_valid = all(pred in labels for pred in agentic_y_pred)

if not baseline_valid:
    print("   ❌ Baseline has invalid predictions")
    all_valid = False
else:
    print("   ✅ Baseline predictions are valid")

if not agentic_valid:
    print("   ❌ Agentic has invalid predictions")
    all_valid = False
else:
    print("   ✅ Agentic predictions are valid")

# 12. Sample comparison
print("\n12. Sample comparison (first 5 samples):")
print("=" * 80)
for i in range(min(5, len(test_samples))):
    sample = test_samples[i]
    baseline_result = baseline_results[i]
    agentic_result = agentic_results[i]
    
    true_label = sample.get('status_label', 'unknown')
    baseline_pred = baseline_result.get('pred_label', 'unknown')
    agentic_pred = agentic_result.get('final_label', 'unknown')
    
    baseline_match = "✅" if baseline_pred == true_label else "❌"
    agentic_match = "✅" if agentic_pred == true_label else "❌"
    
    print(f"\nSample {i+1}:")
    print(f"   Text: {sample['text'][:80]}...")
    print(f"   Trigger: {sample['trigger_text']}")
    print(f"   Gold: {true_label}")
    print(f"   Baseline: {baseline_pred} {baseline_match}")
    print(f"   Agentic:  {agentic_pred} {agentic_match}")
    if agentic_result.get('proposer_letter'):
        print(f"   Agentic trace: Proposer={agentic_result.get('proposer_letter')}, "
              f"Refuter={agentic_result.get('refuter_letter')}, "
              f"Judge={agentic_result.get('final_choice')}")

# 13. Save results
print("\n13. Saving results...")
results_path = project_root / 'tests' / 'integration' / 'phase3_comparison_results.json'
results_data = {
    'baseline_metrics': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in baseline_metrics.items()},
    'agentic_metrics': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in agentic_metrics.items()},
    'comparison': {
        'accuracy_diff': float(acc_diff),
        'accuracy_pct_change': float(acc_pct_change),
        'fpr_diff': float(fpr_diff),
        'fpr_pct_change': float(fpr_pct_change),
        'samples_tested': len(test_samples)
    },
    'baseline_predictions': baseline_results[:10],  # Save first 10 for reference
    'agentic_predictions': agentic_results[:10]
}

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=2, default=str)
print(f"   Saved results to: {results_path}")

# 14. Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Samples tested: {len(test_samples)}")
print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.2%}, FPR: {baseline_metrics['fpr']:.2%}")
print(f"Agentic Accuracy:  {agentic_metrics['accuracy']:.2%}, FPR: {agentic_metrics['fpr']:.2%}")
print(f"FPR Improvement:   {abs(fpr_diff):.2%} ({fpr_pct_change:+.1f}%)")
print(f"Accuracy Change:   {acc_diff:+.2%} ({acc_pct_change:+.1f}%)")

if all_valid:
    print("\n✅ All validations passed!")
    print("=" * 80)
    sys.exit(0)
else:
    print("\n❌ Some validations failed!")
    print("=" * 80)
    sys.exit(1)

