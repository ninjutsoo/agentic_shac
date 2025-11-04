"""
Comprehensive diagnostic test for Agentic Pipeline.

This test identifies why improvements are minimal:
1. Checks Refuter behavior
2. Checks Judge decision logic
3. Compares accuracy when Judge follows vs doesn't follow Proposer
4. Identifies where the pipeline is failing
"""

import sys
from pathlib import Path
import json
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.pipeline import AgenticPipeline
from src.utils.preprocess import load_from_jsonl
from src.evaluation.metrics import compute_all_metrics
import yaml

print("=" * 80)
print("COMPREHENSIVE AGENTIC PIPELINE DIAGNOSTIC")
print("=" * 80)

# 1. Load config
print("\n1. Loading config...")
config_path = project_root / 'configs' / 'agentic.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 2. Load test samples
print("\n2. Loading test samples...")
data_path = project_root / 'data' / 'processed' / 'dev.jsonl'
samples = load_from_jsonl(data_path)[:50]  # First 50 samples
print(f"   Loaded {len(samples)} samples")

# 3. Initialize pipeline
print("\n3. Initializing pipeline...")
pipeline = AgenticPipeline(config)
pipeline.setup_cache()
pipeline.load_agents()
print("   ✅ Pipeline initialized")

# 4. Run pipeline
print("\n4. Running pipeline...")
results = pipeline.run_pipeline(samples)
print(f"   ✅ Processed {len(results)} samples")

# 5. Analyze results
print("\n5. Analyzing results...")
print("=" * 80)

# Proposer analysis
proposer_letters = Counter(r.get('proposer_letter') for r in results)
proposer_correct = sum(1 for r in results if 
    (r.get('proposer_letter') == 'a' and r.get('status_label') == 'none') or
    (r.get('proposer_letter') == 'b' and r.get('status_label') == 'current') or
    (r.get('proposer_letter') == 'c' and r.get('status_label') == 'past') or
    (r.get('proposer_letter') == 'd' and r.get('status_label') == 'Not Applicable'))

print(f"\nProposer Analysis:")
print(f"  Letter distribution: {dict(proposer_letters)}")
print(f"  Accuracy: {proposer_correct}/{len(results)} ({proposer_correct/len(results)*100:.1f}%)")

# Refuter analysis
refuter_letters = Counter(r.get('refuter_letter') for r in results)
refuter_spans_count = sum(1 for r in results if len(r.get('refuter_spans', [])) > 0)
refuter_has_spans = [r for r in results if len(r.get('refuter_spans', [])) > 0]

print(f"\nRefuter Analysis:")
print(f"  Letter distribution: {dict(refuter_letters)}")
print(f"  Samples with spans: {refuter_spans_count}/{len(results)} ({refuter_spans_count/len(results)*100:.1f}%)")
if refuter_has_spans:
    print(f"  Sample spans: {refuter_has_spans[0].get('refuter_spans', [])[:2]}")

# Judge analysis
judge_choices = Counter(r.get('final_choice') for r in results)
judge_reasons = Counter(r.get('reason') for r in results)
judge_correct = sum(1 for r in results if r.get('final_label') == r.get('status_label'))

print(f"\nJudge Analysis:")
print(f"  Final choice distribution: {dict(judge_choices)}")
print(f"  Decision reasons: {dict(judge_reasons)}")
print(f"  Accuracy: {judge_correct}/{len(results)} ({judge_correct/len(results)*100:.1f}%)")

# Agreement analysis
proposer_judge_agreement = sum(1 for r in results if r.get('proposer_letter') == r.get('final_choice'))
refuter_judge_agreement = sum(1 for r in results if r.get('refuter_letter') == r.get('final_choice'))

print(f"\nAgreement Analysis:")
print(f"  Judge follows Proposer: {proposer_judge_agreement}/{len(results)} ({proposer_judge_agreement/len(results)*100:.1f}%)")
print(f"  Judge follows Refuter: {refuter_judge_agreement}/{len(results)} ({refuter_judge_agreement/len(results)*100:.1f}%)")

# Accuracy by decision path
follows_proposer = [r for r in results if r.get('proposer_letter') == r.get('final_choice')]
follows_refuter = [r for r in results if r.get('refuter_letter') == r.get('final_choice') and r.get('refuter_letter') != r.get('proposer_letter')]
uses_other = [r for r in results if r.get('proposer_letter') != r.get('final_choice') and r.get('refuter_letter') != r.get('final_choice')]

print(f"\nAccuracy by Decision Path:")
if follows_proposer:
    correct = sum(1 for r in follows_proposer if r.get('final_label') == r.get('status_label'))
    print(f"  When Judge follows Proposer: {correct}/{len(follows_proposer)} ({correct/len(follows_proposer)*100:.1f}%)")
if follows_refuter:
    correct = sum(1 for r in follows_refuter if r.get('final_label') == r.get('status_label'))
    print(f"  When Judge follows Refuter: {correct}/{len(follows_refuter)} ({correct/len(follows_refuter)*100:.1f}%)")
if uses_other:
    correct = sum(1 for r in uses_other if r.get('final_label') == r.get('status_label'))
    print(f"  When Judge uses other: {correct}/{len(uses_other)} ({correct/len(uses_other)*100:.1f}%)")

# Sample cases where Judge != Proposer
print(f"\nSample cases where Judge != Proposer (first 5):")
count = 0
for r in results:
    if r.get('proposer_letter') != r.get('final_choice') and count < 5:
        print(f"  Gold: {r.get('status_label'):<15} Proposer: {r.get('proposer_letter')} → Judge: {r.get('final_choice')}")
        print(f"    Refuter: {r.get('refuter_letter')}, Spans: {r.get('refuter_spans', [])}, Reason: {r.get('reason')}")
        count += 1

# 6. Final metrics
print("\n6. Final Metrics:")
print("=" * 80)
y_true = [r.get('status_label', 'Not Applicable') for r in results]
y_pred = [r.get('final_label', 'Not Applicable') for r in results]
labels = sorted(set(y_true) | set(y_pred))
metrics = compute_all_metrics(y_true, y_pred, labels=labels)

print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  FPR: {metrics['fpr']:.4f}")
print(f"  Per-class F1: {metrics.get('per_class_f1', {})}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)


