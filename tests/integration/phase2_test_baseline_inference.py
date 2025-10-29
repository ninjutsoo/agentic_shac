"""Integration test for baseline inference engine"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.llama_single import LlamaSingleBaseline
from src.evaluation.metrics import compute_all_metrics, print_metrics_report
from src.utils.preprocess import load_from_jsonl
import yaml
import torch

print("="*80)
print("Testing Baseline Inference Engine")
print("="*80)

# 1. Load config
print("\n1. Loading config...")
config_path = project_root / 'configs' / 'baseline.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print(f"   Model: {config['model_name']}")
print(f"   Dtype: {config['dtype']}")

# 2. Check GPU
print("\n2. GPU check...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# 3. Initialize baseline
print("\n3. Initializing baseline...")
baseline = LlamaSingleBaseline(config)
baseline.load_model()
print(f"   ✅ Model loaded")

# 4. Test on sample data
print("\n4. Testing on sample data...")

# Load a few samples from processed data
data_path = project_root / 'data' / 'processed' / 'train.jsonl'

if data_path.exists():
    print(f"   Loading from: {data_path}")
    all_samples = load_from_jsonl(data_path)
    
    # Take first 5 samples
    samples = all_samples[:5]
    print(f"   Testing on {len(samples)} samples")
    
    # Run inference
    print("\n   Running inference...")
    results = baseline.predict_batch(samples, show_progress=True)
    
    # Display results
    print("\n5. Results:")
    for i, result in enumerate(results, 1):
        true_label = result.get('status_label', 'unknown')
        pred_label = result.get('pred_label', 'unknown')
        match = "✅" if true_label == pred_label else "❌"
        
        print(f"\n   Sample {i}:")
        print(f"      Text: {result['text'][:80]}...")
        print(f"      Trigger: {result['trigger_text']}")
        print(f"      True: {true_label}")
        print(f"      Pred: {pred_label} ({result['pred_letter']})")
        print(f"      {match}")
    
    # Compute metrics
    print("\n6. Metrics on sample:")
    y_true = [r.get('status_label', 'unknown') for r in results]
    y_pred = [r['pred_label'] for r in results]
    
    labels = ['none', 'current', 'past', 'Not Applicable']
    metrics = compute_all_metrics(y_true, y_pred, labels=labels)
    
    print(f"   Accuracy: {metrics['accuracy']:.2%} ({sum(1 for t,p in zip(y_true, y_pred) if t==p)}/{len(y_true)})")
    print(f"   True labels: {y_true}")
    print(f"   Pred labels: {y_pred}")
    
else:
    print(f"   ⚠️  No processed data found at {data_path}")
    print(f"   Testing on hardcoded sample instead...")
    
    # Test on hardcoded sample
    text = "Social History:\nPatient denies drug use. No history of IVDU."
    trigger = "IVDU"
    
    result = baseline.predict_single(text, trigger)
    print(f"\n   Text: {text}")
    print(f"   Trigger: {trigger}")
    print(f"   Predicted: {result['pred_label']} (letter: {result['pred_letter']})")
    print(f"   Raw output (last 100 chars): {result['raw_output']}")

print("\n" + "="*80)
print("✅ Baseline inference test completed!")
print("="*80)

