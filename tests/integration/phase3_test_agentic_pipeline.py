"""
Integration test for Agentic Pipeline (Phase 3).

Tests full pipeline with Proposer, Refuter, and Judge agents.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.pipeline import AgenticPipeline
import yaml
import torch

print("=" * 80)
print("Testing Agentic Pipeline (Phase 3)")
print("=" * 80)

# 1. Load config
print("\n1. Loading config...")
config_path = project_root / 'configs' / 'agentic.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print(f"   Model: {config['model_name']}")
print(f"   Dtype: {config['dtype']}")
print(f"   Prompts: {config['prompts']}")

# 2. Check GPU
print("\n2. GPU check...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# 3. Initialize pipeline
print("\n3. Initializing pipeline...")
pipeline = AgenticPipeline(config)
pipeline.setup_cache()
pipeline.load_agents()
print(f"   ✅ All agents loaded")

# 4. Create test samples (3 tiny notes covering each class)
print("\n4. Creating test samples...")

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

print(f"   Created {len(test_samples)} test samples")
for i, sample in enumerate(test_samples, 1):
    print(f"      Sample {i}: {sample['status_label']}")

# 5. Run pipeline
print("\n5. Running pipeline...")
results = pipeline.run_pipeline(test_samples)

# 6. Validate results
print("\n6. Validating results...")
valid_letters = ['a', 'b', 'c', 'd']
all_valid = True

for i, result in enumerate(results, 1):
    final_choice = result.get('final_choice')
    proposer_letter = result.get('proposer_letter')
    refuter_letter = result.get('refuter_letter')
    final_label = result.get('final_label')
    
    # Check if final_choice is valid
    if final_choice not in valid_letters:
        print(f"   ❌ Sample {i}: Invalid final_choice '{final_choice}'")
        all_valid = False
    else:
        print(f"   ✅ Sample {i}: final_choice='{final_choice}' (label={final_label})")
        print(f"      Proposer: {proposer_letter}, Refuter: {refuter_letter}")
        print(f"      Refuter spans: {result.get('refuter_spans', [])}")
        print(f"      Reason: {result.get('reason', 'unknown')}")

if all_valid:
    print("\n   ✅ All samples have valid final_choice")
else:
    print("\n   ❌ Some samples have invalid final_choice")
    sys.exit(1)

# 7. Save snapshot JSON
print("\n7. Saving snapshot JSON...")
snapshot_path = project_root / 'tests' / 'integration' / 'phase3_pipeline_snapshot.json'
snapshot_data = {
    'config': config,
    'samples': test_samples,
    'results': results
}

# Convert to JSON-serializable format
for result in snapshot_data['results']:
    # Convert any non-serializable types
    if 'sections' in result:
        result['sections'] = dict(result['sections'])

with open(snapshot_path, 'w', encoding='utf-8') as f:
    json.dump(snapshot_data, f, indent=2, default=str)

print(f"   Saved snapshot to: {snapshot_path}")

# 8. Summary
print("\n8. Summary:")
print(f"   Total samples: {len(test_samples)}")
print(f"   All valid: {all_valid}")
print(f"   Proposer outputs: {sum(1 for r in results if r.get('proposer_letter') in valid_letters)}/{len(results)}")
print(f"   Refuter outputs: {sum(1 for r in results if r.get('refuter_letter') in valid_letters)}/{len(results)}")
print(f"   Judge outputs: {sum(1 for r in results if r.get('final_choice') in valid_letters)}/{len(results)}")

print("\n" + "=" * 80)
print("✅ Agentic Pipeline Test Complete")
print("=" * 80)

