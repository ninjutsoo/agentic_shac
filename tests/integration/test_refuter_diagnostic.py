"""
Diagnostic test for Refuter agent - why is it always returning 'd'?

This test:
1. Tests Refuter with sample inputs
2. Checks actual LLM outputs
3. Verifies parsing logic
4. Identifies why Refuter isn't working
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.refuter import RefuterAgent
from src.agentic.prompts import get_prompt, format_for_llama, parse_refuter_output
from src.agentic.pipeline import AgenticPipeline
from src.utils.preprocess import load_from_jsonl
import yaml
import torch

print("=" * 80)
print("REFUTER DIAGNOSTIC TEST")
print("=" * 80)

# 1. Load config
print("\n1. Loading config...")
config_path = project_root / 'configs' / 'agentic.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 2. Test with sample cases
print("\n2. Testing Refuter with sample cases...")
print("-" * 80)

test_cases = [
    {
        'name': 'Case 1: Non-cue text has evidence for current',
        'non_cue_text': 'Assessment: Active substance abuse. Plan: Refer to addiction services.',
        'proposer_letter': 'a',  # Proposer said 'none'
        'expected': 'b'  # Should argue for 'current'
    },
    {
        'name': 'Case 2: Non-cue text has evidence for past',
        'non_cue_text': 'Assessment: Past substance abuse, currently in remission.',
        'proposer_letter': 'b',  # Proposer said 'current'
        'expected': 'c'  # Should argue for 'past'
    },
    {
        'name': 'Case 3: Non-cue text has no evidence',
        'non_cue_text': 'Assessment: No evidence of substance abuse.',
        'proposer_letter': 'b',  # Proposer said 'current'
        'expected': 'a'  # Should argue for 'none'
    },
]

# Initialize pipeline to get refuter
print("\n3. Initializing pipeline...")
pipeline = AgenticPipeline(config)
pipeline.setup_cache()
pipeline.load_agents()
refuter = pipeline.refuter

print("\n4. Testing Refuter predictions...")
print("-" * 80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{test_case['name']}:")
    print(f"  Non-cue text: {test_case['non_cue_text']}")
    print(f"  Proposer chose: {test_case['proposer_letter']}")
    print(f"  Expected: {test_case['expected']}")
    
    # Get prompt
    prompt_dict = get_prompt('refuter_v1', 
                            non_cue_text=test_case['non_cue_text'],
                            proposer_letter=test_case['proposer_letter'])
    prompt = format_for_llama(prompt_dict['system'], prompt_dict['user'])
    
    # Run refuter
    result = refuter.predict_single(test_case['non_cue_text'], test_case['proposer_letter'])
    
    print(f"  Actual: {result['refuter_letter']}")
    print(f"  Spans: {result['refuter_spans']}")
    
    if result['refuter_letter'] == test_case['expected']:
        print(f"  ✅ Correct")
    else:
        print(f"  ❌ Wrong (expected {test_case['expected']})")
    
    # Check raw output from result
    if 'raw_output' in result:
        raw_output = result['raw_output']
        print(f"  Raw LLM output (first 300 chars): {raw_output[:300]}")
    
    # Also check cache
    cache_key = pipeline._cache_key('refuter', 
                                   non_cue_text=test_case['non_cue_text'],
                                   proposer_letter=test_case['proposer_letter'])
    cached = pipeline._get_cache(cache_key)
    if cached:
        cached_data = json.loads(cached)
        raw_output = cached_data.get('raw_output', 'N/A')
        if raw_output and raw_output != 'N/A':
            print(f"  Cached raw output (first 300 chars): {raw_output[:300]}")

# 5. Check actual predictions from dev set
print("\n5. Checking actual Refuter predictions from dev set...")
print("-" * 80)

data_path = project_root / 'data' / 'processed' / 'dev.jsonl'
if data_path.exists():
    samples = load_from_jsonl(data_path)[:10]  # First 10 samples
    
    # Run through pipeline
    results = pipeline.run_pipeline(samples)
    
    print(f"\nAnalyzing {len(results)} samples:")
    from collections import Counter
    
    refuter_letters = Counter(r.get('refuter_letter') for r in results)
    refuter_spans_count = sum(1 for r in results if len(r.get('refuter_spans', [])) > 0)
    
    print(f"  Refuter letter distribution: {dict(refuter_letters)}")
    print(f"  Samples with spans: {refuter_spans_count}/{len(results)}")
    
    # Show sample outputs
    print(f"\nSample Refuter outputs:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. Proposer: {r.get('proposer_letter')}, Refuter: {r.get('refuter_letter')}, Spans: {r.get('refuter_spans', [])}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

