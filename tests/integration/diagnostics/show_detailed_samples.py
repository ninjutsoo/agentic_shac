"""
Show detailed Proposer/Refuter/Judge outputs for problematic samples.
"""

import sys
from pathlib import Path
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.pipeline import AgenticPipeline
from src.utils.preprocess import load_from_jsonl
from src.agentic.prompts import get_prompt, format_for_llama, letter_to_label

print("=" * 80)
print("DETAILED AGENTIC PIPELINE DIAGNOSTIC")
print("=" * 80)

# Load config
config_path = project_root / 'configs' / 'agentic.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load problematic samples
data_path = project_root / 'data' / 'processed' / 'dev.jsonl'
all_samples = load_from_jsonl(data_path)

# Select problematic cases
problem_samples = [
    'm_dev_0419_drug_E3',  # Gold: none, Agentic: current (false positive)
    'm_dev_0447_drug_E2',  # Gold: none, Agentic: past (false positive)
    'm_dev_0457_drug_E4',  # Gold: none, Agentic: past (false positive)
    'm_dev_0430_drug_E2',  # Gold: current, Agentic: current (correct)
    'm_dev_0470_drug_E5',  # Gold: past, Agentic: current (regression)
]

selected_samples = [s for s in all_samples if s['id'] in problem_samples]
print(f"\nSelected {len(selected_samples)} problematic samples\n")

# Initialize pipeline
pipeline = AgenticPipeline(config)
pipeline.setup_cache()
pipeline.load_agents()

# Run pipeline
results = pipeline.run_pipeline(selected_samples)

# Display detailed analysis
for i, result in enumerate(results):
    print("=" * 80)
    print(f"SAMPLE {i+1}: {result['id']}")
    print("=" * 80)
    
    print(f"\n📋 GOLD LABEL: {result['status_label']}")
    print(f"🎯 FINAL PREDICTION: {result['final_label']} (letter: {result['final_choice']})")
    print(f"   Reason: {result['reason']}")
    
    match = "✅" if result['status_label'] == result['final_label'] else "❌"
    print(f"   Match: {match}")
    
    print(f"\n📝 FULL NOTE:")
    print(f"   {result['text']}")
    
    print(f"\n🔍 TRIGGER: {result['trigger_text']}")
    
    print(f"\n📑 MASKED NOTE:")
    print(f"   {result['masked_note'] if result['masked_note'] else '(empty)'}")
    
    print(f"\n📄 NON-CUE TEXT:")
    print(f"   {result['non_cue_text']}")
    
    print(f"\n💡 PROPOSER:")
    print(f"   Full note → {result['proposer_letter']} ({letter_to_label(result['proposer_letter'])})")
    print(f"   Masked note → {result['proposer_masked_letter']} ({letter_to_label(result['proposer_masked_letter'])})")
    
    print(f"\n🛡️ REFUTER:")
    print(f"   Letter: {result['refuter_letter']} ({letter_to_label(result['refuter_letter'])})")
    print(f"   Spans: {result['refuter_spans']}")
    
    # Show refuter prompt
    refuter_prompt = get_prompt(
        config['prompts']['refuter'],
        non_cue_text=result['non_cue_text'],
        proposer_letter=result['proposer_letter'],
        proposer_label=letter_to_label(result['proposer_letter'])
    )
    print(f"\n   Refuter Prompt:")
    print(f"   System: {refuter_prompt['system']}")
    print(f"   User:\n{refuter_prompt['user']}")
    
    print(f"\n⚖️ JUDGE:")
    print(f"   Final choice: {result['final_choice']} ({result['final_label']})")
    print(f"   Reason: {result['reason']}")
    
    # Show judge prompt
    spans_str = "\n".join([f"- \"{span}\"" for span in result['refuter_spans']]) if result['refuter_spans'] else "- \"\""
    judge_prompt = get_prompt(
        config['prompts']['judge'],
        proposer_letter=result['proposer_letter'] or "none",
        refuter_letter=result['refuter_letter'] or "none",
        refuter_spans=spans_str,
        non_cue_text=result['non_cue_text'],
        proposer_masked_letter=result['proposer_masked_letter'] or "none"
    )
    print(f"\n   Judge Prompt:")
    print(f"   System: {judge_prompt['system']}")
    print(f"   User:\n{judge_prompt['user']}")
    
    print("\n" + "-" * 80 + "\n")

