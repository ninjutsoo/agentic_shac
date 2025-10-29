"""Quick test for prompts module"""

from src.agentic.prompts import (
    get_prompt, format_for_llama, parse_model_output, 
    letter_to_label, LABEL_TO_LETTER, LETTER_TO_LABEL
)

print("="*80)
print("Testing Prompts Module")
print("="*80)

# Test 1: Get prompt
print("\n1. Testing get_prompt()...")
sample_note = "Patient denies drug use. No history of IVDU."
sample_trigger = "IVDU"

prompt = get_prompt("status_v1", note=sample_note, trigger=sample_trigger)
print(f"   System: {prompt['system']}")
print(f"   User (first 100 chars): {prompt['user'][:100]}...")

# Test 2: Format for Llama
print("\n2. Testing format_for_llama()...")
formatted = format_for_llama(prompt['system'], prompt['user'])
print(f"   Formatted length: {len(formatted)} chars")
print(f"   Contains header tags: {' <|start_header_id|>' in formatted}")

# Test 3: Label mappings
print("\n3. Testing label mappings...")
print(f"   LABEL_TO_LETTER: {LABEL_TO_LETTER}")
print(f"   LETTER_TO_LABEL: {LETTER_TO_LABEL}")

# Test 4: Parse model output
print("\n4. Testing parse_model_output()...")
test_outputs = [
    "a",
    "The answer is b",
    "c) past",
    "Based on the note, I would say (d)",
    "invalid output xyz"
]

for output in test_outputs:
    parsed = parse_model_output(output)
    label = letter_to_label(parsed) if parsed else "PARSE_ERROR"
    print(f"   Input: '{output[:30]}...'")
    print(f"   Parsed: {parsed} → {label}")

print("\n" + "="*80)
print("✅ Prompts module test completed successfully!")
print("="*80)

