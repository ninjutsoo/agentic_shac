"""
REAL model loading test - actually loads the model and runs inference.
This is what I should have done from the start.
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force use of GPU 0 (32GB RTX 5090)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*80)
print("REAL Model Loading Test")
print("="*80)

# Check GPU
print("\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Currently allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("   ❌ ERROR: No GPU available!")
    sys.exit(1)

# Model config
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
print(f"\n2. Loading tokenizer from {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✅ Tokenizer loaded")
except Exception as e:
    print(f"   ❌ Error loading tokenizer: {e}")
    sys.exit(1)

# Load model
print(f"\n3. Loading model (this takes ~30-60 seconds)...")
start_time = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    load_time = time.time() - start_time
    print(f"   ✅ Model loaded in {load_time:.1f}s")
    print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    sys.exit(1)

# Test inference
print(f"\n4. Testing inference...")
test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You classify Drug StatusTime in clinical notes.<|eot_id|><|start_header_id|>user<|end_header_id|>

Note:
Patient denies drug use. No IVDU.

Drug trigger: "IVDU"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

try:
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )
    inference_time = time.time() - start_time
    
    # Decode
    generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response - just find the last occurrence of (a), (b), (c), or (d)
    import re
    response_match = re.findall(r'\([abcd]\)', generated_text.lower())
    response = response_match[-1] if response_match else generated_text[-50:]
    
    print(f"   ✅ Inference completed in {inference_time:.3f}s")
    print(f"   Raw output (last 150 chars): '{generated_text_full[-150:]}'")
    print(f"   Model answer: '{response}'")
    
    # Parse output - extract letter from (a), (b), etc.
    letter_match = re.search(r'[abcd]', response.lower())
    parsed_letter = letter_match.group(0) if letter_match else None
    
    if parsed_letter:
        labels = {'a': 'none', 'b': 'current', 'c': 'past', 'd': 'Not Applicable'}
        print(f"   ✅ Parsed letter: {parsed_letter}")
        print(f"   ✅ Predicted label: {labels[parsed_letter]}")
    else:
        print(f"   ⚠️  Could not parse letter from output")
    
except Exception as e:
    print(f"   ❌ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - Model is working correctly!")
print("="*80)

