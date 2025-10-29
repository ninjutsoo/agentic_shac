"""
Baseline inference engine using Llama-3.1-8B-Instruct.

Single-model baseline for Drug StatusTime classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import re
from tqdm import tqdm


class LlamaSingleBaseline:
    """Single Llama model baseline for classification."""
    
    def __init__(self, config: Dict):
        """
        Initialize baseline model.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading tokenizer from {self.config['model_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        print(f"Loading model...")
        
        # Load with appropriate dtype
        if self.config.get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            dtype = torch.bfloat16 if self.config['dtype'] == 'bf16' else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                device_map="auto",
                torch_dtype=dtype
            )
        
        print(f"✅ Model loaded successfully")
        
    def format_prompt(self, note: str, trigger: str) -> str:
        """
        Format prompt using Llama-3.1-Instruct format.
        
        Args:
            note: Clinical note text
            trigger: Drug trigger word
            
        Returns:
            Formatted prompt string
        """
        system_msg = "You classify Drug StatusTime in clinical notes."
        user_msg = f"""Note:
{note}

Drug trigger: "{trigger}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter."""
        
        # Llama-3.1-Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def parse_output(self, generated_text: str) -> str:
        """
        Parse model output to extract letter choice.
        
        Args:
            generated_text: Raw model output
            
        Returns:
            Parsed letter (a/b/c/d) or None
        """
        # Find last occurrence of (a), (b), (c), or (d)
        matches = re.findall(r'\([abcd]\)', generated_text.lower())
        if matches:
            return matches[-1][1]  # Extract letter from (x)
        
        # Fallback: find first letter
        for char in generated_text.lower():
            if char in ['a', 'b', 'c', 'd']:
                return char
        
        return None
    
    def letter_to_label(self, letter: str) -> str:
        """
        Convert letter to status label.
        
        Args:
            letter: Letter choice (a/b/c/d)
            
        Returns:
            Status label
        """
        mapping = {
            'a': 'none',
            'b': 'current',
            'c': 'past',
            'd': 'Not Applicable'
        }
        return mapping.get(letter, 'Not Applicable')
    
    def predict_batch(self, samples: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Run inference on a batch of samples.
        
        Args:
            samples: List of dicts with 'text' and 'trigger_text' keys
            show_progress: Whether to show progress bar
            
        Returns:
            List of predictions with added 'pred_letter' and 'pred_label' keys
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        iterator = tqdm(samples) if show_progress else samples
        
        for sample in iterator:
            # Format prompt
            prompt = self.format_prompt(sample['text'], sample['trigger_text'])
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 8),
                    temperature=self.config.get('temperature', 0.1),
                    top_p=self.config.get('top_p', 0.9),
                    do_sample=self.config.get('temperature', 0.1) > 0
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse output
            letter = self.parse_output(generated_text)
            label = self.letter_to_label(letter) if letter else "Not Applicable"
            
            # Add to results
            result = sample.copy()
            result['pred_letter'] = letter
            result['pred_label'] = label
            result['raw_output'] = generated_text[-100:]  # Last 100 chars
            results.append(result)
        
        return results
    
    def predict_single(self, text: str, trigger: str) -> Dict:
        """
        Run inference on a single sample.
        
        Args:
            text: Clinical note text
            trigger: Drug trigger word
            
        Returns:
            Prediction dict with letter and label
        """
        sample = {'text': text, 'trigger_text': trigger}
        results = self.predict_batch([sample], show_progress=False)
        return results[0]

