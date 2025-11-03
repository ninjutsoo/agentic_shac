"""
Baseline inference engine using Llama-3.1-8B-Instruct.

Single-model baseline for Drug StatusTime classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import re
from tqdm import tqdm
from src.agentic.prompts import (
    get_prompt,
    format_for_llama,
    parse_model_output as parse_output_from_prompts,
    letter_to_label as letter_to_label_from_prompts,
)


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
        
    def format_prompt(self, note: str) -> str:
        """
        Format prompt using Llama-3.1-Instruct format.
        
        Args:
            note: Clinical note text (trigger is implicit in the text)
            
        Returns:
            Formatted prompt string
        """
        # Use shared prompt templates as specified by roadmap/config
        template_name = self.config.get('prompt_template', 'status_v1')
        prompt_dict = get_prompt(template_name, note=note)
        return format_for_llama(prompt_dict['system'], prompt_dict['user'])
    
    def parse_output(self, generated_text: str) -> str:
        """
        Parse model output to extract letter choice.
        
        Args:
            generated_text: Raw model output
            
        Returns:
            Parsed letter (a/b/c/d) or None
        """
        # Delegate to shared parser for consistency
        return parse_output_from_prompts(generated_text)
    
    def letter_to_label(self, letter: str) -> str:
        """
        Convert letter to status label.
        
        Args:
            letter: Letter choice (a/b/c/d)
            
        Returns:
            Status label
        """
        return letter_to_label_from_prompts(letter)
    
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
            # Format prompt (trigger is implicit in the text)
            prompt = self.format_prompt(sample['text'])
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 8),
                    temperature=self.config.get('temperature', 0.0),
                    top_p=self.config.get('top_p', 1.0),
                    do_sample=self.config.get('temperature', 0.0) > 0.0
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
    
    def predict_single(self, text: str, trigger: str = None) -> Dict:
        """
        Run inference on a single sample.
        
        Args:
            text: Clinical note text (trigger is implicit in the text)
            trigger: Drug trigger word (not used in prompts, kept for compatibility)
            
        Returns:
            Prediction dict with letter and label
        """
        sample = {'text': text, 'trigger_text': trigger or ''}
        results = self.predict_batch([sample], show_progress=False)
        return results[0]

