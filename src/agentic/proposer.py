"""
Proposer Agent for Agentic Pipeline (Phase 3).

Runs proposer_v1 on full note and masked note to detect trigger dependency.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from tqdm import tqdm

from src.agentic.prompts import (
    get_prompt,
    format_for_llama,
    parse_model_output,
)


class ProposerAgent:
    """Proposer agent that classifies drug status from full and masked notes."""
    
    def __init__(self, config: Dict, model=None, tokenizer=None):
        """
        Initialize Proposer agent.
        
        Args:
            config: Configuration dictionary with model settings
            model: Shared model instance (optional, will load if not provided)
            tokenizer: Shared tokenizer instance (optional, will load if not provided)
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
    def load_model(self):
        """Load model and tokenizer (only if not already provided)."""
        if self.model is not None and self.tokenizer is not None:
            return  # Already provided
        
        print(f"Loading tokenizer from {self.config['model_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        print(f"Loading model for Proposer...")
        
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
        
        print(f"✅ Proposer model loaded successfully")
    
    def predict_single(self, note: str, trigger: str, masked_note: Optional[str] = None) -> Dict:
        """
        Run Proposer on a single sample.
        
        Args:
            note: Full clinical note text
            trigger: Drug trigger word
            masked_note: Masked note (if None, will be computed from note)
            
        Returns:
            Dictionary with proposer_letter and proposer_masked_letter
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prompt template
        prompt_template = self.config.get('prompts', {}).get('proposer', 'proposer_v1')
        
        # Run on full note
        prompt_dict = get_prompt(prompt_template, note=note, trigger=trigger)
        prompt = format_for_llama(prompt_dict['system'], prompt_dict['user'])
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_new_tokens', 8),
                temperature=self.config.get('temperature', 0.1),
                top_p=self.config.get('top_p', 0.9),
                do_sample=self.config.get('temperature', 0.1) > 0.0
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        proposer_letter = parse_model_output(generated_text)
        
        # Run on masked note
        if masked_note is None:
            masked_note = note  # Fallback if not provided
        
        prompt_dict_masked = get_prompt(prompt_template, note=masked_note, trigger=trigger)
        prompt_masked = format_for_llama(prompt_dict_masked['system'], prompt_dict_masked['user'])
        
        inputs_masked = self.tokenizer(prompt_masked, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs_masked = self.model.generate(
                **inputs_masked,
                max_new_tokens=self.config.get('max_new_tokens', 8),
                temperature=self.config.get('temperature', 0.1),
                top_p=self.config.get('top_p', 0.9),
                do_sample=self.config.get('temperature', 0.1) > 0.0
            )
        
        generated_text_masked = self.tokenizer.decode(outputs_masked[0], skip_special_tokens=True)
        proposer_masked_letter = parse_model_output(generated_text_masked)
        
        return {
            'proposer_letter': proposer_letter,
            'proposer_masked_letter': proposer_masked_letter
        }
    
    def predict_batch(self, samples: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Run Proposer on a batch of samples.
        
        Args:
            samples: List of dicts with 'text', 'trigger_text', and 'masked_note' keys
            show_progress: Whether to show progress bar
            
        Returns:
            List of predictions with added 'proposer_letter' and 'proposer_masked_letter' keys
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        iterator = tqdm(samples, desc="Proposer") if show_progress else samples
        
        for sample in iterator:
            note = sample['text']
            trigger = sample['trigger_text']
            masked_note = sample.get('masked_note', note)
            
            prediction = self.predict_single(note, trigger, masked_note)
            
            # Add to results
            result = sample.copy()
            result['proposer_letter'] = prediction['proposer_letter']
            result['proposer_masked_letter'] = prediction['proposer_masked_letter']
            results.append(result)
        
        return results

