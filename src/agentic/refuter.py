"""
Refuter Agent for Agentic Pipeline (Phase 3).

Challenges Proposer decisions using non-cue text and provides evidence spans.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from tqdm import tqdm

from src.agentic.prompts import (
    get_prompt,
    format_for_llama,
    parse_refuter_output,
)


class RefuterAgent:
    """Refuter agent that challenges Proposer decisions using non-cue text."""
    
    def __init__(self, config: Dict, model=None, tokenizer=None):
        """
        Initialize Refuter agent.
        
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
        
        print(f"Loading model for Refuter...")
        
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
        
        print(f"✅ Refuter model loaded successfully")
    
    def validate_spans(self, spans: List[str], non_cue_text: str) -> List[str]:
        """
        Validate that spans exist in non_cue_text.
        
        Args:
            spans: List of span strings
            non_cue_text: Non-cue text to validate against
            
        Returns:
            List of validated spans (only those found in non_cue_text)
        """
        validated = []
        non_cue_lower = non_cue_text.lower()
        
        for span in spans:
            if not span or not span.strip():
                continue
            # Check if span (case-insensitive) exists in non_cue_text
            if span.lower() in non_cue_lower:
                validated.append(span)
        
        return validated
    
    def predict_single(self, non_cue_text: str, proposer_letter: str) -> Dict:
        """
        Run Refuter on a single sample.
        
        Args:
            non_cue_text: Non-cue text sections
            proposer_letter: Proposer's letter choice (a/b/c/d)
            
        Returns:
            Dictionary with refuter_letter and refuter_spans
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prompt template
        prompt_template = self.config.get('prompts', {}).get('refuter', 'refuter_v1')
        
        # Format prompt
        prompt_dict = get_prompt(
            prompt_template,
            non_cue_text=non_cue_text,
            proposer_letter=proposer_letter
        )
        prompt = format_for_llama(prompt_dict['system'], prompt_dict['user'])
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_new_tokens', 64),  # Longer for spans
                temperature=self.config.get('temperature', 0.1),
                top_p=self.config.get('top_p', 0.9),
                do_sample=self.config.get('temperature', 0.1) > 0.0
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse output
        refuter_letter, raw_spans = parse_refuter_output(generated_text)
        
        # Validate spans
        validated_spans = self.validate_spans(raw_spans, non_cue_text)
        
        return {
            'refuter_letter': refuter_letter,
            'refuter_spans': validated_spans
        }
    
    def predict_batch(self, samples: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Run Refuter on a batch of samples.
        
        Args:
            samples: List of dicts with 'non_cue_text' and 'proposer_letter' keys
            show_progress: Whether to show progress bar
            
        Returns:
            List of predictions with added 'refuter_letter' and 'refuter_spans' keys
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        iterator = tqdm(samples, desc="Refuter") if show_progress else samples
        
        for sample in iterator:
            non_cue_text = sample.get('non_cue_text', '')
            proposer_letter = sample.get('proposer_letter')
            
            if not proposer_letter:
                # Skip if no proposer letter
                result = sample.copy()
                result['refuter_letter'] = None
                result['refuter_spans'] = []
                results.append(result)
                continue
            
            prediction = self.predict_single(non_cue_text, proposer_letter)
            
            # Add to results
            result = sample.copy()
            result['refuter_letter'] = prediction['refuter_letter']
            result['refuter_spans'] = prediction['refuter_spans']
            results.append(result)
        
        return results

