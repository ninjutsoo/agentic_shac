"""
Judge Agent for Agentic Pipeline (Phase 3).

Makes final decision based on Proposer and Refuter outputs using non-cue evidence.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from tqdm import tqdm

from src.agentic.prompts import (
    get_prompt,
    format_for_llama,
    parse_judge_output,
)


class JudgeAgent:
    """Judge agent that makes final decision based on Proposer and Refuter outputs."""
    
    def __init__(self, config: Dict, model=None, tokenizer=None):
        """
        Initialize Judge agent.
        
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
        
        print(f"Loading model for Judge...")
        
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
        
        print(f"✅ Judge model loaded successfully")
    
    def apply_decision_rules(self, proposer_letter: Optional[str], refuter_letter: Optional[str],
                            refuter_spans: List[str], proposer_masked_letter: Optional[str],
                            non_cue_text: str) -> str:
        """
        Apply decision rules to determine final choice.
        
        Rules:
        1) Prefer the letter supported by Refuter spans.
        2) If spans give no support, choose (a) none.
        3) If Proposer flips on masked note and Refuter stays stable, prefer Refuter.
        
        Args:
            proposer_letter: Proposer's letter on full note
            refuter_letter: Refuter's letter
            refuter_spans: Refuter's evidence spans
            proposer_masked_letter: Proposer's letter on masked note
            non_cue_text: Non-cue text
            
        Returns:
            Final letter choice (a/b/c/d)
        """
        # Check if we have valid spans
        has_valid_spans = any(span and span.strip() for span in refuter_spans)
        non_cue_lower = non_cue_text.lower() if non_cue_text else ""
        
        # Rule 1: Prefer the letter supported by Refuter spans (if spans exist and are valid)
        if refuter_letter and refuter_letter in ['a', 'b', 'c', 'd'] and has_valid_spans:
            # Check if spans actually exist in non_cue_text
            if any(span.lower() in non_cue_lower for span in refuter_spans if span.strip()):
                return refuter_letter
        
        # Rule 3: If Proposer flips on masked note and Refuter stays stable, prefer Refuter
        # "Refuter stays stable" means Refuter didn't agree with the flipped proposer
        if (proposer_letter and proposer_masked_letter and 
            proposer_letter != proposer_masked_letter and
            refuter_letter and refuter_letter in ['a', 'b', 'c', 'd'] and
            refuter_letter != proposer_masked_letter):
            # Proposer flipped, Refuter stable (didn't follow the flip) -> prefer Refuter
            return refuter_letter
        
        # Fallback to proposer if available (proposer is primary decision maker)
        if proposer_letter and proposer_letter in ['a', 'b', 'c', 'd']:
            return proposer_letter
        
        # Fallback to refuter letter if valid (but only if proposer unavailable)
        if refuter_letter and refuter_letter in ['a', 'b', 'c', 'd']:
            return refuter_letter
        
        # Rule 2: If spans give no support AND we have no other evidence, choose (a) none
        # Only default to "a" if we truly have no evidence at all
        return "a"
    
    def predict_single(self, proposer_letter: Optional[str], refuter_letter: Optional[str],
                      refuter_spans: List[str], non_cue_text: str,
                      proposer_masked_letter: Optional[str], use_rules: bool = True) -> Dict:
        """
        Run Judge on a single sample.
        
        Args:
            proposer_letter: Proposer's letter on full note
            refuter_letter: Refuter's letter
            refuter_spans: Refuter's evidence spans
            non_cue_text: Non-cue text
            proposer_masked_letter: Proposer's letter on masked note
            use_rules: Whether to apply decision rules first (default: True)
            
        Returns:
            Dictionary with final_choice and reason
        """
        # Apply decision rules first if enabled
        if use_rules:
            rule_choice = self.apply_decision_rules(
                proposer_letter, refuter_letter, refuter_spans,
                proposer_masked_letter, non_cue_text
            )
            # If rules give a clear answer, use it
            if rule_choice and rule_choice in ['a', 'b', 'c', 'd']:
                return {
                    'final_choice': rule_choice,
                    'reason': 'decision_rules'
                }
        
        # Fallback to LLM if rules don't apply or use_rules=False
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format refuter spans as string
        spans_str = "\n".join([f"- \"{span}\"" for span in refuter_spans]) if refuter_spans else "- \"\""
        
        # Get prompt template
        prompt_template = self.config.get('prompts', {}).get('judge', 'judge_v1')
        
        # Format prompt
        prompt_dict = get_prompt(
            prompt_template,
            proposer_letter=proposer_letter or "none",
            refuter_letter=refuter_letter or "none",
            refuter_spans=spans_str,
            non_cue_text=non_cue_text,
            proposer_masked_letter=proposer_masked_letter or "none"
        )
        prompt = format_for_llama(prompt_dict['system'], prompt_dict['user'])
        
        # Generate
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
        
        # Parse output
        final_choice = parse_judge_output(generated_text)
        
        # Fallback to rules if parsing fails
        if not final_choice or final_choice not in ['a', 'b', 'c', 'd']:
            final_choice = self.apply_decision_rules(
                proposer_letter, refuter_letter, refuter_spans,
                proposer_masked_letter, non_cue_text
            )
            reason = 'decision_rules_fallback'
        else:
            reason = 'llm_output'
        
        return {
            'final_choice': final_choice,
            'reason': reason
        }
    
    def predict_batch(self, samples: List[Dict], show_progress: bool = True, use_rules: bool = True) -> List[Dict]:
        """
        Run Judge on a batch of samples.
        
        Args:
            samples: List of dicts with proposer/refuter outputs
            show_progress: Whether to show progress bar
            use_rules: Whether to apply decision rules first
            
        Returns:
            List of predictions with added 'final_choice' and 'reason' keys
        """
        results = []
        iterator = tqdm(samples, desc="Judge") if show_progress else samples
        
        for sample in iterator:
            proposer_letter = sample.get('proposer_letter')
            refuter_letter = sample.get('refuter_letter')
            refuter_spans = sample.get('refuter_spans', [])
            non_cue_text = sample.get('non_cue_text', '')
            proposer_masked_letter = sample.get('proposer_masked_letter')
            
            prediction = self.predict_single(
                proposer_letter, refuter_letter, refuter_spans,
                non_cue_text, proposer_masked_letter, use_rules=use_rules
            )
            
            # Add to results
            result = sample.copy()
            result['final_choice'] = prediction['final_choice']
            result['reason'] = prediction['reason']
            results.append(result)
        
        return results

