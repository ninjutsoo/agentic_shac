"""
Pipeline Orchestrator for Agentic Pipeline (Phase 3).

Orchestrates batched runs through Proposer, Refuter, and Judge agents.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.agentic.proposer import ProposerAgent
from src.agentic.refuter import RefuterAgent
from src.agentic.judge import JudgeAgent
from src.utils.sectionizer import sectionize_note
from src.agentic.prompts import letter_to_label


class AgenticPipeline:
    """Orchestrates the full agentic pipeline (Proposer → Refuter → Judge)."""
    
    def __init__(self, config: Dict):
        """
        Initialize Agentic Pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.proposer = None
        self.refuter = None
        self.judge = None
        self.cache_dir = None
        
    def setup_cache(self, cache_dir: Optional[Path] = None):
        """
        Set up cache directory for prompts→outputs.
        
        Args:
            cache_dir: Cache directory path (default: experiments/agentic/cache/)
        """
        if cache_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            cache_dir = project_root / "experiments" / "agentic" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")
    
    def _cache_key(self, prompt_type: str, **kwargs) -> str:
        """
        Generate cache key from prompt type and kwargs.
        
        Args:
            prompt_type: Type of prompt (proposer/refuter/judge)
            **kwargs: Prompt parameters
            
        Returns:
            Cache key string
        """
        # Create deterministic string from kwargs
        key_str = f"{prompt_type}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache(self, cache_key: str) -> Optional[str]:
        """
        Get cached output if exists.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached output or None
        """
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('output')
        return None
    
    def _set_cache(self, cache_key: str, output: str):
        """
        Cache output.
        
        Args:
            cache_key: Cache key
            output: Output to cache
        """
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'output': output}, f)
    
    def load_shared_model(self):
        """
        Load shared model and tokenizer once for all agents.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading shared tokenizer from {self.config['model_name']}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        print(f"Loading shared model (will be shared across Proposer/Refuter/Judge)...")
        
        # Determine device map - explicitly use GPU 0 by default
        device_map = self.config.get('device_map', 'cuda:0')
        if device_map == 'cuda:0' or device_map is None:
            device_map = {'': 0}  # Put entire model on GPU 0
            print(f"Using GPU 0 for shared model")
        elif device_map == 'auto':
            # Check GPU memory and choose the GPU with more free memory
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus >= 2:
                    # Check free memory on both GPUs
                    free_mem = []
                    for i in range(num_gpus):
                        torch.cuda.set_device(i)
                        free_mem.append(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i))
                    best_gpu = free_mem.index(max(free_mem))
                    print(f"Selecting GPU {best_gpu} with {free_mem[best_gpu] / 1e9:.2f} GB free memory")
                    device_map = {'': best_gpu}
                else:
                    device_map = {'': 0}
            else:
                device_map = 'cpu'
        
        # Load with appropriate dtype
        if self.config.get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16
            )
        else:
            dtype = torch.bfloat16 if self.config['dtype'] == 'bf16' else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                device_map=device_map,
                torch_dtype=dtype
            )
        
        print(f"✅ Shared model loaded successfully on device: {next(model.parameters()).device}")
        return model, tokenizer
    
    def load_agents(self):
        """Load all three agents with shared model."""
        print("Loading agents with shared model...")
        
        # Load shared model once
        shared_model, shared_tokenizer = self.load_shared_model()
        
        # Initialize agents with shared model
        self.proposer = ProposerAgent(self.config, model=shared_model, tokenizer=shared_tokenizer)
        self.refuter = RefuterAgent(self.config, model=shared_model, tokenizer=shared_tokenizer)
        self.judge = JudgeAgent(self.config, model=shared_model, tokenizer=shared_tokenizer)
        
        print("✅ All agents loaded successfully (sharing model)")
    
    def validate_letter(self, letter: Optional[str]) -> bool:
        """
        Validate that a letter is valid.
        
        Args:
            letter: Letter to validate
            
        Returns:
            True if valid (a/b/c/d), False otherwise
        """
        return letter is not None and letter.lower() in ['a', 'b', 'c', 'd']
    
    def run_proposer(self, samples: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        Run Proposer agent on samples.
        
        Args:
            samples: List of sample dicts with 'text' and 'trigger_text'
            max_retries: Maximum retries on parsing failures
            
        Returns:
            List of samples with proposer outputs
        """
        results = []
        
        for sample in samples:
            note = sample['text']
            trigger = sample['trigger_text']
            masked_note = sample.get('masked_note')
            
            # Sectionize if needed
            if masked_note is None:
                sectionized = sectionize_note(
                    note, trigger,
                    use_sections=self.config.get('sectionizer', {}).get('use_sections'),
                    mask_trigger=self.config.get('sectionizer', {}).get('mask_trigger_sentence', True)
                )
                sample['masked_note'] = sectionized['masked_note']
                sample['non_cue_text'] = sectionized['non_cue_text']
                sample['sections'] = sectionized['sections']
            
            # Check cache first
            cache_key = self._cache_key('proposer', note=note, trigger=trigger, masked_note=sample['masked_note'])
            cached_output = self._get_cache(cache_key)
            
            if cached_output:
                try:
                    cached_data = json.loads(cached_output)
                    proposer_letter = cached_data.get('proposer_letter')
                    proposer_masked_letter = cached_data.get('proposer_masked_letter')
                    if self.validate_letter(proposer_letter) and self.validate_letter(proposer_masked_letter):
                        result = sample.copy()
                        result['proposer_letter'] = proposer_letter
                        result['proposer_masked_letter'] = proposer_masked_letter
                        results.append(result)
                        continue
                except:
                    pass  # Cache miss or invalid, continue to run
            
            # Run Proposer with retries
            proposer_letter = None
            proposer_masked_letter = None
            
            for attempt in range(max_retries):
                try:
                    prediction = self.proposer.predict_single(
                        note, trigger, sample['masked_note']
                    )
                    proposer_letter = prediction['proposer_letter']
                    proposer_masked_letter = prediction['proposer_masked_letter']
                    
                    # Validate letters
                    if self.validate_letter(proposer_letter) and self.validate_letter(proposer_masked_letter):
                        # Cache the result
                        self._set_cache(cache_key, json.dumps({
                            'proposer_letter': proposer_letter,
                            'proposer_masked_letter': proposer_masked_letter
                        }))
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Warning: Proposer failed after {max_retries} attempts: {e}")
            
            # Default to 'a' if parsing failed
            if not self.validate_letter(proposer_letter):
                proposer_letter = 'a'
            if not self.validate_letter(proposer_masked_letter):
                proposer_masked_letter = 'a'
            
            result = sample.copy()
            result['proposer_letter'] = proposer_letter
            result['proposer_masked_letter'] = proposer_masked_letter
            results.append(result)
        
        return results
    
    def run_refuter(self, samples: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        Run Refuter agent on samples.
        
        Args:
            samples: List of samples with proposer outputs
            max_retries: Maximum retries on parsing failures
            
        Returns:
            List of samples with refuter outputs
        """
        results = []
        
        for sample in samples:
            non_cue_text = sample.get('non_cue_text', '')
            proposer_letter = sample.get('proposer_letter')
            
            # Skip if no proposer letter
            if not self.validate_letter(proposer_letter):
                result = sample.copy()
                result['refuter_letter'] = None
                result['refuter_spans'] = []
                results.append(result)
                continue
            
            # Check cache first
            cache_key = self._cache_key('refuter', non_cue_text=non_cue_text, proposer_letter=proposer_letter)
            cached_output = self._get_cache(cache_key)
            
            if cached_output:
                try:
                    cached_data = json.loads(cached_output)
                    refuter_letter = cached_data.get('refuter_letter')
                    refuter_spans = cached_data.get('refuter_spans', [])
                    if self.validate_letter(refuter_letter):
                        result = sample.copy()
                        result['refuter_letter'] = refuter_letter
                        result['refuter_spans'] = refuter_spans
                        results.append(result)
                        continue
                except:
                    pass  # Cache miss or invalid, continue to run
            
            # Run Refuter with retries
            refuter_letter = None
            refuter_spans = []
            
            for attempt in range(max_retries):
                try:
                    prediction = self.refuter.predict_single(non_cue_text, proposer_letter)
                    refuter_letter = prediction['refuter_letter']
                    refuter_spans = prediction['refuter_spans']
                    
                    # Validate letter
                    if self.validate_letter(refuter_letter):
                        # Cache the result
                        self._set_cache(cache_key, json.dumps({
                            'refuter_letter': refuter_letter,
                            'refuter_spans': refuter_spans
                        }))
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Warning: Refuter failed after {max_retries} attempts: {e}")
            
            # Handle empty spans
            if not refuter_spans:
                refuter_spans = []
            
            # Preserve proposer choice if parsing failed (don't override with 'a')
            if not self.validate_letter(refuter_letter):
                refuter_letter = proposer_letter  # Use proposer choice instead of defaulting to 'a'
            
            result = sample.copy()
            result['refuter_letter'] = refuter_letter
            result['refuter_spans'] = refuter_spans
            results.append(result)
        
        return results
    
    def run_judge(self, samples: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        Run Judge agent on samples.
        
        Args:
            samples: List of samples with proposer and refuter outputs
            max_retries: Maximum retries on parsing failures
            
        Returns:
            List of samples with final verdicts
        """
        results = []
        
        for sample in samples:
            proposer_letter = sample.get('proposer_letter')
            refuter_letter = sample.get('refuter_letter')
            refuter_spans = sample.get('refuter_spans', [])
            non_cue_text = sample.get('non_cue_text', '')
            proposer_masked_letter = sample.get('proposer_masked_letter')
            
            # Check cache first
            cache_key = self._cache_key(
                'judge',
                proposer_letter=proposer_letter,
                refuter_letter=refuter_letter,
                refuter_spans=refuter_spans,
                non_cue_text=non_cue_text,
                proposer_masked_letter=proposer_masked_letter
            )
            cached_output = self._get_cache(cache_key)
            
            if cached_output:
                try:
                    cached_data = json.loads(cached_output)
                    final_choice = cached_data.get('final_choice')
                    reason = cached_data.get('reason')
                    if self.validate_letter(final_choice):
                        result = sample.copy()
                        result['final_choice'] = final_choice
                        result['reason'] = reason
                        result['final_label'] = letter_to_label(final_choice)
                        results.append(result)
                        continue
                except:
                    pass  # Cache miss or invalid, continue to run
            
            # Run Judge with retries
            final_choice = None
            reason = None
            
            for attempt in range(max_retries):
                try:
                    prediction = self.judge.predict_single(
                        proposer_letter, refuter_letter, refuter_spans,
                        non_cue_text, proposer_masked_letter, use_rules=True
                    )
                    final_choice = prediction['final_choice']
                    reason = prediction['reason']
                    
                    # Validate letter
                    if self.validate_letter(final_choice):
                        # Cache the result
                        self._set_cache(cache_key, json.dumps({
                            'final_choice': final_choice,
                            'reason': reason
                        }))
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Warning: Judge failed after {max_retries} attempts: {e}")
            
            # Default to 'a' if parsing failed
            if not self.validate_letter(final_choice):
                final_choice = 'a'
                reason = 'default_fallback'
            
            result = sample.copy()
            result['final_choice'] = final_choice
            result['reason'] = reason
            result['final_label'] = letter_to_label(final_choice)
            results.append(result)
        
        return results
    
    def run_pipeline(self, samples: List[Dict], ablation: Optional[Dict] = None) -> List[Dict]:
        """
        Run full pipeline (Proposer → Refuter → Judge).
        
        Args:
            samples: List of sample dicts with 'id', 'text', 'trigger_text'
            ablation: Ablation configuration (proposer_only, proposer_refuter, full_triad)
            
        Returns:
            List of samples with all agent outputs and final verdict
        """
        if ablation is None:
            ablation = self.config.get('ablation', {})
        
        print("=" * 80)
        print("AGENTIC PIPELINE")
        print("=" * 80)
        
        # Step 1: Sectionize notes
        print("\nStep 1: Sectionizing notes...")
        sectionized_samples = []
        for sample in samples:
            note = sample['text']
            trigger = sample['trigger_text']
            
            sectionized = sectionize_note(
                note, trigger,
                use_sections=self.config.get('sectionizer', {}).get('use_sections'),
                mask_trigger=self.config.get('sectionizer', {}).get('mask_trigger_sentence', True)
            )
            
            result = sample.copy()
            result['masked_note'] = sectionized['masked_note']
            result['non_cue_text'] = sectionized['non_cue_text']
            result['sections'] = sectionized['sections']
            sectionized_samples.append(result)
        
        # Step 2: Run Proposer
        print("\nStep 2: Running Proposer...")
        proposer_samples = self.run_proposer(sectionized_samples)
        
        if ablation.get('proposer_only', False):
            print("Abortion: proposer_only mode, stopping after Proposer")
            return proposer_samples
        
        # Step 3: Run Refuter
        print("\nStep 3: Running Refuter...")
        refuter_samples = self.run_refuter(proposer_samples)
        
        if ablation.get('proposer_refuter', False):
            print("Abortion: proposer_refuter mode, stopping after Refuter")
            return refuter_samples
        
        # Step 4: Run Judge
        print("\nStep 4: Running Judge...")
        final_samples = self.run_judge(refuter_samples)
        
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print("=" * 80)
        
        return final_samples

