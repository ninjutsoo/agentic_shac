"""
Prompt templates for Drug StatusTime classification.

All prompts are terse, no CoT, no warnings.
"""

from typing import Dict, Optional, Tuple, List
import re

# Prompt templates
PROMPTS = {
    "status_v1": {
        "system": "You classify Drug StatusTime in clinical notes.",
        "user_template": """Note:
{note}

Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter."""
    },
    "status_v2": {
        "system": "You are a clinical NLP assistant. Classify temporal drug use status for the PATIENT only.",
        "user_template": """Note:
{note}

Choose EXACTLY ONE letter:
(a) none            = patient denies use OR no evidence about patient using
(b) current         = evidence patient currently/recently uses
(c) past            = patient used in the past but not currently
(d) Not Applicable  = mention is not about the PATIENT's status (e.g., family/other/context)

Rules:
- If negation about the patient's use (denies, negative, no), choose (a).
- If history/quit/clean for years without current use, choose (c).
- Only use (d) when the mention is clearly NOT about the patient or status is indeterminate despite evidence.
- Otherwise choose the best of (a/b/c).

Answer with ONE letter only: a or b or c or d."""
    },
    "status_v3": {
        "system": "Classify drug use status from clinical notes.",
        "user_template": """Note:
{note}

Classify patient's drug use status:
(a) none - denies use or no evidence
(b) current - currently uses
(c) past - history of use, not current
(d) Not Applicable - not about patient

Answer:"""
    },
    "proposer_v1": {
        "system": "You classify Drug StatusTime in clinical notes.",
        "user_template": """Note:
{note}

Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter only."""
    },
    "refuter_v1": {
        "system": "You challenge Drug StatusTime decisions using only non-cue sections.",
        "user_template": """Non-cue text:
{non_cue_text}

Proposer chose: {proposer_letter}
Argue for the opposite letter if supported by non-cue text.
Options: (a) none (b) current (c) past (d) Not Applicable
Return:
letter: <a|b|c|d>
spans:
- "<short quote or empty>"
- "<short quote or empty>"""
    },
    "judge_v1": {
        "system": "You decide Drug StatusTime using only non-cue evidence quality.",
        "user_template": """Inputs:
- Proposer: {proposer_letter}
- Refuter: {refuter_letter}
- Refuter spans: {refuter_spans}
- Non-cue text: {non_cue_text}
- Proposer on masked note: {proposer_masked_letter}

Rules:
1) Prefer the letter supported by Refuter spans.
2) If spans give no support, choose (a) none.
3) If Proposer flips on masked note and Refuter stays stable, prefer Refuter.

Return one letter."""
    }
}


def get_prompt(template_name: str, **kwargs) -> Dict[str, str]:
    """
    Get a formatted prompt.
    
    Args:
        template_name: Name of the prompt template
        **kwargs: Values to fill in the template
        
    Returns:
        Dictionary with 'system' and 'user' keys
    """
    if template_name not in PROMPTS:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(PROMPTS.keys())}")
    
    template = PROMPTS[template_name]
    
    return {
        "system": template["system"],
        "user": template["user_template"].format(**kwargs)
    }


def format_for_llama(system: str, user: str) -> str:
    """
    Format prompt for Llama model.
    
    Args:
        system: System message
        user: User message
        
    Returns:
        Formatted prompt string
    """
    # Llama-3.1-Instruct format
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# Label mapping
LABEL_TO_LETTER = {
    "none": "a",
    "current": "b", 
    "past": "c",
    "Not Applicable": "d"
}

LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}


def parse_model_output(output: str) -> str:
    """
    Parse model output to extract letter choice.
    
    Args:
        output: Raw model output
        
    Returns:
        Extracted letter (a/b/c/d) or None if invalid
    """
    output = output.strip().lower()

    # Prefer the LAST explicit option in parentheses to avoid prompt echoes
    matches = re.findall(r"\([abcd]\)", output)
    if matches:
        return matches[-1][1]

    # Fallback: scan from end for a standalone letter a/b/c/d
    for char in reversed(output):
        if char in ['a', 'b', 'c', 'd']:
            return char

    return None


def letter_to_label(letter: str) -> str:
    """
    Convert letter to label.
    
    Args:
        letter: Letter choice (a/b/c/d)
        
    Returns:
        Status label
    """
    return LETTER_TO_LABEL.get(letter, "Not Applicable")


def parse_refuter_output(output: str) -> Tuple[Optional[str], List[str]]:
    """
    Parse Refuter agent output to extract letter and spans.
    
    Args:
        output: Raw model output
        
    Returns:
        Tuple of (letter, spans_list) where letter is a/b/c/d or None, spans_list is list of strings
    """
    output = output.strip()
    letter = None
    spans = []
    
    # Extract letter
    letter_match = re.search(r'letter:\s*([abcd])', output, re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).lower()
    else:
        # Fallback: try parse_model_output
        letter = parse_model_output(output)
    
    # Extract spans
    spans_section = re.search(r'spans?:?\s*\n((?:-.*\n?)*)', output, re.IGNORECASE | re.MULTILINE)
    if spans_section:
        span_lines = spans_section.group(1).strip().split('\n')
        for line in span_lines:
            # Remove bullet point markers and quotes
            span = re.sub(r'^[\s-]*["\']?', '', line.strip())
            span = re.sub(r'["\']?\s*$', '', span)
            if span and span.lower() not in ['empty', '<empty>', '']:
                spans.append(span)
    
    # Limit to 2 spans as per prompt
    spans = spans[:2]
    
    return letter, spans


def parse_judge_output(output: str) -> Optional[str]:
    """
    Parse Judge agent output to extract final letter choice.
    
    Args:
        output: Raw model output
        
    Returns:
        Extracted letter (a/b/c/d) or None if invalid
    """
    return parse_model_output(output)

