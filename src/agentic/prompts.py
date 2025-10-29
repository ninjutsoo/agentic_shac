"""
Prompt templates for Drug StatusTime classification.

All prompts are terse, no CoT, no warnings.
"""

from typing import Dict

# Prompt templates
PROMPTS = {
    "status_v1": {
        "system": "You classify Drug StatusTime in clinical notes.",
        "user_template": """Note:
{note}

Drug trigger: "{trigger}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter."""
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
    
    # Extract first letter if present
    for char in output:
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

