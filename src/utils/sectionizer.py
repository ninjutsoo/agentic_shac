"""
Sectionizer for clinical notes (Phase 3).

Heuristic splitting of notes into sections for non-cue text extraction.
"""

import re
from typing import Dict, List, Optional


def split_into_sections(note: str) -> Dict[str, str]:
    """
    Split a clinical note into sections using heuristics.
    
    Sections:
    - assessment_plan: Assessment and plan sections
    - problems: Problem list / chief complaint
    - meds: Medications / medication list
    - labs: Laboratory results
    - other: Everything else
    
    Args:
        note: Full clinical note text
        
    Returns:
        Dictionary mapping section names to their text content
    """
    sections = {
        "assessment_plan": "",
        "problems": "",
        "meds": "",
        "labs": "",
        "other": ""
    }
    
    # Normalize text for matching
    note_lower = note.lower()
    
    # Patterns for section headers (case-insensitive)
    patterns = {
        "assessment_plan": [
            r"(?:^|\n)\s*(?:assessment|plan|a&p|a/p|assessment\s*and\s*plan)[\s:]*\n",
            r"(?:^|\n)\s*(?:impression|imp)[\s:]*\n",
        ],
        "problems": [
            r"(?:^|\n)\s*(?:problems?|chief\s*complaint|cc|problem\s*list)[\s:]*\n",
            r"(?:^|\n)\s*(?:active\s*problems?)[\s:]*\n",
        ],
        "meds": [
            r"(?:^|\n)\s*(?:medications?|meds?|current\s*medications?)[\s:]*\n",
            r"(?:^|\n)\s*(?:medication\s*list|med\s*list)[\s:]*\n",
        ],
        "labs": [
            r"(?:^|\n)\s*(?:laboratory|labs?|lab\s*results?|lab\s*values?)[\s:]*\n",
            r"(?:^|\n)\s*(?:test\s*results?)[\s:]*\n",
        ],
    }
    
    # Find all section boundaries
    boundaries = []
    for section_name, section_patterns in patterns.items():
        for pattern in section_patterns:
            for match in re.finditer(pattern, note_lower, re.MULTILINE | re.IGNORECASE):
                boundaries.append((match.start(), section_name))
    
    # Sort boundaries by position
    boundaries.sort(key=lambda x: x[0])
    
    # Extract sections
    if not boundaries:
        # No sections found, put everything in "other"
        sections["other"] = note.strip()
        return sections
    
    # Split text based on boundaries
    current_pos = 0
    current_section = "other"
    
    for boundary_pos, section_name in boundaries:
        # Add text before boundary to current section
        if current_pos < boundary_pos:
            text_chunk = note[current_pos:boundary_pos].strip()
            if text_chunk:
                if sections[current_section]:
                    sections[current_section] += " " + text_chunk
                else:
                    sections[current_section] = text_chunk
        
        # Update position and section
        current_pos = boundary_pos
        current_section = section_name
    
    # Add remaining text
    if current_pos < len(note):
        text_chunk = note[current_pos:].strip()
        if text_chunk:
            if sections[current_section]:
                sections[current_section] += " " + text_chunk
            else:
                sections[current_section] = text_chunk
    
    # Clean up sections (remove extra whitespace)
    for key in sections:
        sections[key] = re.sub(r'\s+', ' ', sections[key]).strip()
    
    return sections


def build_non_cue_text(sections: Dict[str, str], use_sections: Optional[List[str]] = None) -> str:
    """
    Build non-cue text by concatenating specified sections.
    
    Args:
        sections: Dictionary of section name -> text
        use_sections: List of section names to include (default: assessment_plan, problems, meds, labs)
        
    Returns:
        Concatenated non-cue text
    """
    if use_sections is None:
        use_sections = ["assessment_plan", "problems", "meds", "labs"]
    
    non_cue_parts = []
    for section_name in use_sections:
        if section_name in sections and sections[section_name]:
            non_cue_parts.append(sections[section_name])
    
    return " ".join(non_cue_parts).strip()


def mask_trigger_sentence(note: str, trigger: str) -> str:
    """
    Remove the sentence containing the trigger from the note.
    
    Args:
        note: Full clinical note text
        trigger: Trigger word/phrase to mask
        
    Returns:
        Note with trigger sentence removed
    """
    # Split into sentences (simple heuristic: split on . ! ?)
    sentences = re.split(r'([.!?]\s+)', note)
    
    # Reconstruct sentences (alternating sentence and punctuation)
    reconstructed = []
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            reconstructed.append((sentence, punct))
    
    # Find and remove sentence containing trigger
    masked_sentences = []
    for sentence, punct in reconstructed:
        if trigger.lower() not in sentence.lower():
            masked_sentences.append(sentence + punct)
    
    return "".join(masked_sentences).strip()


def sectionize_note(note: str, trigger: str, use_sections: Optional[List[str]] = None, 
                     mask_trigger: bool = True) -> Dict[str, str]:
    """
    Full sectionization pipeline for a note.
    
    Args:
        note: Full clinical note text
        trigger: Trigger word/phrase
        use_sections: List of section names to include in non_cue_text
        mask_trigger: Whether to mask the trigger sentence
        
    Returns:
        Dictionary with:
        - sections: Dict of section name -> text
        - non_cue_text: Concatenated non-cue sections
        - masked_note: Note with trigger sentence removed (if mask_trigger=True)
    """
    sections = split_into_sections(note)
    non_cue_text = build_non_cue_text(sections, use_sections=use_sections)
    
    result = {
        "sections": sections,
        "non_cue_text": non_cue_text
    }
    
    if mask_trigger:
        result["masked_note"] = mask_trigger_sentence(note, trigger)
    else:
        result["masked_note"] = note
    
    return result

