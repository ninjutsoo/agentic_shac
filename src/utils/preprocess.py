"""
Preprocessing module for SHAC data.

Cleans and processes loaded BRAT data into JSONL format.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def clean_text(text: str) -> str:
    """
    Clean text minimally while preserving semantics.
    
    Args:
        text: Raw text from BRAT files
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_event(event: Dict) -> bool:
    """
    Validate that an event has all required fields.
    
    Args:
        event: Event dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['id', 'split', 'source', 'note_id', 'text', 'trigger_text', 'status_label']
    
    for field in required_fields:
        if field not in event:
            return False
        if event[field] is None or (isinstance(event[field], str) and event[field] == ''):
            if field != 'trigger_text':  # Empty trigger text is invalid
                return False
    
    return True


def process_events(events: List[Dict], clean: bool = True) -> List[Dict]:
    """
    Process a list of events.
    
    Args:
        events: List of raw event dictionaries
        clean: Whether to clean text (default: True)
        
    Returns:
        List of processed event dictionaries
    """
    processed = []
    
    for event in events:
        # Validate
        if not validate_event(event):
            print(f"Warning: Invalid event skipped: {event.get('id', 'unknown')}")
            continue
        
        # Clean text if requested
        if clean:
            event['text'] = clean_text(event['text'])
            event['trigger_text'] = clean_text(event['trigger_text'])
        
        processed.append(event)
    
    return processed


def save_to_jsonl(events: List[Dict], output_path: Path):
    """
    Save events to JSONL format.
    
    Args:
        events: List of event dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    print(f"Saved {len(events)} events to {output_path}")


def load_from_jsonl(input_path: Path) -> List[Dict]:
    """
    Load events from JSONL format.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        List of event dictionaries
    """
    events = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    
    return events


def get_split_statistics(events: List[Dict]) -> Dict:
    """
    Compute statistics for a split.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_events': len(events),
        'sources': defaultdict(int),
        'labels': defaultdict(int),
        'notes': set(),
        'text_lengths': []
    }
    
    for event in events:
        stats['sources'][event['source']] += 1
        stats['labels'][event['status_label']] += 1
        stats['notes'].add(event['note_id'])
        stats['text_lengths'].append(len(event['text']))
    
    stats['unique_notes'] = len(stats['notes'])
    stats['sources'] = dict(stats['sources'])
    stats['labels'] = dict(stats['labels'])
    
    if stats['text_lengths']:
        stats['avg_text_length'] = sum(stats['text_lengths']) / len(stats['text_lengths'])
        stats['min_text_length'] = min(stats['text_lengths'])
        stats['max_text_length'] = max(stats['text_lengths'])
    
    return stats


def preprocess_and_save(
    events: List[Dict],
    output_dir: Path,
    split_by: str = 'split',
    clean: bool = True
):
    """
    Preprocess events and save to JSONL files, split by a field.
    
    Args:
        events: List of raw event dictionaries
        output_dir: Output directory for JSONL files
        split_by: Field to split by (default: 'split')
        clean: Whether to clean text (default: True)
    """
    # Process events
    processed = process_events(events, clean=clean)
    
    print(f"\nProcessed {len(processed)} events (from {len(events)} raw events)")
    
    # Group by split
    splits = defaultdict(list)
    for event in processed:
        split_val = event.get(split_by, 'unknown')
        splits[split_val].append(event)
    
    # Save each split
    output_dir = Path(output_dir)
    
    for split_name, split_events in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        save_to_jsonl(split_events, output_path)
        
        # Print statistics
        stats = get_split_statistics(split_events)
        print(f"\n{split_name.upper()} statistics:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Unique notes: {stats['unique_notes']}")
        print(f"  Sources: {stats['sources']}")
        print(f"  Labels: {stats['labels']}")
        print(f"  Avg text length: {stats['avg_text_length']:.1f} chars")

