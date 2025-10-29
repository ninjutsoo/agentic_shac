"""
BRAT annotation loader for SHAC dataset.

Parses .txt and .ann files to extract Drug events with StatusTime labels.
"""

from pathlib import Path
from typing import List, Dict, Optional
import re


class BRATLoader:
    """Load and parse BRAT annotations for SHAC dataset."""
    
    def __init__(self, target_event: str = "Drug"):
        """
        Initialize BRAT loader.
        
        Args:
            target_event: Event type to extract (default: "Drug")
        """
        self.target_event = target_event
    
    def parse_ann_file(self, ann_path: Path) -> Dict:
        """
        Parse a single .ann file.
        
        Args:
            ann_path: Path to .ann file
            
        Returns:
            Dictionary with entities, events, and attributes
        """
        entities = {}  # T_id -> {type, start, end, text}
        events = {}    # E_id -> {type, args}
        attributes = {}  # A_id -> {type, target, value}
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse text-bound annotations (entities)
                if line.startswith('T'):
                    parts = line.split('\t')
                    t_id = parts[0]
                    
                    # Handle annotations with multiple spans (e.g., "T13	TypeLiving 76 80;81 113	with ...")
                    type_and_spans = parts[1].split()
                    entity_type = type_and_spans[0]
                    
                    # Get first span (start and end)
                    span_parts = type_and_spans[1].split(';')
                    start = int(span_parts[0].split()[0] if ' ' in span_parts[0] else span_parts[0])
                    
                    # Get last span end
                    if ';' in parts[1]:
                        # Multiple spans - get end from last span
                        all_spans = parts[1].split(entity_type)[1].strip().split(';')
                        end = int(all_spans[-1].strip().split()[-1])
                    else:
                        # Single span
                        end = int(type_and_spans[2])
                    
                    text = parts[2] if len(parts) > 2 else ""
                    
                    entities[t_id] = {
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'text': text
                    }
                
                # Parse events
                elif line.startswith('E'):
                    parts = line.split('\t')
                    e_id = parts[0]
                    event_info = parts[1]
                    
                    # Parse event: "Drug:T7 Type:T9 Status:T8"
                    event_parts = event_info.split()
                    event_type = event_parts[0].split(':')[0]
                    trigger = event_parts[0].split(':')[1]
                    
                    # Parse arguments
                    args = {'trigger': trigger}
                    for part in event_parts[1:]:
                        arg_name, arg_val = part.split(':')
                        args[arg_name] = arg_val
                    
                    events[e_id] = {
                        'type': event_type,
                        'args': args
                    }
                
                # Parse attributes
                elif line.startswith('A'):
                    parts = line.split()
                    a_id = parts[0]
                    attr_type = parts[1]
                    target_id = parts[2]
                    value = parts[3] if len(parts) > 3 else None
                    
                    attributes[a_id] = {
                        'type': attr_type,
                        'target': target_id,
                        'value': value
                    }
        
        return {
            'entities': entities,
            'events': events,
            'attributes': attributes
        }
    
    def extract_drug_events(
        self, 
        ann_data: Dict, 
        text: str,
        note_id: str,
        source: str,
        split: str
    ) -> List[Dict]:
        """
        Extract Drug events with StatusTime labels.
        
        Args:
            ann_data: Parsed annotation data
            text: Full text content
            note_id: Note identifier
            source: Data source (mimic/uw)
            split: Data split (train/dev/test)
            
        Returns:
            List of drug event dictionaries
        """
        drug_events = []
        
        entities = ann_data['entities']
        events = ann_data['events']
        attributes = ann_data['attributes']
        
        # Find all Drug events
        for e_id, event in events.items():
            if event['type'] != self.target_event:
                continue
            
            # Get trigger and status
            trigger_id = event['args'].get('trigger')
            status_id = event['args'].get('Status')
            
            if not trigger_id or not status_id:
                continue
            
            # Get trigger text
            trigger_entity = entities.get(trigger_id, {})
            trigger_text = trigger_entity.get('text', '')
            
            # Find StatusTimeVal attribute for this status
            status_label = None
            for attr_id, attr in attributes.items():
                if attr['type'] == 'StatusTimeVal' and attr['target'] == status_id:
                    status_label = attr['value']
                    break
            
            # Handle missing status - mark as "Not Applicable"
            if status_label is None:
                status_label = "Not Applicable"
            
            # Create unique ID
            event_id = f"{source[0]}_{split}_{note_id}_{self.target_event.lower()}_{e_id}"
            
            drug_events.append({
                'id': event_id,
                'split': split,
                'source': source,
                'note_id': note_id,
                'text': text,
                'trigger_text': trigger_text,
                'status_label': status_label
            })
        
        return drug_events
    
    def load_from_directory(
        self, 
        data_root: Path, 
        sources: List[str], 
        splits: List[str]
    ) -> List[Dict]:
        """
        Load all Drug events from SHAC dataset directory.
        
        Args:
            data_root: Root directory of SHAC data
            sources: List of sources to load (e.g., ['mimic', 'uw'])
            splits: List of splits to load (e.g., ['train', 'dev', 'test'])
            
        Returns:
            List of all drug event dictionaries
        """
        all_events = []
        
        for split in splits:
            for source in sources:
                dir_path = Path(data_root) / split / source
                
                if not dir_path.exists():
                    print(f"Warning: Directory not found: {dir_path}")
                    continue
                
                # Find all .txt files
                txt_files = sorted(dir_path.glob("*.txt"))
                
                for txt_file in txt_files:
                    ann_file = txt_file.with_suffix('.ann')
                    
                    if not ann_file.exists():
                        print(f"Warning: Missing .ann file for {txt_file}")
                        continue
                    
                    # Read text
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Parse annotations
                    ann_data = self.parse_ann_file(ann_file)
                    
                    # Extract drug events
                    note_id = txt_file.stem
                    events = self.extract_drug_events(
                        ann_data=ann_data,
                        text=text,
                        note_id=note_id,
                        source=source,
                        split=split
                    )
                    
                    all_events.extend(events)
        
        return all_events


def load_shac_data(
    data_root: str,
    sources: List[str] = ["mimic", "uw"],
    splits: List[str] = ["train", "dev", "test"],
    target_event: str = "Drug"
) -> List[Dict]:
    """
    Convenience function to load SHAC data.
    
    Args:
        data_root: Root directory of SHAC data
        sources: List of sources to load
        splits: List of splits to load
        target_event: Event type to extract
        
    Returns:
        List of drug event dictionaries
    """
    loader = BRATLoader(target_event=target_event)
    return loader.load_from_directory(
        data_root=Path(data_root),
        sources=sources,
        splits=splits
    )

