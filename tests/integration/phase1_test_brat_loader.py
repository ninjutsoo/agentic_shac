"""Quick test script for BRAT loader"""

from pathlib import Path
from src.utils.brat_loader import BRATLoader
import yaml

# Load config
with open('configs/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*80)
print("Testing BRAT Loader")
print("="*80)

# Test single file
print("\n1. Testing single file parsing...")
sample_txt = Path(config['raw_root']) / 'train' / 'mimic' / '0101.txt'
sample_ann = sample_txt.with_suffix('.ann')

print(f"   Reading: {sample_txt}")

# Read text
with open(sample_txt, 'r') as f:
    text = f.read()

print(f"   Text length: {len(text)} chars")
print(f"   Text: {text[:100]}...")

# Parse annotations
loader = BRATLoader(target_event="Drug")
ann_data = loader.parse_ann_file(sample_ann)

print(f"\n   Parsed entities: {len(ann_data['entities'])}")
print(f"   Parsed events: {len(ann_data['events'])}")
print(f"   Parsed attributes: {len(ann_data['attributes'])}")

# Extract Drug events
events = loader.extract_drug_events(
    ann_data=ann_data,
    text=text,
    note_id='0101',
    source='mimic',
    split='train'
)

print(f"\n   ✅ Extracted {len(events)} Drug events")
for i, event in enumerate(events, 1):
    print(f"      Event {i}: trigger='{event['trigger_text']}', status='{event['status_label']}'")

# Test multiple files
print("\n2. Testing multiple files (3 from each source)...")
all_events = []

for source in ['mimic', 'uw']:
    dir_path = Path(config['raw_root']) / 'train' / source
    txt_files = sorted(dir_path.glob('*.txt'))[:3]
    
    print(f"\n   Processing {source}:")
    for txt_file in txt_files:
        ann_file = txt_file.with_suffix('.ann')
        
        with open(txt_file, 'r') as f:
            text = f.read()
        
        ann_data = loader.parse_ann_file(ann_file)
        events = loader.extract_drug_events(
            ann_data=ann_data,
            text=text,
            note_id=txt_file.stem,
            source=source,
            split='train'
        )
        all_events.extend(events)
        print(f"     {txt_file.stem}: {len(events)} events")

print(f"\n   ✅ Total events extracted: {len(all_events)}")

# Check labels
labels = {}
for event in all_events:
    label = event['status_label']
    labels[label] = labels.get(label, 0) + 1

print("\n3. Label distribution:")
for label, count in sorted(labels.items()):
    print(f"   {label}: {count} ({100*count/len(all_events):.1f}%)")

print("\n" + "="*80)
print("✅ BRAT Loader test completed successfully!")
print("="*80)

