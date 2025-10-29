"""
Process full SHAC dataset and save to JSONL files.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.brat_loader import load_shac_data
from src.utils.preprocess import preprocess_and_save
import yaml
import time

def main():
    print("="*80)
    print("Processing Full SHAC Dataset")
    print("="*80)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'data.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfiguration:")
    print(f"  Data root: {config['raw_root']}")
    print(f"  Sources: {config['sources']}")
    print(f"  Splits: {config['splits']}")
    print(f"  Target event: {config['target_event']}")
    
    # Load all data
    print(f"\n{'='*80}")
    print("Loading BRAT data...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    all_events = load_shac_data(
        data_root=config['raw_root'],
        sources=config['sources'],
        splits=config['splits'],
        target_event=config['target_event']
    )
    
    load_time = time.time() - start_time
    
    print(f"\n✅ Loaded {len(all_events)} Drug events in {load_time:.1f}s")
    
    # Preprocess and save
    print(f"\n{'='*80}")
    print("Preprocessing and saving...")
    print(f"{'='*80}")
    
    output_dir = Path(__file__).parent.parent / config['processed_dir']
    
    preprocess_and_save(
        events=all_events,
        output_dir=output_dir,
        split_by='split',
        clean=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ Processing complete in {total_time:.1f}s")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    for jsonl_file in sorted(output_dir.glob('*.jsonl')):
        size_kb = jsonl_file.stat().st_size / 1024
        print(f"  {jsonl_file.name}: {size_kb:.1f} KB")

if __name__ == '__main__':
    main()

