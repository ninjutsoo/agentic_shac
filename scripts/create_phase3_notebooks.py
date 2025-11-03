"""Script to create Phase 3 test notebooks."""

import json
from pathlib import Path

# Notebook templates
notebooks = {
    '07_test_sectionizer.ipynb': {
        'title': '07_test_sectionizer',
        'description': 'Test text sectionization and masking for Phase 3 agentic pipeline.',
        'cells': [
            {
                'type': 'markdown',
                'source': '''# 07_test_sectionizer

Test text sectionization and masking for Phase 3 agentic pipeline.

This notebook validates:
1. Text sectionization into assessment_plan, problems, meds, labs, other
2. Non-cue text extraction (concatenated sections)
3. Trigger sentence masking (removing sentences containing trigger)'''
            },
            {
                'type': 'code',
                'source': '''# Setup
import sys
from pathlib import Path

# Resolve project root
CWD = Path.cwd()
if (CWD / 'configs').exists() and (CWD / 'src').exists():
    PROJECT_ROOT = CWD
elif CWD.name == 'notebooks' and (CWD.parent / 'configs').exists():
    PROJECT_ROOT = CWD.parent
else:
    cur = CWD
    PROJECT_ROOT = CWD
    while cur != cur.parent:
        if (cur / 'configs').exists() and (cur / 'src').exists():
            PROJECT_ROOT = cur
            break
        cur = cur.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sectionizer import sectionize_note

print('Project root:', PROJECT_ROOT)'''
            },
            {
                'type': 'code',
                'source': '''# Test samples
test_samples = [
    {
        'text': 'Patient reports daily cocaine use. Assessment: Active substance abuse. Plan: Refer to addiction services.',
        'trigger': 'cocaine'
    },
    {
        'text': 'Problems: Drug overdose. Assessment and Plan: Patient has history of heroin use. Medications: None.',
        'trigger': 'heroin'
    },
    {
        'text': 'Chief Complaint: Patient denies current drug use. History: Past cocaine dependence. Labs: Normal.',
        'trigger': 'cocaine'
    }
]

print(f'Testing {len(test_samples)} samples')'''
            },
            {
                'type': 'code',
                'source': '''# Test sectionization
for i, sample in enumerate(test_samples, 1):
    print(f"\\n{'='*80}")
    print(f"Sample {i}")
    print(f"{'='*80}")
    print(f"Original text: {sample['text']}")
    print(f"Trigger: {sample['trigger']}")
    
    result = sectionize_note(
        sample['text'],
        sample['trigger'],
        use_sections=['assessment_plan', 'problems', 'meds', 'labs'],
        mask_trigger=True
    )
    
    print(f"\\nSections:")
    for section_name, section_text in result['sections'].items():
        if section_text:
            print(f"  - {section_name}: {section_text[:80]}...")
    
    print(f"\\nNon-cue text: {result['non_cue_text'][:100]}...")
    print(f"\\nMasked note: {result['masked_note']}")
    
    # Verify trigger is not in masked note
    if sample['trigger'].lower() in result['masked_note'].lower():
        print(f"  ⚠️  Warning: Trigger '{sample['trigger']}' still found in masked note!")
    else:
        print(f"  ✅ Trigger '{sample['trigger']}' successfully masked")'''
            }
        ]
    },
    # Add more notebooks here as needed - truncated for brevity
    # The other notebooks would follow similar patterns
}

def create_notebook(name, cells_data):
    """Create a Jupyter notebook file."""
    cells = []
    for cell_data in cells_data['cells']:
        if cell_data['type'] == 'markdown':
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': cell_data['source'].split('\\n')
            })
        else:
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': cell_data['source'].split('\\n')
            })
    
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    return notebook

if __name__ == '__main__':
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks_dir.mkdir(exist_ok=True)
    
    # Create sectionizer notebook
    notebook = create_notebook('07_test_sectionizer.ipynb', notebooks['07_test_sectionizer.ipynb'])
    with open(notebooks_dir / '07_test_sectionizer.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created {notebooks_dir / '07_test_sectionizer.ipynb'}")

