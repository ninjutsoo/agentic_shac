# Agentic SHAC: Drug StatusTime Classification

## Project Overview

This project implements an **agentic multi-model system** for classifying Drug StatusTime in clinical social history notes from the SHAC dataset. The approach uses adversarial agents (Proposer/Refuter/Judge) to improve false positive rate (FPR) compared to single-model baselines.

### Key Features
- **Data Source**: SHAC social-history sections (MIMIC subset as primary)
- **Task**: Drug StatusTime classification as multiple-choice argument-resolution
- **Primary Metric**: False Positive Rate (FPR) for safety
- **Model**: Llama-3.1-8B-Instruct
- **Architecture**: Agentic pipeline with Proposer, Refuter, and Judge agents

### Status Labels
- `none` - No current or past drug use
- `current` - Current drug use
- `past` - Past drug use
- `Not Applicable` - Cannot be determined

---

## Repository Structure

```
agentic_shac/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── agentic.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/            # symlink from Track_2_SHAC/SHAC/{train,dev,test}/{mimic,uw}
│   └── processed/
├── experiments/
│   ├── baseline/
│   ├── agentic/
│   └── reports/
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_baseline_results.ipynb
│   ├── 02_agentic_pipeline.ipynb
│   ├── 03_fpr_analysis.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── utils/
│   │   ├── brat_loader.py
│   │   ├── preprocess.py
│   │   ├── sectionizer.py
│   │   ├── io.py
│   │   ├── logger.py
│   │   └── seed.py
│   ├── baselines/
│   │   └── llama_single.py
│   ├── agentic/
│   │   ├── interfaces.py
│   │   ├── prompts.py
│   │   ├── proposer.py
│   │   ├── refuter.py
│   │   ├── judge.py
│   │   └── pipeline.py
│   └── evaluation/
│       ├── metrics.py
│       ├── run_baseline.py
│       ├── run_agentic.py
│       ├── compare_runs.py
│       └── plots.py
├── tests/
│   ├── test_brat_loader.py
│   ├── test_preprocess.py
│   ├── test_prompts.py
│   └── test_agentic_pipeline.py
└── scripts/
    └── run_smoke.sh
```

---

## Design Principles

### Model Selection
- **Primary Model**: Llama-3.1-8B-Instruct
- **Rationale**: Instruction-strong, fits available GPUs (bf16 on 32GB; 4-bit on 24GB)
- **Compatibility**: Aligns with paper if fine-tuning later

### Data Pipeline
- Single data loader and preprocessor (no duplicate pipes)
- Keep only SHAC social-history sections
- Link to MIMIC only for demographics if needed

### Evaluation Metrics
- **Primary**: False Positive Rate (FPR)
- **Secondary**: Accuracy
- **Excluded**: ROC/AUC (not meaningful for 3-class with asymmetric costs)

### Prompt Strategy
- Keep prompts terse and direct
- No Chain-of-Thought (CoT)
- No explicit warnings
- Adversarial pressure provided by multi-agent architecture

---

## Implementation Roadmap

### 1. Data Handling

#### Configuration (`configs/data.yaml`)

```yaml
raw_root: "/home/amin/Dropbox/code/SDOH/Track_2_SHAC/SHAC"
sources: ["mimic"]         # Use "mimic" only to match paper; can switch to ["mimic","uw"]
splits: ["train", "dev", "test"]
target_event: "Drug"
status_labels: ["none", "current", "past", "Not Applicable"]
```

#### BRAT Loader (`src/utils/brat_loader.py`)

**Goal**: Parse `.txt`/`.ann` files → rows for Drug events with StatusTime

**Output Schema**:
```json
{
  "id": "m_train_000123_drug_1",
  "split": "train",
  "source": "mimic",
  "note_id": "...",
  "text": "<full social-history text>",
  "trigger_text": "drug",
  "status_label": "current" | "past" | "none" | "Not Applicable"
}
```

**BRAT Mapping**:
- Event (E) links Alcohol|Drug|Tobacco trigger (T) to Status arg (T)
- Attribute (A) StatusTimeVal holds the categorical label
- **Filter**: Keep only `event_type == "Drug"`

#### Preprocessor (`src/utils/preprocess.py`)

**Input**: List of dicts from `brat_loader`

**Tasks**:
- Clean text minimally (normalize whitespace; preserve semantics)
- Create processed JSONL per split: `data/processed/{split}.jsonl`
- Optional: Ensure patient-level non-leak splits if patient IDs exist

#### Testing (`tests/test_brat_loader.py`)

- Feed 2–3 tiny `.txt`/`.ann` fixtures
- Assert one row per Drug trigger and correct status mapping

#### EDA Notebook (`notebooks/00_eda.ipynb`)

- Label counts and distribution
- Split sizes
- FPR-relevant class ratios (gold `none`/`Not Applicable` proportion)

---

### 2. Baseline Model (Single Predictor)

**Task**: Predict StatusTime for Drug given the full SHAC note and known trigger (mirrors paper's argument-resolution step)

#### Configuration (`configs/baseline.yaml`)

```yaml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
dtype: bf16           # Set load_in_4bit: true for 24 GB GPUs
load_in_4bit: false
max_new_tokens: 8
temperature: 0.1
top_p: 0.9
batch_size: 8
prompt_template: "status_v1"
```

#### Inference Engine (`src/baselines/llama_single.py`)

- Batched inference
- Strict option parsing: `a|b|c|d` → status label
- Save to `experiments/baseline/preds.jsonl` with fields: `id`, `pred`, `gold`, `raw_text`

#### Prompt Template (`src/agentic/prompts.py`)

**status_v1** (terse, no CoT, no warnings):
```
System: You classify Drug StatusTime in clinical notes.
User:
Note:
{NOTE}

Drug trigger: "{TRIGGER}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter.
```

#### Baseline Runner (`src/evaluation/run_baseline.py`)

1. Load `data/processed/{split}.jsonl`
2. Assemble prompts
3. Run `llama_single.py`
4. Write predictions

#### Metrics (`src/evaluation/metrics.py`)

**Map letters → labels**

**Compute**:
- **Accuracy**: Overall classification accuracy
- **FPR**: False Positive Rate where gold ∈ {`none`, `Not Applicable`}
  - Policy: `none` is negative class
  - Decision on `Not Applicable`: Exclude from FPR denominator OR treat as negative (state choice in README)

#### Testing (`tests/test_baseline_prompt.py`)

- Verify correct letter extraction and label mapping

#### Results Notebook (`notebooks/01_baseline_results.ipynb`)

- Summarize baseline accuracy + FPR per split

---

### 3. Agentic Pipeline (Proposer / Refuter / Judge)

#### Data Interfaces (`src/agentic/interfaces.py`)

```python
from dataclasses import dataclass
from typing import List, Optional

LABELS = ["none", "current", "past", "Not Applicable"]

@dataclass
class Claim:
    id: str
    choice: str  # "a"|"b"|"c"|"d"

@dataclass
class Evidence:
    spans: List[str]
    choice: str

@dataclass
class Verdict:
    final_choice: str
    reason: str
```

#### Agent Prompts (Concise)

**Proposer** (`proposer_v1.txt`):
```
System: You classify Drug StatusTime in clinical notes.
User:
Note:
{NOTE}

Drug trigger: "{TRIGGER}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter only.
```

**Refuter** (`refuter_v1.txt`):
```
System: You challenge Drug StatusTime decisions using only non-cue sections.
User:
Non-cue text:
{NON_CUE_TEXT}

Proposer chose: {P_LETTER}
Argue for the opposite letter if supported by non-cue text.
Options: (a) none (b) current (c) past (d) Not Applicable
Return:
letter: <a|b|c|d>
spans:
- "<short quote or empty>"
- "<short quote or empty>"
```

**Judge** (`judge_v1.txt`):
```
System: You decide Drug StatusTime using only non-cue evidence quality.
User:
Inputs:
- Proposer: {P_LETTER}
- Refuter: {R_LETTER}
- Refuter spans: {R_SPANS}
- Non-cue text: {NON_CUE_TEXT}
- Proposer on masked note: {P_MASKED}

Rules:
1) Prefer the letter supported by Refuter spans.
2) If spans give no support, choose (a) none.
3) If Proposer flips on masked note and Refuter stays stable, prefer Refuter.

Return one letter.
```

#### Sectionizer (`src/utils/sectionizer.py`)

- Heuristic split into: `assessment_plan`, `problems`, `meds`, `labs`, `other`
- Build `non_cue_text` = concat(`assessment_plan`, `problems`, `meds`, `labs`)
- Build `masked_note` = note with sentence containing TRIGGER removed

#### Proposer Agent (`src/agentic/proposer.py`)

- Run `proposer_v1.txt` per item on full NOTE
- Compute `P_MASKED` by running on `masked_note`
- Save `proposer.jsonl`: `id`, `proposer_letter`, `proposer_masked_letter`

#### Refuter Agent (`src/agentic/refuter.py`)

- Input: `non_cue_text` + `proposer_letter`
- Output: `refuter_letter` and up to two quoted spans
- Save `refuter.jsonl`

#### Judge Agent (`src/agentic/judge.py`)

- Input: proposer/refuter outputs + `non_cue_text`
- Apply decision rules
- Default to `(a) none` if ambiguous (per rule 2)
- Save `final.jsonl`

#### Pipeline Orchestrator (`src/agentic/pipeline.py`)

- Orchestrates batched runs through all three agents
- Retries on parsing failures
- Caches prompts→outputs in `experiments/agentic/cache/`

#### Configuration (`configs/agentic.yaml`)

```yaml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
dtype: bf16
load_in_4bit: false
batch_size: 8
max_new_tokens: 8
temperature: 0.1
top_p: 0.9
prompts:
  proposer: "proposer_v1"
  refuter: "refuter_v1"
  judge: "judge_v1"
sectionizer:
  use_sections: ["assessment_plan", "problems", "meds", "labs"]
mask_trigger_sentence: true
ablation:
  proposer_only: false
  proposer_refuter: false
  full_triad: true
```

#### Testing (`tests/test_agentic_pipeline.py`)

- Fixture: 3 tiny notes covering each class
- Assert pipeline returns one of `a|b|c|d` for each
- Snapshot JSON of results for regression testing

#### Pipeline Notebook (`notebooks/02_agentic_pipeline.ipynb`)

- Show sample rows with proposer/refuter/judge decisions
- Display quoted spans from refuter

---

### 4. Evaluation & FPR Analysis

#### Agentic Runner (`src/evaluation/run_agentic.py`)

- Run pipeline on dev/test splits
- Save to `experiments/agentic/<run_id>/final.jsonl`

#### Comparison Tool (`src/evaluation/compare_runs.py`)

- Load baseline vs agentic predictions
- Compute FPR deltas and accuracy deltas (per split and overall)
- Output CSV and Markdown summary to `experiments/reports/`

#### Visualization (`src/evaluation/plots.py`)

- Matplotlib bar plots for FPR by configuration
- No seaborn dependency

#### FPR Analysis Notebook (`notebooks/03_fpr_analysis.ipynb`)

- Compare baseline vs agentic FPR drop
- Optional: Alcohol+/Smoking+ subgroup analyses
  - Paper stratifies by substance context
  - Can replicate by tagging notes with alcohol/smoking mentions

---

### 5. Testing & Quality Assurance

#### Smoke Test (`tests/run_smoke.sh`)

End-to-end smoke test pipeline:
1. Loader → Preprocess
2. Baseline on 3 samples
3. Agentic on 3 samples
4. Metrics computation

#### Dependencies (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
# Add default markers

# Optional linters
# ruff + black + mypy
```

---

### 6. Experiment Management

#### Logger (`src/utils/logger.py`)

- Writes JSON logs per run
- Every run writes:
  - `experiments/<run_id>/config.json`
  - `experiments/<run_id>/system_prompts.txt`

#### Reproducibility (`src/utils/seed.py`)

- Fix random seeds for all sampling operations

---

### 7. Deployment & Future Work

#### Optional CLI (`src/cli.py`)

```bash
# Run baseline
python -m src.cli baseline --split test

# Run agentic pipeline
python -m src.cli agentic --split test

# Evaluate and compare
python -m src.cli evaluate --runs baseline,agentic
```

#### Future Enhancements

1. **Few-shot ICL**: Try 3-shot in-context learning to mirror paper variants
2. **Fine-tuning**: LoRA fine-tune Llama-3.1-8B with paper hyperparameters
3. **Extended Events**: Add Alcohol and Tobacco StatusTime classification
4. **Multi-dataset**: Expand to UW dataset for generalization testing

---

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (32GB for bf16, 24GB for 4-bit)
- Access to SHAC dataset

### Installation

```bash
# Clone repository
git clone git@github.com:ninjutsoo/agentic_shac.git
cd agentic_shac

# Create conda environment
conda create -n agentic_shac python=3.10
conda activate agentic_shac

# Install dependencies
pip install -e .
```

### Quick Start

```bash
# 1. Prepare data
python -m src.utils.preprocess

# 2. Run baseline
python -m src.evaluation.run_baseline --split dev

# 3. Run agentic pipeline
python -m src.evaluation.run_agentic --split dev

# 4. Compare results
python -m src.evaluation.compare_runs --baseline experiments/baseline --agentic experiments/agentic
```

---

## License

[Add appropriate license]

## Citation

If you use this code, please cite the original SHAC paper:

```
[Add citation]
```

---

**Last Updated**: October 2025