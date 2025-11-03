# Agentic SHAC: Drug StatusTime Classification

## Project Overview

This project implements an **agentic multi-model system** for classifying Drug StatusTime in clinical social history notes from the SHAC dataset. The approach uses adversarial agents (Proposer/Refuter/Judge) to improve false positive rate (FPR) compared to single-model baselines.

### Key Features
- **Data Source**: SHAC social-history sections (MIMIC + UW datasets)
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
├── configs/                    # Configuration files
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── agentic.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/                    # Symlink from Track_2_SHAC/SHAC/
│   └── processed/              # Processed JSONL files
├── experiments/                # Experiment outputs
│   ├── baseline/
│   ├── agentic/
│   └── reports/
├── results/                    # Comprehensive comparison reports
│   └── comparison_report_*.txt # Detailed text reports with all metrics and timing
├── notebooks/                  # 📓 Interactive notebooks (USER runs these)
│   ├── 01_test_brat_loader.ipynb
│   ├── 02_test_preprocess.ipynb
│   ├── 03_data_eda.ipynb
│   ├── 04_test_model_loading.ipynb
│   ├── 05_test_baseline_inference.ipynb
│   └── ...
├── src/                        # Production code
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
│       ├── run_full_comparison.py
│       └── plots.py
├── tests/                      # 🧪 Automated tests
│   ├── unit/                   # Unit tests (pytest)
│   │   ├── test_brat_loader.py
│   │   ├── test_preprocess.py
│   │   ├── test_prompts.py
│   │   └── test_metrics.py
│   └── integration/            # Integration tests (AI runs these)
│       ├── phase1_test_brat_loader.py
│       ├── phase2_test_prompts.py
│       ├── phase2_test_model_loading.py
│       └── ...
└── scripts/                    # Utility scripts
    ├── process_full_dataset.py
    ├── cleanup_old_models.sh
    └── run_smoke.sh
```

---

## Design Principles

### Code Structure & Cleanliness

**Repository Organization:**
- All code must follow the repository structure defined above
- New files must be placed in appropriate directories (src/, tests/, notebooks/, scripts/)
- No accidental files should be committed (check `.gitignore` before committing)

**File Naming:**
- Use descriptive, lowercase names with underscores (e.g., `test_metrics.py`, not `testMetrics.py` or `test-metrics.py`)
- Avoid special characters in filenames (especially `=`, `>`, `<`, `|` which can be created by command redirection errors)
- All files must have clear purpose documented in roadmap

**Git Hygiene:**
- Always review `git status` before committing
- Check for accidental files (files with weird names like `=0.17`, `=10`, etc.)
- Use `.gitignore` to prevent committing:
  - Build artifacts (`__pycache__/`, `*.pyc`, `*.egg-info/`)
  - Accidental files from command errors (`=*`, `>`, `<`, etc.)
  - Experiment outputs that shouldn't be versioned (check `experiments/` patterns)
- Before pushing, verify all changes align with roadmap structure

**Code Quality:**
- Follow existing code patterns and structure
- Update roadmap when adding new files or major changes
- Keep code clean and well-documented
- Remove temporary files and unused code

### Model Selection
- **Primary Model**: Llama-3.1-8B-Instruct
- **Rationale**: Instruction-strong, fits available GPUs (bf16 on 32GB; 4-bit on 24GB)
- **Compatibility**: Aligns with paper if fine-tuning later

### Data Pipeline
- Single data loader and preprocessor (no duplicate pipes)
- Keep only SHAC social-history sections
- Link to MIMIC only for demographics if needed

### Evaluation Metrics

#### Primary Metric: False Positive Rate (FPR)

Per paper definition, FPR is computed using binary risk grouping:

**Classification Space:**
- Model outputs: 3 labels (none, current, past)

**Evaluation Grouping:**
- **Negative class** (no drug use): `none` or `Not Applicable`
- **Positive class** (drug use): `current` or `past`

**FPR Calculation:**
```
FPR = FP / (FP + TN)
```
Where:
- **FP** = predicted positive (current/past) when truth is negative (none/Not Applicable)
- **TN** = predicted negative (none/Not Applicable) when truth is negative (none/Not Applicable)
- **Only computed on samples where ground truth is negative**

**Interpretation:**
- Measures: How often we incorrectly claim drug use when patient has none
- Critical for safety: Minimize false alarms about drug use
- Lower is better (target: <15% for safety)

#### Secondary Metrics
- **Accuracy**: Overall 3-class accuracy (for completeness)
- **Per-class F1**: Precision/Recall for each class
- **Confusion Matrix**: Full 3×3 matrix

#### Excluded Metrics
- **ROC/AUC**: Not meaningful for 3-class with asymmetric costs

### Prompt Strategy
- Keep prompts terse and direct
- No Chain-of-Thought (CoT)
- No explicit warnings
- Adversarial pressure provided by multi-agent architecture

---

## Testing Philosophy

We use a **dual-testing approach** to ensure rigor and transparency:

### 1. Interactive Notebooks (`notebooks/`) 
**Purpose**: For **USER** to run and validate

- **Format**: Jupyter notebooks (`.ipynb`)
- **Who runs**: User runs these interactively
- **Purpose**: 
  - Verify each cell's output makes sense
  - See actual data and intermediate results
  - Validate on sample data (3-10 examples)
  - Understand what the code is doing
- **When**: Before committing to next phase
- **Example**: `notebooks/04_test_model_loading.ipynb`

### 2. Integration Test Scripts (`tests/integration/`)
**Purpose**: For **AI** to rigorously validate

- **Format**: Python scripts (`.py`)
- **Who runs**: AI runs these before claiming something works
- **Purpose**:
  - Automatically verify code works end-to-end
  - Test on actual data (not mocked)
  - Catch errors before user sees them
  - Provide honest pass/fail results
- **When**: Before creating notebooks or claiming completion
- **Example**: `tests/integration/phase2_test_model_loading.py`

### 3. Unit Tests (`tests/unit/`)
**Purpose**: For regression testing

- **Format**: pytest tests
- **Who runs**: Both (during development and CI/CD)
- **Purpose**: Test individual functions in isolation

### Testing Workflow

For each phase:

```
1. AI creates integration test script (tests/integration/phaseN_*.py)
2. AI runs the script and verifies it passes
3. AI creates interactive notebook (notebooks/NN_*.ipynb) based on working script
4. User runs notebook and validates outputs
5. ✅ Only then proceed to next phase
```

**Key Rule**: **AI MUST run integration tests and HONESTLY report results** before claiming anything works.

---

## Implementation Roadmap - Phased Approach

Each phase includes:
- **Production modules** (`.py`) - For full dataset processing
- **Integration test script** (`.py`) - AI runs this FIRST to validate
- **Interactive notebook** (`.ipynb`) - User runs this to verify
- **Validation criteria** - Checklist before proceeding

---

### Phase 0: Repository Setup

#### Deliverables
- Project structure and dependencies
- Configuration files
- Basic utilities

#### Tasks
1. Create directory structure
2. Set up `pyproject.toml` with dependencies
3. Initialize configuration files (data.yaml, baseline.yaml, agentic.yaml)
4. Set up basic utilities (logger, seed, io)

#### Test Notebook
- `notebooks/00_setup_check.ipynb`
  - Verify imports work
  - Check GPU availability
  - Test logger and seed utilities
  - Verify directory structure

#### Validation Criteria
- ✅ All dependencies install without errors
- ✅ GPU detected and accessible
- ✅ All config files load properly
- ✅ Logger writes test logs successfully

---

### Phase 1: Data Loading & Processing

**Goal**: Load and preprocess SHAC BRAT annotations for Drug StatusTime classification

#### Production Modules

##### 1. Configuration (`configs/data.yaml`)
```yaml
raw_root: "/home/amin/Dropbox/code/SDOH/Track_2_SHAC/SHAC"
sources: ["mimic", "uw"]   # Use both MIMIC and UW datasets
splits: ["train", "dev", "test"]
target_event: "Drug"
status_labels: ["none", "current", "past", "Not Applicable"]
```

##### 2. BRAT Loader (`src/utils/brat_loader.py`)
- Parse `.txt`/`.ann` files → structured data for Drug events
- **Output Schema**:
  ```json
{
  "id": "m_train_000123_drug_1",
  "split": "train",
    "source": "mimic" | "uw",
  "note_id": "...",
  "text": "<full social-history text>",
  "trigger_text": "drug",
  "status_label": "current" | "past" | "none" | "Not Applicable"
}
  ```
- **BRAT Mapping**:
  - Event (E) links Alcohol|Drug|Tobacco trigger (T) to Status arg (T)
  - Attribute (A) StatusTimeVal holds categorical label
  - **Filter**: Keep only `event_type == "Drug"`
  - **Source tracking**: Preserve whether sample is from MIMIC or UW

##### 3. Preprocessor (`src/utils/preprocess.py`)
- Input: List of dicts from `brat_loader`
- Clean text minimally (normalize whitespace; preserve semantics)
- Create processed JSONL per split: `data/processed/{split}.jsonl`
- Optional: Ensure patient-level non-leak splits if patient IDs exist

##### 4. Unit Tests (`tests/test_brat_loader.py`)
- Feed 2–3 tiny `.txt`/`.ann` fixtures
- Assert one row per Drug trigger and correct status mapping

#### Test Notebooks

##### `notebooks/01_test_brat_loader.ipynb`
**Purpose**: Validate BRAT parsing on sample files

**Tests**:
1. Load 2-3 sample `.txt`/.`ann` files from each split
2. Display raw BRAT annotations
3. Show parsed output (id, text, trigger, label)
4. Verify label distribution
5. Check for parsing errors

**Expected Output**:
```python
# Sample output
{
  'id': 'm_train_000123_drug_1',
  'text': 'Patient has history of cocaine use...',
  'trigger_text': 'cocaine',
  'status_label': 'past'
}
```

##### `notebooks/02_test_preprocess.ipynb`
**Purpose**: Validate preprocessing pipeline

**Tests**:
1. Load sample parsed data
2. Show text cleaning (before/after)
3. Display processed JSONL format
4. Verify data integrity (no lost samples)
5. Check for any edge cases (empty text, special characters)

##### `notebooks/03_data_eda.ipynb`
**Purpose**: Exploratory data analysis on full dataset

**Analysis**:
1. Label counts and distribution per split
2. Text length statistics
3. **FPR-relevant class ratios**:
   - Negative class proportion: `none` + `Not Applicable` (no drug use)
   - Positive class proportion: `current` + `past` (drug use)
   - Important for understanding FPR denominator
4. Sample examples from each class
5. Data quality checks (duplicates, missing values)
6. **MIMIC vs UW comparison**:
   - Label distribution differences
   - Text length differences
   - Any domain-specific patterns

#### Validation Criteria
- ✅ BRAT loader successfully parses all sample files
- ✅ All Drug events extracted with correct StatusTime labels
- ✅ Preprocessing runs without errors on full dataset
- ✅ Output JSONL files created for train/dev/test splits
- ✅ Label distribution matches expectations
- ✅ No data leakage between splits
- ✅ Text preserved with minimal cleaning

**📊 Expected Stats**:
- Train samples: ~XXX Drug events (MIMIC + UW)
- Dev samples: ~XXX Drug events (MIMIC + UW)
- Test samples: ~XXX Drug events (MIMIC + UW)
- MIMIC vs UW split: Document in EDA notebook
- Class distribution: Document in EDA notebook per source

---

### Phase 2: Baseline Model (Single-Agent Predictor)

**Goal**: Implement single-model baseline for Drug StatusTime classification

#### Production Modules

##### 1. Configuration (`configs/baseline.yaml`)
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

##### 2. Prompt Template (`src/agentic/prompts.py`)
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

##### 3. Inference Engine (`src/baselines/llama_single.py`)
- Load Llama-3.1-8B-Instruct model
- Batched inference on processed data
- Strict option parsing: `a|b|c|d` → status label
- Save to `experiments/baseline/preds.jsonl` with fields: `id`, `pred`, `gold`, `raw_text`

##### 4. Baseline Runner (`src/evaluation/run_baseline.py`)
1. Load `data/processed/{split}.jsonl`
2. Assemble prompts with template
3. Run `llama_single.py` inference
4. Write predictions with timestamps

##### 5. Metrics Module (`src/evaluation/metrics.py`)
- Map letters → labels
- Compute metrics per paper definition:
  - **Accuracy**: Overall 3-class classification accuracy
  - **FPR**: False Positive Rate (primary metric)
    - Negative class: `none` OR `Not Applicable` (no drug use)
    - Positive class: `current` OR `past` (drug use)
    - FPR = FP / (FP + TN)
    - FP = predicted positive when truth is negative
    - TN = predicted negative when truth is negative
    - Computed ONLY on samples where ground truth is negative
  - **Per-class metrics**: Precision, Recall, F1 for each class
  - **Confusion matrix**: Full 3×3 matrix

##### 6. Unit Tests (`tests/test_baseline_prompt.py`)
- Verify correct letter extraction and label mapping
- Test prompt formatting with edge cases

#### Test Notebooks

##### `notebooks/04_test_model_loading.ipynb`
**Purpose**: Verify model loads and runs inference

**Tests**:
1. Load Llama-3.1-8B-Instruct model
2. Check GPU memory usage
3. Test inference on 1-2 sample prompts
4. Verify output format (a|b|c|d)
5. Test with different temperature settings
6. Measure inference speed

**Expected Output**:
```python
# Model loaded successfully
# GPU Memory: 15.2 GB / 32 GB
# Sample input: "Patient denies drug use..."
# Model output: "a"
# Parsed label: "none"
# Inference time: 0.3s per sample
```

##### `notebooks/05_test_baseline_inference.ipynb`
**Purpose**: Test baseline on 5-10 samples from each class

**Tests**:
1. Load 5-10 samples from each label class
2. Run baseline inference
3. Display input (note + trigger)
4. Display model output and parsed prediction
5. Compare prediction vs. gold label
6. Show raw model logits/probabilities (if available)

**Expected Output**:
```python
# Sample 1:
# Text: "Patient has history of cocaine abuse but clean for 5 years..."
# Trigger: "cocaine"
# Gold: "past"
# Prediction: "c" → "past" ✓
# Confidence: 0.89
```

##### `notebooks/06_baseline_full_eval.ipynb`
**Purpose**: Evaluate baseline on full dev/test set

**Analysis**:
1. Run baseline on full dev set
2. Compute accuracy and FPR
3. Show confusion matrix
4. Analyze errors by class
5. Show examples of correct/incorrect predictions
6. Compare performance across splits

#### Validation Criteria
- ✅ Model loads successfully on GPU
- ✅ Inference runs without errors on sample data
- ✅ Output parsing (a|b|c|d → labels) works correctly
- ✅ Baseline achieves reasonable accuracy on dev set (>60%)
- ✅ FPR metric computed correctly
- ✅ Predictions saved in correct format
- ✅ All edge cases handled (empty text, multiple triggers, etc.)

**📊 Expected Performance**:
- Dev Accuracy: 60-75%
- Dev FPR: 15-30%
- Test Accuracy: Document after validation
- Inference Speed: ~0.3s per sample (bf16) or ~0.1s (4-bit)

---

### Phase 3: Agentic Pipeline (Proposer / Refuter / Judge)

**Goal**: Implement multi-agent system with adversarial reasoning to reduce FPR

#### Production Modules

##### 1. Data Interfaces (`src/agentic/interfaces.py`)
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

##### 2. Sectionizer (`src/utils/sectionizer.py`)
- Heuristic split into: `assessment_plan`, `problems`, `meds`, `labs`, `other`
- Build `non_cue_text` = concat(`assessment_plan`, `problems`, `meds`, `labs`)
- Build `masked_note` = note with sentence containing TRIGGER removed

##### 3. Agent Prompts

**Proposer** (`src/agentic/prompts.py` - `proposer_v1`):
```
System: You classify Drug StatusTime in clinical notes.
User:
Note:
{NOTE}

Drug trigger: "{TRIGGER}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter only.
```

**Refuter** (`refuter_v1`):
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

**Judge** (`judge_v1`):
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

##### 4. Proposer Agent (`src/agentic/proposer.py`)
- Run `proposer_v1` on full NOTE
- Compute `P_MASKED` by running on `masked_note`
- Save `proposer.jsonl`: `id`, `proposer_letter`, `proposer_masked_letter`

##### 5. Refuter Agent (`src/agentic/refuter.py`)
- Input: `non_cue_text` + `proposer_letter`
- Output: `refuter_letter` and up to two quoted spans
- Parse and validate spans from output
- Save `refuter.jsonl`

##### 6. Judge Agent (`src/agentic/judge.py`)
- Input: proposer/refuter outputs + `non_cue_text`
- Apply decision rules
- Default to `(a) none` if ambiguous (per rule 2)
- Save `final.jsonl`

##### 7. Pipeline Orchestrator (`src/agentic/pipeline.py`)
- Orchestrates batched runs through all three agents
- Retries on parsing failures (max 3 attempts)
- Caches prompts→outputs in `experiments/agentic/cache/`
- Handles edge cases (empty spans, invalid letters)

##### 8. Configuration (`configs/agentic.yaml`)
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

##### 9. Integration Tests

**Basic Functionality Test** (`tests/integration/phase3_test_agentic_pipeline.py`):
- Fixture: 3 tiny notes covering each class
- Assert pipeline returns one of `a|b|c|d` for each
- Snapshot JSON of results for regression testing

**Performance Comparison Test** (`tests/integration/phase3_test_agentic_vs_baseline.py`):
- Load 20-50 samples from dev set (average batch size)
- Run baseline on same samples
- Run agentic pipeline on same samples
- Compare metrics:
  - FPR (primary metric - should be lower for agentic)
  - Accuracy (should remain competitive, within 5% of baseline)
  - Per-class performance
- Assert agentic FPR < baseline FPR (or at least not significantly worse)
- Assert agentic accuracy within acceptable range of baseline
- Exit with error if agentic performance is worse than expected

#### Test Notebooks

##### `notebooks/07_test_sectionizer.ipynb`
**Purpose**: Validate text sectionization and masking

**Tests**:
1. Load 3-5 sample notes
2. Display original text
3. Show section splits (assessment_plan, problems, meds, labs, other)
4. Display `non_cue_text` (concatenated sections)
5. Display `masked_note` (trigger sentence removed)
6. Verify masking works correctly (trigger not in masked text)

**Expected Output**:
```python
# Original: "Patient reports cocaine use daily. Assessment: active substance abuse..."
# Sections:
#   - problems: "active substance abuse"
#   - assessment_plan: "refer to addiction services"
# Non-cue text: "active substance abuse refer to addiction services"
# Masked note: "Assessment: active substance abuse..." (trigger sentence removed)
```

##### `notebooks/08_test_proposer.ipynb`
**Purpose**: Test Proposer agent on samples

**Tests**:
1. Load 5-10 samples across all classes
2. Run Proposer on full note
3. Run Proposer on masked note
4. Display: text → proposer_letter, masked_text → proposer_masked_letter
5. Check if Proposer flips decision when trigger masked
6. Analyze flip patterns by class

**Expected Output**:
```python
# Sample 1:
# Text: "Patient admits daily cocaine use..."
# Proposer (full): "b" (current)
# Proposer (masked): "a" (none)
# → Proposer flips: True (relies on trigger)
```

##### `notebooks/09_test_refuter.ipynb`
**Purpose**: Test Refuter agent on samples

**Tests**:
1. Load samples with Proposer outputs
2. Run Refuter with non_cue_text
3. Display: proposer_letter → refuter_letter + spans
4. Verify spans exist in non_cue_text
5. Check if Refuter challenges appropriately
6. Analyze span quality and relevance

**Expected Output**:
```python
# Sample 1:
# Non-cue text: "Assessment shows active substance use disorder..."
# Proposer: "a" (none)
# Refuter: "b" (current)
# Spans: ["active substance use disorder"]
# → Refuter challenges with evidence from non-cue text
```

##### `notebooks/10_test_judge.ipynb`
**Purpose**: Test Judge decision-making

**Tests**:
1. Load samples with Proposer + Refuter outputs
2. Run Judge with all inputs
3. Display decision process:
   - Proposer vs Refuter letters
   - Refuter spans
   - Proposer flip status (masked)
   - Judge final decision
4. Verify Judge follows rules correctly
5. Show cases for each rule (1, 2, 3)

**Expected Output**:
```python
# Sample 1:
# Proposer: "a", Proposer (masked): "a"
# Refuter: "b", Spans: ["active use noted"]
# Judge applies Rule 1: Prefer Refuter (has span evidence)
# Judge decision: "b" (current)
```

##### `notebooks/11_test_full_pipeline.ipynb`
**Purpose**: Test end-to-end agentic pipeline on samples

**Tests**:
1. Load 10-15 samples across all classes
2. Run full pipeline: Proposer → Refuter → Judge
3. Display full trace for each sample:
   - Input text + trigger
   - Proposer decision
   - Refuter challenge + spans
   - Judge final verdict
   - Gold label
4. Compare: Baseline vs Agentic vs Gold
5. Identify cases where agentic improves over baseline

**Expected Output**:
```python
# Sample 1:
# Text: "Patient denies current drug use. History notes past cocaine dependence."
# Trigger: "cocaine"
# Gold: "past"
# Baseline: "a" (none) ✗
# ---
# Proposer: "a" (none)
# Refuter: "c" (past), Span: ["past cocaine dependence"]
# Judge: "c" (past) ✓
# → Agentic corrects baseline error
```

##### `notebooks/12_agentic_full_eval.ipynb`
**Purpose**: Evaluate agentic pipeline on full dev/test set

**Analysis**:
1. Run full agentic pipeline on dev set
2. Compute accuracy and FPR
3. Compare: Baseline vs Agentic
4. Show FPR reduction (primary goal)
5. Confusion matrices (baseline vs agentic)
6. Error analysis: Where does agentic help? Where does it hurt?
7. Analyze by label class

#### Validation Criteria
- ✅ Sectionizer correctly splits notes into components
- ✅ Masked notes remove trigger sentences
- ✅ Proposer runs on both full and masked notes
- ✅ Proposer flips detected correctly
- ✅ Refuter extracts valid spans from non_cue_text
- ✅ Refuter challenges appropriately
- ✅ Judge applies rules correctly
- ✅ Pipeline handles all edge cases without errors
- ✅ Agentic reduces FPR compared to baseline
- ✅ Accuracy remains competitive (within 5% of baseline)

**📊 Expected Performance**:
- Dev FPR: 5-15% (vs baseline 15-30%) → **50% FPR reduction**
- Dev Accuracy: 60-75% (similar to baseline)
- Proposer flip rate: 20-40% of samples
- Refuter challenge rate: 30-50% of samples

---

### Phase 4: Comparative Evaluation & Analysis

**Goal**: Compare baseline vs agentic performance with focus on FPR reduction

#### Production Modules

##### 1. Agentic Runner (`src/evaluation/run_agentic.py`)
- Run full agentic pipeline on dev/test splits
- Save to `experiments/agentic/<run_id>/final.jsonl`
- Log timing and resource usage
- Handle errors gracefully with fallback to baseline

##### 2. Comparison Tool (`src/evaluation/compare_runs.py`)
- Load baseline vs agentic predictions/metrics from JSON files
- Compute comprehensive deltas:
  - FPR deltas (absolute and relative %)
  - Accuracy deltas (absolute and relative %)
  - Per-class performance (F1, precision, recall)
- Auto-find latest runs or accept specific run directories
- Output multiple formats:
  - CSV: Detailed metrics table (`experiments/reports/comparison_*.csv`)
  - Markdown: Formatted report with tables (`experiments/reports/comparison_*.md`)
  - JSON: Structured data for programmatic use (`experiments/reports/comparison_*.json`)
- Highlight FPR improvements (primary metric)
- Print detailed comparison report to console

##### 3. Full Comparison Runner (`src/evaluation/run_full_comparison.py`)
- Orchestrates complete comparison pipeline:
  1. Run baseline on full dev/test splits (with timing tracking)
  2. Run agentic on full dev/test splits (with timing tracking)
  3. Automatically compare results
  4. Generate comprehensive comparison reports
- Supports skipping steps (e.g., if baseline already run)
- Provides big batch quantitative comparison
- Usage: `python -m src.evaluation.run_full_comparison --split dev test`

##### 4. Comprehensive Text Reports (`src/evaluation/compare_runs.py`)
- Generates detailed text reports saved in `results/` folder
- Report includes:
  - Experiment metadata (date, run IDs, split)
  - Data description (sources, label distribution, sample counts)
  - Timing information (inference time, time per sample, speedup factor)
  - Overall metrics (accuracy, FPR with relative changes)
  - Per-class metrics (F1, precision, recall for each label)
  - Detailed metrics breakdown (macro averages)
- File naming: `results/comparison_report_{split}_{timestamp}.txt`
- Also generates CSV, Markdown, and JSON reports in `experiments/reports/`

##### 5. Visualization (`src/evaluation/plots.py`)
- Matplotlib bar plots for FPR by configuration
- Confusion matrix heatmaps
- Per-class performance comparisons
- No seaborn dependency (use matplotlib only)

#### Test Notebooks

##### `notebooks/13_compare_baseline_vs_agentic.ipynb`
**Purpose**: Compare performance on sample data

**Analysis**:
1. Load predictions from baseline and agentic on 20-30 samples
2. Side-by-side comparison:
   - Text + trigger
   - Baseline prediction
   - Agentic prediction (with trace)
   - Gold label
3. Highlight improvements (baseline wrong, agentic correct)
4. Highlight regressions (baseline correct, agentic wrong)
5. Compute sample-level metrics

**Expected Output**:
```python
# Improvements: 5/30 samples
# Regressions: 1/30 samples
# Sample improvement:
# Text: "Patient denies current use but used heroin in the past"
# Baseline: "b" (current) ✗
# Agentic: "c" (past) ✓
# Refuter span: "used heroin in the past"
```

##### `notebooks/14_fpr_analysis.ipynb`
**Purpose**: Deep dive into FPR reduction

**Analysis**:
1. Load all predictions where gold ∈ {none, Not Applicable}
2. Identify false positives:
   - Baseline FPs
   - Agentic FPs
3. Show FPs corrected by agentic system
4. Show new FPs introduced by agentic
5. Analyze patterns in FP corrections:
   - Role of Refuter spans
   - Role of Proposer flips
   - Judge decision patterns
6. Calculate FPR by subgroups:
   - **By source**: MIMIC vs UW
   - By label class
   - By text length

**Expected Output**:
```python
# Baseline FPs: 45/150 → FPR = 30%
# Agentic FPs: 18/150 → FPR = 12%
# FPR reduction: 60% relative reduction
# 
# FPs corrected: 27 samples
# New FPs: 0 samples
# Most common pattern: Refuter found evidence in non-cue text
```

##### `notebooks/15_error_analysis.ipynb`
**Purpose**: Analyze remaining errors and failure modes

**Analysis**:
1. Identify all errors (baseline and agentic)
2. Categorize errors:
   - Ambiguous text
   - Missing context
   - Parsing failures
   - Agent disagreement
3. Show examples from each category
4. Identify opportunities for improvement
5. Analyze per-class error patterns

**Expected Output**:
```python
# Agentic errors: 15/100 samples
# Error categories:
#   - Ambiguous text: 8 samples
#   - Missing context: 4 samples
#   - Refuter wrong challenge: 3 samples
# 
# Example (ambiguous):
# Text: "Patient reports substance issues"
# Issue: "substance" could be drug/alcohol, unclear timeframe
```

##### `notebooks/16_ablation_study.ipynb`
**Purpose**: Understand contribution of each agent

**Analysis**:
1. Compare configurations:
   - Baseline (single model)
   - Proposer only
   - Proposer + Refuter (no Judge)
   - Full triad (Proposer + Refuter + Judge)
2. Measure FPR and accuracy for each
3. Identify critical components
4. Analyze: When does Refuter help? When does Judge help?

**Expected Output**:
```python
# Configuration | Accuracy | FPR
# Baseline      | 70%      | 30%
# Proposer      | 70%      | 30%  (same as baseline)
# Prop+Ref      | 68%      | 18%  (FPR↓, Acc↓ slightly)
# Full Triad    | 70%      | 12%  (FPR↓, Acc maintained)
# 
# → Judge is critical for maintaining accuracy while reducing FPR
```

#### Validation Criteria
- ✅ Comparison runs successfully on full dataset
- ✅ FPR reduction achieved (target: >30% relative reduction)
- ✅ Accuracy maintained (within 5% of baseline)
- ✅ Statistical significance of improvements verified
- ✅ Error analysis completed and documented
- ✅ Ablation study shows value of each component
- ✅ Reports generated and saved

**📊 Expected Results**:
- **FPR**: Baseline 15-30% → Agentic 5-15% (50% relative reduction)
- **Accuracy**: Maintained within 2-5% of baseline
- **Primary benefit**: Reducing false positives (safety critical)

---

### Phase 5: Testing & Quality Assurance

**Goal**: Comprehensive testing infrastructure and code quality

#### Production Modules

##### 1. Unit Tests
- `tests/test_brat_loader.py`: BRAT parsing
- `tests/test_preprocess.py`: Data preprocessing
- `tests/test_prompts.py`: Prompt formatting
- `tests/test_baseline_prompt.py`: Baseline inference
- `tests/test_agentic_pipeline.py`: Full pipeline
- `tests/test_metrics.py`: Evaluation metrics

##### 2. Smoke Test (`tests/run_smoke.sh`)
End-to-end smoke test:
```bash
#!/bin/bash
# 1. Test data loading (3 samples)
# 2. Test baseline on samples
# 3. Test agentic on samples
# 4. Test metrics computation
# 5. Verify outputs exist
```

##### 3. Dependencies (`pyproject.toml`)
```toml
[project]
name = "agentic_shac"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "transformers>=4.30",
    "accelerate",
    "pandas",
    "numpy",
    "pyyaml",
    "tqdm",
    "matplotlib",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "black", "mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.ruff]
line-length = 100
```

#### Test Notebooks

##### `notebooks/17_integration_test.ipynb`
**Purpose**: End-to-end integration test

**Tests**:
1. Run smoke test on 5 samples
2. Verify: Data loading → Baseline → Agentic → Metrics
3. Check all outputs created correctly
4. Verify no errors or warnings
5. Test edge cases (empty text, long text, special characters)

#### Validation Criteria
- ✅ All unit tests pass
- ✅ Smoke test completes successfully
- ✅ Code coverage >80% on critical paths
- ✅ No linter errors (ruff, black)
- ✅ Type checking passes (mypy)
- ✅ Integration test passes on edge cases

---

### Phase 6: Experiment Management & Reproducibility

**Goal**: Robust experiment tracking and reproducibility

#### Production Modules

##### 1. Logger (`src/utils/logger.py`)
```python
class ExperimentLogger:
    """
    - Writes JSON logs per run
    - Tracks timing and resource usage
    - Saves config and prompts
    """
    def log_experiment(self, run_id, config, results):
        # experiments/<run_id>/config.json
        # experiments/<run_id>/log.json
        # experiments/<run_id>/system_prompts.txt
```

##### 2. Seed Manager (`src/utils/seed.py`)
```python
def set_seed(seed: int):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

##### 3. Config Manager (`src/utils/config.py`)
- Load YAML configs
- Validate config schemas
- Merge configs with CLI arguments

#### Test Notebooks

##### `notebooks/18_experiment_tracking.ipynb`
**Purpose**: Test experiment logging and tracking

**Tests**:
1. Run 2-3 mini experiments
2. Verify logs created with correct format
3. Test config loading and validation
4. Verify reproducibility with same seed
5. Compare outputs from runs with same seed

#### Validation Criteria
- ✅ All experiments logged correctly
- ✅ Config validation works
- ✅ Seeds produce reproducible results
- ✅ Easy to compare different runs
- ✅ All metadata captured (model, data, hyperparams)

---

### Phase 7: Final Integration & Documentation

**Goal**: Polish, documentation, and deployment readiness

#### Deliverables

##### 1. Optional CLI (`src/cli.py`)
```bash
# Run baseline
python -m src.cli baseline --split test --config configs/baseline.yaml

# Run agentic pipeline
python -m src.cli agentic --split test --config configs/agentic.yaml

# Evaluate and compare
python -m src.cli evaluate --baseline_dir experiments/baseline \
                           --agentic_dir experiments/agentic
```

##### 2. README.md
- Project overview
- Installation instructions
- Quick start guide
- Usage examples
- Results summary
- Citation

##### 3. Documentation
- API documentation
- Configuration guide
- Troubleshooting
- Development guide

#### Test Notebooks

##### `notebooks/19_final_results.ipynb`
**Purpose**: Generate final results and visualizations

**Output**:
1. Final performance tables (baseline vs agentic)
2. Publication-ready figures
3. Statistical significance tests
4. Key findings summary
5. Recommendations for future work

#### Validation Criteria
- ✅ CLI works for all major commands
- ✅ README is comprehensive and accurate
- ✅ All notebooks run without errors
- ✅ Final results documented and reproducible
- ✅ Code is clean and well-documented
- ✅ Ready for publication/sharing

---

## Notebook Summary

All test notebooks for incremental validation:

### Phase 0: Setup
- `00_setup_check.ipynb` - Verify environment and dependencies

### Phase 1: Data Pipeline
- `01_test_brat_loader.ipynb` - Validate BRAT parsing
- `02_test_preprocess.ipynb` - Validate preprocessing
- `03_data_eda.ipynb` - Exploratory data analysis

### Phase 2: Baseline
- `04_test_model_loading.ipynb` - Test model loading
- `05_test_baseline_inference.ipynb` - Test baseline on samples
- `06_baseline_full_eval.ipynb` - Full baseline evaluation

### Phase 3: Agentic Pipeline
- `07_test_sectionizer.ipynb` - Test text sectionization
- `08_test_proposer.ipynb` - Test Proposer agent
- `09_test_refuter.ipynb` - Test Refuter agent
- `10_test_judge.ipynb` - Test Judge agent
- `11_test_full_pipeline.ipynb` - End-to-end agentic test
- `12_agentic_full_eval.ipynb` - Full agentic evaluation

### Phase 4: Analysis
- `13_compare_baseline_vs_agentic.ipynb` - Side-by-side comparison
- `14_fpr_analysis.ipynb` - FPR reduction analysis
- `15_error_analysis.ipynb` - Error patterns
- `16_ablation_study.ipynb` - Component contribution

### Phase 5-7: Integration
- `17_integration_test.ipynb` - End-to-end integration
- `18_experiment_tracking.ipynb` - Experiment management
- `19_final_results.ipynb` - Final results and figures

**Total: 19 notebooks** for comprehensive testing and validation

---

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (32GB for bf16, 24GB for 4-bit)
- Access to SHAC dataset (Track_2_SHAC)

### Installation

```bash
# Clone repository
git clone git@github.com:ninjutsoo/agentic_shac.git
cd agentic_shac

# Create conda environment (use 'temp' as per workspace rules)
conda create -n temp python=3.10
conda activate temp

# Install dependencies
pip install -e .
```

### Development Workflow

**Follow phases sequentially - validate each before proceeding!**

#### Phase 0: Setup (Day 1)
```bash
# 1. Set up project structure
mkdir -p src/{utils,baselines,agentic,evaluation} tests configs data/{raw,processed} experiments notebooks

# 2. Create pyproject.toml with dependencies
# 3. Run: notebooks/00_setup_check.ipynb
# ✅ Verify all dependencies and GPU work
```

#### Phase 1: Data Pipeline (Days 2-3)
```bash
# 1. Implement: src/utils/{brat_loader.py, preprocess.py}
# 2. Run: notebooks/01_test_brat_loader.ipynb
# ✅ Verify BRAT parsing on 3 samples

# 3. Run: notebooks/02_test_preprocess.ipynb
# ✅ Verify preprocessing on samples

# 4. Process full dataset
python -m src.utils.preprocess

# 5. Run: notebooks/03_data_eda.ipynb
# ✅ Verify full dataset statistics
```

#### Phase 2: Baseline (Days 4-6)
```bash
# 1. Implement: src/agentic/prompts.py, src/baselines/llama_single.py
# 2. Run: notebooks/04_test_model_loading.ipynb
# ✅ Verify model loads and runs

# 3. Run: notebooks/05_test_baseline_inference.ipynb
# ✅ Test on 5-10 samples per class

# 4. Implement: src/evaluation/{run_baseline.py, metrics.py}
# 5. Run baseline on full dev set
python -m src.evaluation.run_baseline --split dev

# 6. Run: notebooks/06_baseline_full_eval.ipynb
# ✅ Verify baseline metrics (Acc >60%, FPR <30%)
```

#### Phase 3: Agentic Pipeline (Days 7-12)
```bash
# 1. Implement: src/utils/sectionizer.py
# 2. Run: notebooks/07_test_sectionizer.ipynb
# ✅ Verify sectionization on 5 samples

# 3. Implement: src/agentic/{interfaces.py, proposer.py}
# 4. Run: notebooks/08_test_proposer.ipynb
# ✅ Test Proposer on samples

# 5. Implement: src/agentic/refuter.py
# 6. Run: notebooks/09_test_refuter.ipynb
# ✅ Test Refuter on samples

# 7. Implement: src/agentic/judge.py
# 8. Run: notebooks/10_test_judge.ipynb
# ✅ Test Judge on samples

# 9. Implement: src/agentic/pipeline.py
# 10. Run: notebooks/11_test_full_pipeline.ipynb
# ✅ Test full pipeline on 10-15 samples

# 11. Run agentic on full dev set
python -m src.evaluation.run_agentic --split dev

# 12. Run: notebooks/12_agentic_full_eval.ipynb
# ✅ Verify agentic metrics (FPR reduction >30%)
```

#### Phase 4: Analysis (Days 13-15)
```bash
# 1. Implement: src/evaluation/{compare_runs.py, run_full_comparison.py, plots.py}
# 2. Run full comparison on dev/test splits
python -m src.evaluation.run_full_comparison --split dev test
# ✅ Generates comprehensive comparison reports (CSV, Markdown, JSON)

# 3. Or compare existing runs manually
python -m src.evaluation.compare_runs --split dev
python -m src.evaluation.compare_runs --split test

# 4. Run: notebooks/13_compare_baseline_vs_agentic.ipynb
# ✅ Side-by-side comparison on samples

# 5. Run: notebooks/14_fpr_analysis.ipynb
# ✅ Analyze FPR reduction

# 6. Run: notebooks/15_error_analysis.ipynb
# ✅ Categorize errors

# 7. Run: notebooks/16_ablation_study.ipynb
# ✅ Understand component contributions
```

#### Phase 5-7: Polish & Document (Days 16-18)
```bash
# 1. Write unit tests, run smoke test
pytest tests/

# 2. Run: notebooks/17_integration_test.ipynb
# ✅ End-to-end integration

# 3. Run: notebooks/18_experiment_tracking.ipynb
# ✅ Verify experiment logging

# 4. Generate final results
# Run: notebooks/19_final_results.ipynb
# ✅ Publication-ready figures and tables

# 5. Write README.md and documentation
```

### Quick Run (After Implementation)

```bash
# Full pipeline
conda activate temp

# 1. Prepare data
python -m src.utils.preprocess

# 2. Run baseline
python -m src.evaluation.run_baseline --split dev --split test

# 3. Run agentic pipeline
python -m src.evaluation.run_agentic --split dev --split test

# 4. Compare and generate comprehensive reports (recommended - runs everything)
python -m src.evaluation.run_full_comparison --split dev test

# Or compare existing runs manually:
# python -m src.evaluation.compare_runs \
    --baseline experiments/baseline \
    --agentic experiments/agentic \
    --output experiments/reports/
```

---

## Future Enhancements

After successful Phase 7 completion, consider:

### 1. Few-Shot In-Context Learning
- Add 3-shot examples to prompts (mirror paper variants)
- Test with different example selection strategies
- Compare 0-shot vs 3-shot vs 5-shot

### 2. Fine-Tuning
- LoRA fine-tune Llama-3.1-8B on SHAC training data
- Use paper hyperparameters
- Compare fine-tuned vs prompt-based

### 3. Extended Event Types
- Add Alcohol StatusTime classification
- Add Tobacco StatusTime classification
- Multi-task learning across all three substances

### 4. Cross-Dataset Evaluation
- Analyze MIMIC vs UW performance differences
- Cross-dataset evaluation (train MIMIC, test UW and vice versa)
- Domain adaptation techniques for cross-dataset generalization

### 5. Advanced Agentic Features
- Dynamic agent selection (when to use Refuter?)
- Confidence-based routing
- Multi-round debate between agents
- Memory/context sharing across samples

### 6. Deployment
- FastAPI service for real-time inference
- Batch processing optimization
- Model distillation for faster inference
- Edge deployment considerations

---

## License

[Add appropriate license]

## Citation

If you use this code, please cite the original SHAC paper:

```bibtex
[Add SHAC paper citation]
```

## Contact

For questions or issues, please open a GitHub issue or contact [your contact info].

---

**Last Updated**: October 2025
**Version**: 0.1.0