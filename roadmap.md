Use only the SHAC social-history sections and focus on Drug StatusTime classification with triggers given. Use the MIMIC subset as primary data. SHAC text is already the social-history slice; link to MIMIC only for demographics if needed. The paper frames status as a multiple-choice argument-resolution task and emphasizes FPR as the safety metric.

Final, corrected roadmap
0) Repo scaffold
agentic_shac/
  README.md
  pyproject.toml
  configs/
    data.yaml
    baseline.yaml
    agentic.yaml
    evaluation.yaml
  data/
    raw/            # symlink or copy from Track_2_SHAC/SHAC/{train,dev,test}/{mimic,uw}
    processed/
  experiments/
    baseline/
    agentic/
    reports/
  notebooks/
    00_eda.ipynb
    01_baseline_results.ipynb
    02_agentic_pipeline.ipynb
    03_fpr_analysis.ipynb
    04_error_analysis.ipynb
  src/
    utils/
      brat_loader.py
      preprocess.py
      sectionizer.py
      io.py
      logger.py
      seed.py
    baselines/
      llama_single.py
    agentic/
      interfaces.py
      prompts.py
      proposer.py
      refuter.py
      judge.py
      pipeline.py
    evaluation/
      metrics.py
      run_baseline.py
      run_agentic.py
      compare_runs.py
      plots.py
  tests/
    test_brat_loader.py
    test_preprocess.py
    test_prompts.py
    test_agentic_pipeline.py
  scripts/
    run_smoke.sh

Notes on fixes

Keep one data loader and one preprocessor. No duplicate pipes.

Baseline model uses Llama-3.1-8B-Instruct. Avoid 3B; 8B is instruction-strong and fits your GPUs (bf16 on 32 GB; 4-bit on 24 GB). It also aligns with the paper if you later fine-tune.

Remove ROC/AUC; not meaningful for 3-class with asymmetric costs. Keep FPR and accuracy.

Keep prompts terse. No CoT. No warnings. Your agents provide adversarial pressure.

1) Data handling
configs/data.yaml
raw_root: "/home/amin/Dropbox/code/SDOH/Track_2_SHAC/SHAC"
sources: ["mimic"]         # use "mimic" only to match paper; you can switch to ["mimic","uw"]
splits: ["train","dev","test"]
target_event: "Drug"
status_labels: ["none","current","past","Not Applicable"]

src/utils/brat_loader.py

Goal: parse .txt/.ann → rows for Drug events with StatusTime.
Output schema (List[dict]):

{
  "id": "m_train_000123_drug_1",
  "split": "train",
  "source": "mimic",
  "note_id": "...",
  "text": "<full social-history text>",
  "trigger_text": "drug",
  "status_label": "current" | "past" | "none" | "Not Applicable"
}


Map from BRAT: E links Alcohol|Drug|Tobacco trigger (T) to Status arg (T) and A StatusTimeVal holds the categorical label. Keep only event_type == Drug.

src/utils/preprocess.py

Input: list of dicts from brat_loader.

Clean text minimally (normalize whitespace; do not remove semantics).

Create processed JSONL per split under data/processed/{split}.jsonl.

Optional: if patient IDs exist, ensure patient-level non-leak splits; otherwise keep the provided splits.

tests/test_brat_loader.py

Feed 2–3 tiny .txt/.ann fixtures. Assert one row per Drug trigger and correct status mapping.

notebooks/00_eda.ipynb

Label counts, split sizes, FPR-relevant class ratios (gold none/unknown proportion).

Keep it light.

2) Baseline model (single predictor)
Task definition

Predict StatusTime for Drug, given the full SHAC note and the known trigger. This mirrors the paper’s argument-resolution step.

configs/baseline.yaml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
dtype: bf16           # set 4bit:true for 24 GB
load_in_4bit: false
max_new_tokens: 8
temperature: 0.1
top_p: 0.9
batch_size: 8
prompt_template: "status_v1"

src/baselines/llama_single.py

Batched inference.

Strict option parsing from model output a|b|c|d → status.

Save experiments/baseline/preds.jsonl with id, pred, gold, raw_text.

src/agentic/prompts.py (reused by baseline)
status_v1 (no CoT, no warnings)
System: You classify Drug StatusTime in clinical notes.
User:
Note:
{NOTE}

Drug trigger: "{TRIGGER}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter.

src/evaluation/run_baseline.py

Load data/processed/{split}.jsonl.

Assemble prompts, run llama_single.py.

Write predictions.

src/evaluation/metrics.py

Map letters to labels.

Compute:

Accuracy

FPR where gold ∈ {none, Not Applicable?}. Use your policy: in the paper “none/unknown” were negative. Keep none as negative; exclude “Not Applicable” from FPR denominator or treat it as negative if it aligns with your schema. State choice in README.

tests/test_baseline_prompt.py

Verify correct letter extraction and mapping.

notebooks/01_baseline_results.ipynb

Summarize baseline accuracy + FPR per split.

3) Agentic pipeline (Proposer / Refuter / Judge)
src/agentic/interfaces.py
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

Prompts (concise)
proposer_v1.txt
System: You classify Drug StatusTime in clinical notes.
User:
Note:
{NOTE}

Drug trigger: "{TRIGGER}"
Options: (a) none (b) current (c) past (d) Not Applicable
Answer with one letter only.

refuter_v1.txt
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

judge_v1.txt
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

src/utils/sectionizer.py

Heuristic split into assessment_plan, problems, meds, labs, other.

Build non_cue_text = concat(assessment_plan, problems, meds, labs).

Build masked_note = note with the sentence containing TRIGGER removed.

src/agentic/proposer.py

Run proposer_v1.txt per item on NOTE.

Also compute P_MASKED by running on masked_note.

Save proposer.jsonl: id, proposer_letter, proposer_masked_letter.

src/agentic/refuter.py

Input non_cue_text + proposer_letter.

Return refuter_letter and up to two quoted spans.

Save refuter.jsonl.

src/agentic/judge.py

Input proposer/refuter outputs + non_cue_text.

Apply rules; if ambiguous, default to (a) none per rule 2.

Save final.jsonl.

src/agentic/pipeline.py

Orchestrates batched runs.

Retries on parsing failures.

Caches prompts→outputs in experiments/agentic/cache/.

configs/agentic.yaml
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
  judge:   "judge_v1"
sectionizer:
  use_sections: ["assessment_plan","problems","meds","labs"]
mask_trigger_sentence: true
ablation:
  proposer_only: false
  proposer_refuter: false
  full_triad: true

tests/test_agentic_pipeline.py

Fixture: 3 tiny notes covering each class.

Assert pipeline returns one of a|b|c|d for each.

Snapshot a JSON of results for regression.

notebooks/02_agentic_pipeline.ipynb

Show a few rows with proposer/refuter/judge decisions and quoted spans.

4) Evaluation & FPR analysis
src/evaluation/run_agentic.py

Run pipeline on dev/test. Save experiments/agentic/<run_id>/final.jsonl.

src/evaluation/compare_runs.py

Load baseline vs agentic predictions.

Compute FPR deltas and accuracy deltas per split and overall.

Output CSV and a Markdown summary in experiments/reports/.

src/evaluation/plots.py

Matplotlib bar plots for FPR by configuration.

No seaborn requirement.

notebooks/03_fpr_analysis.ipynb

Compare baseline vs agentic FPR drop.

Optional: Alcohol+/Smoking+ subgroup analyses if you add those flags later. The paper stratifies by substance context; you can replicate by tagging notes with alcohol/smoking mentions if needed.

5) Testing & QA
tests/run_smoke.sh

Runs: loader → preprocess → baseline on 3 samples → agentic on 3 samples → metrics.

pyproject.toml

ruff + black + mypy optional.

Add [tool.pytest.ini_options] default markers.

6) Experiment management

src/utils/logger.py writes JSON logs per run.

Every run writes experiments/<run_id>/config.json and system_prompts.txt.

Keep seed.py to fix random seeds for any sampling.

7) Deployment & next steps

Optional CLI src/cli.py:

baseline --split test

agentic --split test

evaluate --runs baseline,agentic

Next: Try few-shot ICL (3 shots) to mirror paper variants, or LoRA fine-tune 8B later with their hyperparams if needed.