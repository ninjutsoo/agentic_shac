"""
Baseline runner for Drug StatusTime classification (Phase 2).

Loads processed JSONL data, runs the single-model baseline, saves predictions,
and prints evaluation metrics (including FPR).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import yaml

from src.baselines.llama_single import LlamaSingleBaseline
from src.utils.preprocess import load_from_jsonl
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def ensure_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference and evaluation")
    parser.add_argument("--split", nargs="+", default=["dev"], help="Splits to evaluate (e.g., dev test)")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to baseline config YAML")
    parser.add_argument("--data_dir", default="data/processed", help="Directory with processed JSONL files")
    parser.add_argument("--out_dir", default="experiments/baseline", help="Output directory for predictions")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config).resolve()
    data_dir = (project_root / args.data_dir).resolve()
    out_base = (project_root / args.out_dir).resolve()

    print("=" * 80)
    print("BASELINE RUNNER")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Data dir: {data_dir}")
    print(f"Output base: {out_base}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    baseline = LlamaSingleBaseline(config)
    baseline.load_model()

    # Prepare output run directory
    run_dir = ensure_output_dir(out_base)
    print(f"Run dir: {run_dir}")

    # Process each requested split
    for split in args.split:
        split_path = data_dir / f"{split}.jsonl"
        if not split_path.exists():
            print(f"⚠️  Missing data file for split '{split}': {split_path}")
            continue

        print(f"\nLoading data for split '{split}' from {split_path}...")
        samples = load_from_jsonl(split_path)
        print(f"Loaded {len(samples)} samples.")

        print("Running inference...")
        start_time = time.time()
        results = baseline.predict_batch(samples, show_progress=True)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds ({inference_time/len(samples):.4f} seconds per sample)")

        # Save predictions
        preds_path = run_dir / f"preds_{split}.jsonl"
        save_jsonl(results, preds_path)
        print(f"Saved predictions to {preds_path}")

        # Evaluate
        y_true = [r.get("status_label", "Not Applicable") for r in results]
        y_pred = [r.get("pred_label", "Not Applicable") for r in results]
        # Include any unexpected labels from data (e.g., 'future')
        base_labels = {"none", "current", "past", "Not Applicable"}
        labels = sorted(base_labels | set(y_true))
        metrics = compute_all_metrics(y_true, y_pred, labels=labels)
        print_metrics_report(metrics)

        # Save metrics JSON (include timing information)
        metrics['inference_time'] = inference_time
        metrics['n_samples'] = len(samples)
        metrics['split'] = split
        metrics['run_id'] = run_dir.name
        metrics_path = run_dir / f"metrics_{split}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else o, indent=2)
        print(f"Saved metrics to {metrics_path}")

    print("\n" + "=" * 80)
    print("✅ Baseline run completed")
    print("=" * 80)


if __name__ == "__main__":
    main()


