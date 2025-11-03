"""
Compare Baseline vs Agentic Performance (Phase 4).

Loads predictions from baseline and agentic runs, computes comprehensive metrics,
and generates detailed comparison reports.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import csv
import time

from src.utils.preprocess import load_from_jsonl
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def load_predictions(preds_path: Path) -> List[Dict]:
    """Load predictions from JSONL file."""
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    return load_from_jsonl(preds_path)


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compute_deltas(baseline_metrics: Dict, agentic_metrics: Dict) -> Dict:
    """Compute deltas between baseline and agentic metrics."""
    deltas = {}
    
    # Overall metrics
    deltas['accuracy'] = {
        'baseline': baseline_metrics['accuracy'],
        'agentic': agentic_metrics['accuracy'],
        'absolute': agentic_metrics['accuracy'] - baseline_metrics['accuracy'],
        'relative': ((agentic_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100) if baseline_metrics['accuracy'] > 0 else 0.0
    }
    
    deltas['fpr'] = {
        'baseline': baseline_metrics['fpr'],
        'agentic': agentic_metrics['fpr'],
        'absolute': agentic_metrics['fpr'] - baseline_metrics['fpr'],
        'relative': ((agentic_metrics['fpr'] - baseline_metrics['fpr']) / baseline_metrics['fpr'] * 100) if baseline_metrics['fpr'] > 0 else 0.0
    }
    
    # Per-class metrics
    deltas['per_class'] = {}
    labels = baseline_metrics.get('labels', sorted(set(baseline_metrics.get('per_class', {}).keys())))
    
    for label in labels:
        baseline_perf = baseline_metrics.get('per_class', {}).get(label, {})
        agentic_perf = agentic_metrics.get('per_class', {}).get(label, {})
        
        deltas['per_class'][label] = {
            'f1': {
                'baseline': baseline_perf.get('f1', 0.0),
                'agentic': agentic_perf.get('f1', 0.0),
                'absolute': agentic_perf.get('f1', 0.0) - baseline_perf.get('f1', 0.0)
            },
            'precision': {
                'baseline': baseline_perf.get('precision', 0.0),
                'agentic': agentic_perf.get('precision', 0.0),
                'absolute': agentic_perf.get('precision', 0.0) - baseline_perf.get('precision', 0.0)
            },
            'recall': {
                'baseline': baseline_perf.get('recall', 0.0),
                'agentic': agentic_perf.get('recall', 0.0),
                'absolute': agentic_perf.get('recall', 0.0) - baseline_perf.get('recall', 0.0)
            },
            'support': baseline_perf.get('support', 0)
        }
    
    deltas['n_samples'] = baseline_metrics.get('n_samples', 0)
    
    return deltas


def print_comparison_report(deltas: Dict, split: str):
    """Print detailed comparison report."""
    print("=" * 80)
    print(f"BASELINE vs AGENTIC COMPARISON - {split.upper()}")
    print("=" * 80)
    
    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    
    print(f"\nAccuracy:")
    print(f"  Baseline: {deltas['accuracy']['baseline']:.4f}")
    print(f"  Agentic:  {deltas['accuracy']['agentic']:.4f}")
    print(f"  Change:   {deltas['accuracy']['absolute']:+.4f} ({deltas['accuracy']['relative']:+.1f}%)")
    
    print(f"\nFPR (False Positive Rate) - PRIMARY METRIC:")
    print(f"  Baseline: {deltas['fpr']['baseline']:.4f}")
    print(f"  Agentic:  {deltas['fpr']['agentic']:.4f}")
    print(f"  Change:   {deltas['fpr']['absolute']:+.4f} ({deltas['fpr']['relative']:+.1f}%)")
    
    if deltas['fpr']['absolute'] < 0:
        improvement = abs(deltas['fpr']['absolute']) / deltas['fpr']['baseline'] * 100 if deltas['fpr']['baseline'] > 0 else 0
        print(f"  ✅ FPR IMPROVED by {improvement:.1f}% (relative reduction)")
    elif deltas['fpr']['absolute'] > 0:
        degradation = deltas['fpr']['absolute'] / deltas['fpr']['baseline'] * 100 if deltas['fpr']['baseline'] > 0 else 0
        print(f"  ⚠️  FPR WORSENED by {degradation:.1f}% (relative increase)")
    else:
        print(f"  ➡️  FPR unchanged")
    
    # Per-class metrics
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)
    
    print(f"\n{'Label':<20} {'Metric':<10} {'Baseline':>10} {'Agentic':>10} {'Change':>10}")
    print("-" * 70)
    
    for label, metrics in deltas['per_class'].items():
        support = metrics['support']
        print(f"\n{label:<20} (support: {support})")
        
        for metric_name in ['f1', 'precision', 'recall']:
            metric = metrics[metric_name]
            baseline_val = metric['baseline']
            agentic_val = metric['agentic']
            change = metric['absolute']
            
            change_str = f"{change:+.4f}" if abs(change) > 0.0001 else "0.0000"
            print(f"  {metric_name:<18} {baseline_val:10.4f} {agentic_val:10.4f} {change_str:>10}")
    
    print(f"\nTotal Samples: {deltas['n_samples']}")
    print("=" * 80)


def save_comparison_csv(deltas: Dict, output_path: Path, split: str):
    """Save comparison to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Metric', 'Baseline', 'Agentic', 'Absolute Change', 'Relative Change (%)'])
        
        # Overall metrics
        writer.writerow(['Accuracy', 
                        f"{deltas['accuracy']['baseline']:.4f}",
                        f"{deltas['accuracy']['agentic']:.4f}",
                        f"{deltas['accuracy']['absolute']:+.4f}",
                        f"{deltas['accuracy']['relative']:+.2f}"])
        
        writer.writerow(['FPR', 
                        f"{deltas['fpr']['baseline']:.4f}",
                        f"{deltas['fpr']['agentic']:.4f}",
                        f"{deltas['fpr']['absolute']:+.4f}",
                        f"{deltas['fpr']['relative']:+.2f}"])
        
        writer.writerow([])  # Empty row
        
        # Per-class metrics
        writer.writerow(['Class', 'Metric', 'Baseline', 'Agentic', 'Absolute Change', 'Support'])
        
        for label, metrics in deltas['per_class'].items():
            support = metrics['support']
            for metric_name in ['f1', 'precision', 'recall']:
                metric = metrics[metric_name]
                writer.writerow([label,
                               metric_name,
                               f"{metric['baseline']:.4f}",
                               f"{metric['agentic']:.4f}",
                               f"{metric['absolute']:+.4f}",
                               support])
    
    print(f"\nSaved comparison CSV to: {output_path}")


def save_comparison_markdown(deltas: Dict, output_path: Path, split: str, baseline_run_id: str, agentic_run_id: str):
    """Save comparison to Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Baseline vs Agentic Comparison - {split.upper()}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Baseline Run:** {baseline_run_id}\n\n")
        f.write(f"**Agentic Run:** {agentic_run_id}\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Baseline | Agentic | Change | Change % |\n")
        f.write("|--------|----------|---------|--------|----------|\n")
        f.write(f"| Accuracy | {deltas['accuracy']['baseline']:.4f} | {deltas['accuracy']['agentic']:.4f} | "
                f"{deltas['accuracy']['absolute']:+.4f} | {deltas['accuracy']['relative']:+.2f}% |\n")
        f.write(f"| **FPR** | **{deltas['fpr']['baseline']:.4f}** | **{deltas['fpr']['agentic']:.4f}** | "
                f"**{deltas['fpr']['absolute']:+.4f}** | **{deltas['fpr']['relative']:+.2f}%** |\n\n")
        
        if deltas['fpr']['absolute'] < 0:
            improvement = abs(deltas['fpr']['absolute']) / deltas['fpr']['baseline'] * 100 if deltas['fpr']['baseline'] > 0 else 0
            f.write(f"✅ **FPR Improved by {improvement:.1f}%** (relative reduction)\n\n")
        elif deltas['fpr']['absolute'] > 0:
            f.write(f"⚠️ **FPR Worsened by {abs(deltas['fpr']['relative']):.1f}%** (relative increase)\n\n")
        
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Metric | Baseline | Agentic | Change | Support |\n")
        f.write("|-------|--------|----------|---------|--------|---------|\n")
        
        for label, metrics in deltas['per_class'].items():
            support = metrics['support']
            for metric_name in ['f1', 'precision', 'recall']:
                metric = metrics[metric_name]
                f.write(f"| {label} | {metric_name} | {metric['baseline']:.4f} | "
                       f"{metric['agentic']:.4f} | {metric['absolute']:+.4f} | {support} |\n")
        
        f.write(f"\n**Total Samples:** {deltas['n_samples']}\n")
    
    print(f"Saved comparison Markdown to: {output_path}")


def save_comprehensive_txt_report(deltas: Dict, baseline_metrics: Dict, agentic_metrics: Dict,
                                   output_path: Path, split: str, baseline_run_id: str, agentic_run_id: str,
                                   baseline_data_path: Path, agentic_data_path: Path,
                                   baseline_time: Optional[float] = None, agentic_time: Optional[float] = None):
    """Save comprehensive text report with all experiment details."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get data description
    from collections import Counter
    try:
        baseline_samples = load_predictions(baseline_data_path) if baseline_data_path.exists() else []
    except:
        baseline_samples = []
    try:
        agentic_samples = load_predictions(agentic_data_path) if agentic_data_path.exists() else []
    except:
        agentic_samples = []
    
    # Analyze data
    baseline_sources = Counter(s.get('source', 'unknown') for s in baseline_samples)
    agentic_sources = Counter(s.get('source', 'unknown') for s in agentic_samples)
    baseline_labels = Counter(s.get('status_label', 'unknown') for s in baseline_samples)
    agentic_labels = Counter(s.get('status_label', 'unknown') for s in agentic_samples)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE vs AGENTIC COMPREHENSIVE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment metadata
        f.write("EXPERIMENT METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split: {split.upper()}\n")
        f.write(f"Baseline Run ID: {baseline_run_id}\n")
        f.write(f"Agentic Run ID: {agentic_run_id}\n")
        f.write("\n")
        
        # Data description
        f.write("DATA DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples: {deltas['n_samples']}\n")
        f.write(f"Baseline Samples: {len(baseline_samples)}\n")
        f.write(f"Agentic Samples: {len(agentic_samples)}\n")
        f.write("\n")
        
        f.write("Data Sources (Baseline):\n")
        for source, count in baseline_sources.items():
            f.write(f"  - {source}: {count}\n")
        f.write("\n")
        
        f.write("Data Sources (Agentic):\n")
        for source, count in agentic_sources.items():
            f.write(f"  - {source}: {count}\n")
        f.write("\n")
        
        f.write("Label Distribution (Baseline):\n")
        for label, count in sorted(baseline_labels.items()):
            f.write(f"  - {label}: {count} ({count/len(baseline_samples)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Label Distribution (Agentic):\n")
        for label, count in sorted(agentic_labels.items()):
            f.write(f"  - {label}: {count} ({count/len(agentic_samples)*100:.1f}%)\n")
        f.write("\n")
        
        # Timing information
        f.write("TIMING INFORMATION\n")
        f.write("-" * 80 + "\n")
        if baseline_time is not None:
            f.write(f"Baseline Inference Time: {baseline_time:.2f} seconds\n")
            f.write(f"Baseline Time per Sample: {baseline_time/len(baseline_samples):.4f} seconds\n")
        else:
            f.write("Baseline Inference Time: Not available\n")
        f.write("\n")
        
        if agentic_time is not None:
            f.write(f"Agentic Inference Time: {agentic_time:.2f} seconds\n")
            f.write(f"Agentic Time per Sample: {agentic_time/len(agentic_samples):.4f} seconds\n")
            if baseline_time is not None:
                speedup = baseline_time / agentic_time if agentic_time > 0 else 0
                f.write(f"Speedup Factor: {speedup:.2f}x\n")
        else:
            f.write("Agentic Inference Time: Not available\n")
        f.write("\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:\n")
        f.write(f"  Baseline: {deltas['accuracy']['baseline']:.4f}\n")
        f.write(f"  Agentic:  {deltas['accuracy']['agentic']:.4f}\n")
        f.write(f"  Change:   {deltas['accuracy']['absolute']:+.4f} ({deltas['accuracy']['relative']:+.1f}%)\n")
        f.write("\n")
        
        f.write(f"FPR (False Positive Rate) - PRIMARY METRIC:\n")
        f.write(f"  Baseline: {deltas['fpr']['baseline']:.4f}\n")
        f.write(f"  Agentic:  {deltas['fpr']['agentic']:.4f}\n")
        f.write(f"  Change:   {deltas['fpr']['absolute']:+.4f} ({deltas['fpr']['relative']:+.1f}%)\n")
        if deltas['fpr']['absolute'] < 0:
            improvement = abs(deltas['fpr']['absolute']) / deltas['fpr']['baseline'] * 100 if deltas['fpr']['baseline'] > 0 else 0
            f.write(f"  ✅ FPR IMPROVED by {improvement:.1f}% (relative reduction)\n")
        elif deltas['fpr']['absolute'] > 0:
            degradation = deltas['fpr']['absolute'] / deltas['fpr']['baseline'] * 100 if deltas['fpr']['baseline'] > 0 else 0
            f.write(f"  ⚠️  FPR WORSENED by {degradation:.1f}% (relative increase)\n")
        else:
            f.write(f"  ➡️  FPR unchanged\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        for label, metrics in sorted(deltas['per_class'].items()):
            support = metrics['support']
            f.write(f"\n{label.upper()} (support: {support}):\n")
            for metric_name in ['f1', 'precision', 'recall']:
                metric = metrics[metric_name]
                f.write(f"  {metric_name.capitalize():<12} Baseline: {metric['baseline']:>6.4f}  "
                       f"Agentic: {metric['agentic']:>6.4f}  Change: {metric['absolute']:>+7.4f}\n")
        f.write("\n")
        
        # Detailed metrics breakdown
        f.write("DETAILED METRICS BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Metrics:\n")
        f.write(f"  Accuracy: {baseline_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  FPR: {baseline_metrics.get('fpr', 0):.4f}\n")
        f.write(f"  Precision (macro): {baseline_metrics.get('precision_macro', 0):.4f}\n")
        f.write(f"  Recall (macro): {baseline_metrics.get('recall_macro', 0):.4f}\n")
        f.write(f"  F1 (macro): {baseline_metrics.get('f1_macro', 0):.4f}\n")
        f.write("\n")
        
        f.write(f"Agentic Metrics:\n")
        f.write(f"  Accuracy: {agentic_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  FPR: {agentic_metrics.get('fpr', 0):.4f}\n")
        f.write(f"  Precision (macro): {agentic_metrics.get('precision_macro', 0):.4f}\n")
        f.write(f"  Recall (macro): {agentic_metrics.get('recall_macro', 0):.4f}\n")
        f.write(f"  F1 (macro): {agentic_metrics.get('f1_macro', 0):.4f}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved comprehensive text report to: {output_path}")


def find_latest_run(experiments_dir: Path, method: str, split: str) -> Optional[Path]:
    """Find the latest run directory for a method and split."""
    if not experiments_dir.exists():
        return None
    
    method_dir = experiments_dir / method
    if not method_dir.exists():
        return None
    
    # Find all run directories
    run_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()], reverse=True)
    
    for run_dir in run_dirs:
        metrics_path = run_dir / f"metrics_{split}.json"
        if metrics_path.exists():
            return run_dir
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs agentic performance")
    parser.add_argument("--split", default="dev", help="Split to compare (dev or test)")
    parser.add_argument("--baseline_dir", type=Path, help="Baseline run directory (or use latest)")
    parser.add_argument("--agentic_dir", type=Path, help="Agentic run directory (or use latest)")
    parser.add_argument("--experiments_dir", type=Path, default="experiments", help="Experiments root directory")
    parser.add_argument("--output_dir", type=Path, default="experiments/reports", help="Output directory for reports")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    experiments_dir = (project_root / args.experiments_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    
    print("=" * 80)
    print("BASELINE vs AGENTIC COMPARISON")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Experiments dir: {experiments_dir}")
    print(f"Output dir: {output_dir}")
    
    # Find run directories
    if args.baseline_dir:
        baseline_run_dir = Path(args.baseline_dir).resolve()
    else:
        baseline_run_dir = find_latest_run(experiments_dir, "baseline", args.split)
        if baseline_run_dir is None:
            print(f"❌ No baseline run found for split '{args.split}'")
            return
        print(f"Using latest baseline run: {baseline_run_dir.name}")
    
    if args.agentic_dir:
        agentic_run_dir = Path(args.agentic_dir).resolve()
    else:
        agentic_run_dir = find_latest_run(experiments_dir, "agentic", args.split)
        if agentic_run_dir is None:
            print(f"❌ No agentic run found for split '{args.split}'")
            return
        print(f"Using latest agentic run: {agentic_run_dir.name}")
    
    # Load metrics
    print("\nLoading metrics...")
    baseline_metrics_path = baseline_run_dir / f"metrics_{args.split}.json"
    agentic_metrics_path = agentic_run_dir / f"metrics_{args.split}.json"
    
    baseline_metrics = load_metrics(baseline_metrics_path)
    agentic_metrics = load_metrics(agentic_metrics_path)
    
    print(f"✅ Loaded baseline metrics from: {baseline_metrics_path}")
    print(f"✅ Loaded agentic metrics from: {agentic_metrics_path}")
    
    # Compute deltas
    print("\nComputing deltas...")
    deltas = compute_deltas(baseline_metrics, agentic_metrics)
    
    # Print report
    print_comparison_report(deltas, args.split)
    
    # Save reports
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = (project_root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = output_dir / f"comparison_{args.split}_{timestamp}.csv"
    save_comparison_csv(deltas, csv_path, args.split)
    
    md_path = output_dir / f"comparison_{args.split}_{timestamp}.md"
    save_comparison_markdown(deltas, md_path, args.split, 
                           baseline_run_dir.name, agentic_run_dir.name)
    
    # Save JSON comparison
    json_path = output_dir / f"comparison_{args.split}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'split': args.split,
            'baseline_run': baseline_run_dir.name,
            'agentic_run': agentic_run_dir.name,
            'timestamp': timestamp,
            'deltas': deltas
        }, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    print(f"Saved comparison JSON to: {json_path}")
    
    # Save comprehensive text report in results/ folder
    baseline_preds_path = baseline_run_dir / f"preds_{args.split}.jsonl"
    agentic_preds_path = agentic_run_dir / f"preds_{args.split}.jsonl"
    baseline_time = baseline_metrics.get('inference_time')
    agentic_time = agentic_metrics.get('inference_time')
    
    txt_path = results_dir / f"comparison_report_{args.split}_{timestamp}.txt"
    save_comprehensive_txt_report(
        deltas, baseline_metrics, agentic_metrics,
        txt_path, args.split, baseline_run_dir.name, agentic_run_dir.name,
        baseline_preds_path, agentic_preds_path,
        baseline_time, agentic_time
    )
    
    print("\n" + "=" * 80)
    print("✅ Comparison Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

