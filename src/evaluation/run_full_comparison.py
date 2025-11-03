"""
Run Full Baseline vs Agentic Comparison (Phase 4).

This script:
1. Runs baseline on full dev/test splits
2. Runs agentic on full dev/test splits
3. Automatically compares results with detailed metrics
4. Generates comprehensive comparison reports

This provides the big batch comparison requested.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Error running: {description}")
        sys.exit(1)
    
    print(f"✅ {description} completed")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run full baseline vs agentic comparison on dev/test splits"
    )
    parser.add_argument("--split", nargs="+", default=["dev", "test"], 
                       help="Splits to evaluate (e.g., dev test)")
    parser.add_argument("--skip_baseline", action="store_true",
                       help="Skip baseline run (use existing)")
    parser.add_argument("--skip_agentic", action="store_true",
                       help="Skip agentic run (use existing)")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip comparison (only run inference)")
    parser.add_argument("--baseline_config", default="configs/baseline.yaml",
                       help="Baseline config path")
    parser.add_argument("--agentic_config", default="configs/agentic.yaml",
                       help="Agentic config path")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    
    print("=" * 80)
    print("FULL BASELINE vs AGENTIC COMPARISON")
    print("=" * 80)
    print(f"Splits: {args.split}")
    print(f"Project root: {project_root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run baseline on full splits
    if not args.skip_baseline:
        baseline_cmd = [
            sys.executable, "-m", "src.evaluation.run_baseline",
            "--split"] + args.split + [
            "--config", args.baseline_config
        ]
        run_command(baseline_cmd, "Baseline Run on Full Splits")
    else:
        print("\n⏭️  Skipping baseline run (using existing)")
    
    # Step 2: Run agentic on full splits
    if not args.skip_agentic:
        agentic_cmd = [
            sys.executable, "-m", "src.evaluation.run_agentic",
            "--split"] + args.split + [
            "--config", args.agentic_config
        ]
        run_command(agentic_cmd, "Agentic Run on Full Splits")
    else:
        print("\n⏭️  Skipping agentic run (using existing)")
    
    # Step 3: Compare results
    if not args.skip_comparison:
        for split in args.split:
            print(f"\n{'='*80}")
            print(f"Comparing {split.upper()} split")
            print(f"{'='*80}")
            
            compare_cmd = [
                sys.executable, "-m", "src.evaluation.compare_runs",
                "--split", split
            ]
            run_command(compare_cmd, f"Comparison for {split.upper()}")
    else:
        print("\n⏭️  Skipping comparison")
    
    print("\n" + "=" * 80)
    print("✅ FULL COMPARISON COMPLETE")
    print("=" * 80)
    print("\nResults saved in:")
    print("  - experiments/baseline/<run_id>/")
    print("  - experiments/agentic/<run_id>/")
    print("  - experiments/reports/comparison_*.{csv,md,json}")
    print("=" * 80)


if __name__ == "__main__":
    main()

