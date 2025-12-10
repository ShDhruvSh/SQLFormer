"""
Master Script to Run All Experiments

Runs all three methods (hybrid, constrained, unconstrained) on both datasets (Spider, BIRD)
and generates comprehensive comparison reports.

Usage:
    python run_all_experiments.py --max-examples 100 --datasets spider bird
    python run_all_experiments.py --datasets spider --methods hybrid constrained
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_spider_experiments(methods: List[str], max_examples: int = None):
    """Run experiments on Spider dataset."""
    print("\n" + "=" * 80)
    print("RUNNING SPIDER EXPERIMENTS")
    print("=" * 80)

    spider_dir = Path(__file__).parent / "spider"
    cmd = [
        sys.executable,
        str(spider_dir / "run_experiment.py"),
        "--data", str(spider_dir / "data" / "spider" / "dev.json"),
        "--tables", str(spider_dir / "data" / "spider" / "tables.json"),
        "--methods"
    ] + methods

    if max_examples:
        cmd += ["--max-examples", str(max_examples)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(spider_dir))

    if result.returncode != 0:
        print(f"ERROR: Spider experiments failed with code {result.returncode}")
        return False

    return True


def run_bird_experiments(methods: List[str], max_examples: int = None):
    """Run experiments on BIRD dataset."""
    print("\n" + "=" * 80)
    print("RUNNING BIRD EXPERIMENTS")
    print("=" * 80)

    bird_dir = Path(__file__).parent / "bird"
    cmd = [
        sys.executable,
        str(bird_dir / "run_experiment.py"),
        "--data", str(bird_dir / "data" / "bird" / "dev" / "dev.json"),
        "--database-dir", str(bird_dir / "data" / "bird" / "dev" / "dev_databases"),
        "--methods"
    ] + methods

    if max_examples:
        cmd += ["--max-examples", str(max_examples)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(bird_dir))

    if result.returncode != 0:
        print(f"ERROR: BIRD experiments failed with code {result.returncode}")
        return False

    return True


def evaluate_spider_results(methods: List[str]):
    """Evaluate Spider predictions."""
    print("\n" + "=" * 80)
    print("EVALUATING SPIDER RESULTS")
    print("=" * 80)

    spider_dir = Path(__file__).parent / "spider"

    for method in methods:
        print(f"\nEvaluating {method}...")
        cmd = [
            sys.executable,
            str(spider_dir / "evaluate.py"),
            "--gold", str(spider_dir / "data" / "spider" / "dev.json"),
            "--predictions", str(spider_dir / "predictions" / f"{method}_predictions.json"),
            "--tables", str(spider_dir / "data" / "spider" / "tables.json"),
            "--database-dir", str(spider_dir / "data" / "spider" / "database"),
            "--output", str(spider_dir / "results" / f"{method}_results.json")
        ]

        result = subprocess.run(cmd, cwd=str(spider_dir))

        if result.returncode != 0:
            print(f"WARNING: Evaluation of {method} failed")


def evaluate_bird_results(methods: List[str]):
    """Evaluate BIRD predictions."""
    print("\n" + "=" * 80)
    print("EVALUATING BIRD RESULTS")
    print("=" * 80)

    bird_dir = Path(__file__).parent / "bird"

    for method in methods:
        print(f"\nEvaluating {method}...")
        cmd = [
            sys.executable,
            str(bird_dir / "evaluate.py"),
            "--gold", str(bird_dir / "data" / "bird" / "dev" / "dev.json"),
            "--predictions", str(bird_dir / "predictions" / f"{method}_predictions.json"),
            "--database-dir", str(bird_dir / "data" / "bird" / "dev" / "dev_databases"),
            "--output", str(bird_dir / "results" / f"{method}_results.json")
        ]

        result = subprocess.run(cmd, cwd=str(bird_dir))

        if result.returncode != 0:
            print(f"WARNING: Evaluation of {method} failed")


def generate_comparison_report():
    """Generate comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORTS")
    print("=" * 80)

    # Spider comparison
    spider_dir = Path(__file__).parent / "spider"
    if (spider_dir / "results").exists():
        print("\nGenerating Spider comparison...")
        cmd = [
            sys.executable,
            str(spider_dir / "analyze_results.py"),
            "--results-dir", str(spider_dir / "results"),
            "--output", str(spider_dir / "COMPARISON_REPORT.md")
        ]
        subprocess.run(cmd, cwd=str(spider_dir))

    # BIRD comparison
    bird_dir = Path(__file__).parent / "bird"
    if (bird_dir / "results").exists():
        print("\nGenerating BIRD comparison...")
        # Check if analyze_results.py exists for BIRD, if not we'll create one
        analyze_script = bird_dir / "analyze_results.py"
        if not analyze_script.exists():
            print("  (BIRD analyze_results.py not found, skipping detailed comparison)")
        else:
            cmd = [
                sys.executable,
                str(analyze_script),
                "--results-dir", str(bird_dir / "results"),
                "--output", str(bird_dir / "COMPARISON_REPORT.md")
            ]
            subprocess.run(cmd, cwd=str(bird_dir))


def main():
    parser = argparse.ArgumentParser(description="Run all SQLFormer experiments")
    parser.add_argument("--datasets", nargs="+",
                        choices=["spider", "bird"],
                        default=["spider", "bird"],
                        help="Datasets to run experiments on")
    parser.add_argument("--methods", nargs="+",
                        choices=["hybrid", "constrained", "unconstrained"],
                        default=["hybrid", "constrained", "unconstrained"],
                        help="Methods to compare")
    parser.add_argument("--max-examples", "--max-samples", type=int, dest="max_examples",
                        help="Maximum examples per dataset (for testing)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip comparison report generation")

    args = parser.parse_args()

    print("=" * 80)
    print("SQLFormer Comprehensive Experiment Runner")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")

    # Run experiments
    if "spider" in args.datasets:
        success = run_spider_experiments(args.methods, args.max_examples)
        if success and not args.skip_eval:
            evaluate_spider_results(args.methods)

    if "bird" in args.datasets:
        success = run_bird_experiments(args.methods, args.max_examples)
        if success and not args.skip_eval:
            evaluate_bird_results(args.methods)

    # Generate comparison
    if not args.skip_comparison:
        generate_comparison_report()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nResults:")
    if "spider" in args.datasets:
        print(f"  Spider: experiments/spider/results/")
    if "bird" in args.datasets:
        print(f"  BIRD: experiments/bird/results/")


if __name__ == "__main__":
    main()
