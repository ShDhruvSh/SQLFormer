"""
Results Analysis and Comparison for Spider Experiments

Generates:
- Comparison tables (LaTeX and Markdown)
- Statistical significance tests
- Error analysis reports
- Visualizations for paper
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ResultsAnalyzer:
    """Analyzes and compares experiment results."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing *_metrics.json files
        """
        self.results_dir = Path(results_dir)
        self.metrics: Dict[str, Dict] = {}
        self.detailed_results: Dict[str, List[Dict]] = {}

    def load_results(self, model_names: Optional[List[str]] = None):
        """Load all metrics files from results directory."""
        if model_names:
            files = [self.results_dir / f"{name}_metrics.json" for name in model_names]
        else:
            files = list(self.results_dir.glob("*_metrics.json"))

        for metrics_file in files:
            if metrics_file.exists():
                model_name = metrics_file.stem.replace("_metrics", "")
                with open(metrics_file, 'r') as f:
                    self.metrics[model_name] = json.load(f)

                # Try to load detailed results
                results_file = self.results_dir / f"{model_name}_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.detailed_results[model_name] = json.load(f)

        print(f"Loaded results for: {list(self.metrics.keys())}")

    def print_comparison_table(self):
        """Print comparison table to console."""
        if not self.metrics:
            print("No results loaded")
            return

        print("\n" + "=" * 80)
        print("  COMPARISON TABLE")
        print("=" * 80)

        # Header
        print(f"\n{'Model':<20} {'Validity':>10} {'Exec Acc':>10} {'Exact Match':>12} {'Total':>8}")
        print("-" * 65)

        # Rows
        for model, m in sorted(self.metrics.items()):
            print(f"{model:<20} {m['validity_rate']:>9.1f}% {m['exec_accuracy']:>9.1f}% {m['exact_match_rate']:>11.1f}% {m['total']:>8}")

        print("-" * 65)

    def generate_latex_table(self, output_file: Optional[str] = None) -> str:
        """Generate LaTeX table for paper."""
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Comparison of SQL Generation Methods on Spider Dev Set}")
        lines.append(r"\label{tab:results}")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Method & Validity (\%) & Exec Acc (\%) & Exact Match (\%) \\")
        lines.append(r"\midrule")

        for model, m in sorted(self.metrics.items()):
            model_display = model.replace("_", " ").title()
            lines.append(f"{model_display} & {m['validity_rate']:.1f} & {m['exec_accuracy']:.1f} & {m['exact_match_rate']:.1f} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(latex)
            print(f"Saved LaTeX table to: {output_file}")

        return latex

    def generate_markdown_table(self, output_file: Optional[str] = None) -> str:
        """Generate Markdown table."""
        lines = []
        lines.append("## Results Comparison")
        lines.append("")
        lines.append("| Model | Validity (%) | Exec Accuracy (%) | Exact Match (%) |")
        lines.append("|-------|-------------|-------------------|-----------------|")

        for model, m in sorted(self.metrics.items()):
            model_display = model.replace("_", " ").title()
            lines.append(f"| {model_display} | {m['validity_rate']:.1f} | {m['exec_accuracy']:.1f} | {m['exact_match_rate']:.1f} |")

        md = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(md)
            print(f"Saved Markdown table to: {output_file}")

        return md

    def generate_difficulty_breakdown(self, output_file: Optional[str] = None) -> str:
        """Generate breakdown by difficulty level."""
        lines = []
        lines.append("\n## Results by Difficulty Level\n")

        # Collect all difficulty levels
        all_difficulties = set()
        for m in self.metrics.values():
            all_difficulties.update(m.get('by_difficulty', {}).keys())

        difficulties = sorted(all_difficulties)

        # Header
        header = "| Difficulty |"
        separator = "|------------|"
        for model in sorted(self.metrics.keys()):
            header += f" {model} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Rows
        for diff in difficulties:
            row = f"| {diff} |"
            for model in sorted(self.metrics.keys()):
                diff_stats = self.metrics[model].get('by_difficulty', {}).get(diff, {})
                if diff_stats.get('total', 0) > 0:
                    acc = diff_stats['exec_correct'] / diff_stats['total'] * 100
                    row += f" {acc:.1f}% |"
                else:
                    row += " - |"
            lines.append(row)

        md = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(md)
            print(f"Saved difficulty breakdown to: {output_file}")

        return md

    def analyze_errors(self, model_name: str) -> Dict:
        """Analyze error patterns for a specific model."""
        if model_name not in self.detailed_results:
            return {}

        results = self.detailed_results[model_name]

        analysis = {
            "total_errors": 0,
            "error_types": defaultdict(int),
            "errors_by_db": defaultdict(int),
            "common_patterns": defaultdict(int),
            "sample_errors": []
        }

        for r in results:
            if not r['is_exec_correct']:
                analysis["total_errors"] += 1
                analysis["errors_by_db"][r['db_id']] += 1

                if r['error_message']:
                    # Categorize error
                    err = r['error_message'].lower()
                    if "syntax" in err:
                        analysis["error_types"]["syntax"] += 1
                    elif "no such table" in err or "no such column" in err:
                        analysis["error_types"]["schema"] += 1
                    elif "ambiguous" in err:
                        analysis["error_types"]["ambiguous"] += 1
                    else:
                        analysis["error_types"]["execution"] += 1

                # Collect sample errors
                if len(analysis["sample_errors"]) < 10:
                    analysis["sample_errors"].append({
                        "question": r['question'],
                        "gold": r['gold_sql'],
                        "predicted": r['predicted_sql'],
                        "error": r['error_message']
                    })

        return analysis

    def generate_error_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive error analysis report."""
        lines = []
        lines.append("# Error Analysis Report\n")

        for model in sorted(self.detailed_results.keys()):
            lines.append(f"\n## {model.replace('_', ' ').title()}\n")

            analysis = self.analyze_errors(model)

            lines.append(f"Total Errors: {analysis['total_errors']}\n")

            lines.append("### Error Types")
            for err_type, count in sorted(analysis['error_types'].items(), key=lambda x: -x[1]):
                lines.append(f"- {err_type}: {count}")

            lines.append("\n### Top 5 Databases with Most Errors")
            sorted_dbs = sorted(analysis['errors_by_db'].items(), key=lambda x: -x[1])[:5]
            for db, count in sorted_dbs:
                lines.append(f"- {db}: {count} errors")

            lines.append("\n### Sample Errors")
            for i, sample in enumerate(analysis['sample_errors'][:5], 1):
                lines.append(f"\n**Error {i}:**")
                lines.append(f"- Question: {sample['question']}")
                lines.append(f"- Gold SQL: `{sample['gold']}`")
                lines.append(f"- Predicted: `{sample['predicted']}`")
                lines.append(f"- Error: {sample['error']}")

        report = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Saved error report to: {output_file}")

        return report

    def statistical_significance(self, model1: str, model2: str) -> Optional[Dict]:
        """
        Compute statistical significance between two models.

        Uses McNemar's test for paired binary outcomes.
        """
        if not HAS_SCIPY:
            print("scipy not installed - skipping statistical tests")
            return None

        if model1 not in self.detailed_results or model2 not in self.detailed_results:
            print(f"Detailed results not available for both models")
            return None

        results1 = {r['idx']: r['is_exec_correct'] for r in self.detailed_results[model1]}
        results2 = {r['idx']: r['is_exec_correct'] for r in self.detailed_results[model2]}

        # Find common examples
        common_idxs = set(results1.keys()) & set(results2.keys())

        # Build contingency table
        # a: both correct, b: model1 correct only, c: model2 correct only, d: both wrong
        a, b, c, d = 0, 0, 0, 0
        for idx in common_idxs:
            r1, r2 = results1[idx], results2[idx]
            if r1 and r2:
                a += 1
            elif r1 and not r2:
                b += 1
            elif not r1 and r2:
                c += 1
            else:
                d += 1

        # McNemar's test
        if b + c > 0:
            # Use exact binomial test for small samples
            n = b + c
            k = min(b, c)
            p_value = 2 * stats.binom.cdf(k, n, 0.5)  # Two-tailed
        else:
            p_value = 1.0

        return {
            "model1": model1,
            "model2": model2,
            "both_correct": a,
            "model1_only": b,
            "model2_only": c,
            "both_wrong": d,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01
        }

    def plot_comparison(self, output_file: str):
        """Generate bar chart comparison."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed - skipping visualization")
            return

        models = list(self.metrics.keys())
        metrics_names = ['validity_rate', 'exec_accuracy', 'exact_match_rate']
        labels = ['Validity', 'Execution\nAccuracy', 'Exact Match']

        x = range(len(labels))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model in enumerate(models):
            values = [self.metrics[model][m] for m in metrics_names]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], values, width, label=model.replace('_', ' ').title())

            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Percentage (%)')
        ax.set_title('SQL Generation Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison chart to: {output_file}")
        plt.close()

    def plot_difficulty_breakdown(self, output_file: str):
        """Generate difficulty breakdown chart."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed - skipping visualization")
            return

        # Collect data
        all_difficulties = set()
        for m in self.metrics.values():
            all_difficulties.update(m.get('by_difficulty', {}).keys())

        difficulties = sorted(all_difficulties)
        models = list(self.metrics.keys())

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(difficulties))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = []
            for diff in difficulties:
                diff_stats = self.metrics[model].get('by_difficulty', {}).get(diff, {})
                if diff_stats.get('total', 0) > 0:
                    acc = diff_stats['exec_correct'] / diff_stats['total'] * 100
                else:
                    acc = 0
                values.append(acc)

            offset = (i - len(models)/2 + 0.5) * width
            ax.bar([xi + offset for xi in x], values, width, label=model.replace('_', ' ').title())

        ax.set_ylabel('Execution Accuracy (%)')
        ax.set_title('Performance by Query Difficulty')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved difficulty chart to: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare Spider experiment results")
    parser.add_argument("--results-dir", required=True, help="Directory containing results")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results-dir)")
    parser.add_argument("--models", nargs="+", help="Specific models to compare")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--markdown", action="store_true", help="Generate Markdown table")
    parser.add_argument("--errors", action="store_true", help="Generate error analysis")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    parser.add_argument("--significance", nargs=2, metavar=("MODEL1", "MODEL2"),
                        help="Compute statistical significance between two models")
    parser.add_argument("--all", action="store_true", help="Generate all outputs")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.load_results(args.models)

    if not analyzer.metrics:
        print("No results found!")
        sys.exit(1)

    # Always print comparison
    analyzer.print_comparison_table()

    if args.all or args.latex:
        analyzer.generate_latex_table(output_dir / "results_table.tex")

    if args.all or args.markdown:
        analyzer.generate_markdown_table(output_dir / "results_table.md")
        analyzer.generate_difficulty_breakdown(output_dir / "difficulty_breakdown.md")

    if args.all or args.errors:
        analyzer.generate_error_report(output_dir / "error_analysis.md")

    if args.all or args.plots:
        analyzer.plot_comparison(output_dir / "comparison_chart.png")
        analyzer.plot_difficulty_breakdown(output_dir / "difficulty_chart.png")

    if args.significance:
        result = analyzer.statistical_significance(args.significance[0], args.significance[1])
        if result:
            print(f"\nStatistical Significance ({result['model1']} vs {result['model2']}):")
            print(f"  Both correct: {result['both_correct']}")
            print(f"  {result['model1']} only: {result['model1_only']}")
            print(f"  {result['model2']} only: {result['model2_only']}")
            print(f"  Both wrong: {result['both_wrong']}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Significant at 0.05: {result['significant_at_0.05']}")
            print(f"  Significant at 0.01: {result['significant_at_0.01']}")


if __name__ == "__main__":
    main()
