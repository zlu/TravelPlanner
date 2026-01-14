#!/usr/bin/env python3
"""
Generate paper figures and tables for the TravelPlanner SMT research paper.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Color palette
COLORS = {
    'baseline': '#4A90A4',     # Blue
    'two_stage': '#5BA55B',    # Green
    'smt_hybrid': '#D4A54D',   # Gold
    'smt_only': '#C75050',     # Red
}


def load_eval_results(results_file: Path) -> Dict[str, Any]:
    """Parse evaluation results from text file."""
    if not results_file.exists():
        return {}
    
    content = results_file.read_text()
    metrics = {}
    
    for line in content.split('\n'):
        if ':' in line:
            for key in ['Delivery Rate', 'Final Pass Rate', 
                       'Commonsense Constraint Micro Pass Rate',
                       'Hard Constraint Micro Pass Rate']:
                if key in line:
                    try:
                        value = float(line.split(':')[1].strip().rstrip('%'))
                        metrics[key] = value
                    except (ValueError, IndexError):
                        pass
    
    return metrics


def create_pass_rate_comparison(output_dir: Path, results: Dict[str, Dict[str, float]]):
    """Create bar chart comparing pass rates across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    metrics = ['Delivery Rate', 'Final Pass Rate', 
               'Commonsense Constraint Micro Pass Rate',
               'Hard Constraint Micro Pass Rate']
    
    x = range(len(metrics))
    width = 0.8 / len(methods)
    
    colors = list(COLORS.values())[:len(methods)]
    
    for i, (method, data) in enumerate(results.items()):
        values = [data.get(m, 0) for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, 
                     label=method, color=colors[i % len(colors)])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('TravelPlanner Constraint Satisfaction Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Delivery', 'Final Pass', 'Commonsense\n(Micro)', 'Hard\n(Micro)'])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'pass_rate_comparison.png', dpi=300)
    fig.savefig(output_dir / 'pass_rate_comparison.pdf')
    plt.close(fig)
    print(f"Saved pass rate comparison to {output_dir}")


def create_token_distribution_chart(output_dir: Path, token_stats: Dict[str, Dict]):
    """Create pie chart showing token distribution by step."""
    if not token_stats:
        return
    
    # Use first experiment with step breakdown
    for exp_name, data in token_stats.items():
        if 'step_contribution' in data:
            contributions = data['step_contribution']
            break
    else:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(contributions.keys())
    sizes = list(contributions.values())
    
    # Color gradient
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * (1 - i/len(labels))) for i in range(len(labels))]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.75
    )
    
    ax.set_title('Token Distribution by Processing Step\n(SMT Pipeline)', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'token_distribution.png', dpi=300)
    fig.savefig(output_dir / 'token_distribution.pdf')
    plt.close(fig)
    print(f"Saved token distribution to {output_dir}")


def create_cost_comparison_table(output_dir: Path, token_stats: Dict[str, Dict]):
    """Create cost comparison table."""
    rows = []
    
    for exp_name, data in token_stats.items():
        if 'total_tokens' not in data:
            continue
        
        tt = data['total_tokens']
        num_queries = data.get('num_queries', 1)
        
        # Cost per 1K tokens for different models
        costs = {
            'DeepSeek': 0.0002,
            'GPT-3.5': 0.0015,
            'GPT-4o': 0.005,
        }
        
        row = {
            'Experiment': exp_name,
            'Queries': num_queries,
            'Avg Tokens': f"{tt['mean']:,.0f}",
            'Total Tokens': f"{tt['sum']:,.0f}",
        }
        
        for model, rate in costs.items():
            cost = (tt['sum'] / 1000) * rate
            row[f'{model} Cost'] = f"${cost:.4f}"
        
        rows.append(row)
    
    # Write as markdown table
    if rows:
        headers = list(rows[0].keys())
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
        
        (output_dir / 'cost_comparison_table.md').write_text(md)
        print(f"Saved cost comparison table to {output_dir}")


def create_constraint_breakdown_chart(output_dir: Path, eval_results: Dict[str, Any]):
    """Create detailed constraint breakdown chart."""
    # Parse the detailed_scores if available
    # This would require parsing the full JSON output from eval.py
    pass


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("evaluation/smt_token_output/section_4_2_smt_smoke_policy"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir or (args.experiment_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation results
    results = {}
    
    two_stage_results = args.experiment_dir / "two_stage_eval_results.txt"
    if two_stage_results.exists():
        results['Two-Stage (DeepSeek)'] = load_eval_results(two_stage_results)
    
    smt_results = args.experiment_dir / "smt_hybrid_eval_results.txt"
    if smt_results.exists():
        results['SMT Hybrid'] = load_eval_results(smt_results)
    
    # Add baseline for reference
    results['TravelPlanner Baseline'] = {
        'Delivery Rate': 90.0,  # Approximate from paper
        'Final Pass Rate': 6.0,
        'Commonsense Constraint Micro Pass Rate': 65.0,
        'Hard Constraint Micro Pass Rate': 25.0,
    }
    
    if results:
        create_pass_rate_comparison(output_dir, results)
    
    # Load token statistics
    token_stats_files = [
        args.experiment_dir / "smt_planner" / "token_stats.jsonl",
        args.experiment_dir.parent.parent.parent / "smt_token_output" / "hybrid_two_stage_smt" / "token_stats.jsonl",
    ]
    
    token_stats = {}
    for f in token_stats_files:
        if f.exists():
            records = []
            for line in f.read_text().strip().split('\n'):
                if line:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
            if records:
                # Analyze
                import statistics
                total_tokens = [r.get('total_prompt_tokens', 0) for r in records if r.get('total_prompt_tokens')]
                if total_tokens:
                    step_breakdown = {}
                    for r in records:
                        for step, tokens in r.get('step_to_code_prompt_tokens', {}).items():
                            step_breakdown.setdefault(step, []).append(tokens)
                    
                    step_contribution = {}
                    total_step = sum(statistics.mean(v) for v in step_breakdown.values())
                    for step, tokens in step_breakdown.items():
                        step_contribution[step] = round(statistics.mean(tokens) / total_step * 100, 1) if total_step else 0
                    
                    token_stats[f.parent.name] = {
                        'total_tokens': {
                            'mean': statistics.mean(total_tokens),
                            'sum': sum(total_tokens),
                        },
                        'num_queries': len(records),
                        'step_contribution': dict(sorted(step_contribution.items(), key=lambda x: -x[1])),
                    }
    
    if token_stats:
        create_token_distribution_chart(output_dir, token_stats)
        create_cost_comparison_table(output_dir, token_stats)
    
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
