#!/usr/bin/env python3
"""
Generate LaTeX tables for the TravelPlanner SMT research paper.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def parse_eval_results(results_file: Path) -> Dict[str, float]:
    """Parse evaluation results from a text file."""
    if not results_file.exists():
        return {}
    
    content = results_file.read_text()
    metrics = {}
    
    for line in content.split('\n'):
        if ':' in line:
            for key in ['Delivery Rate', 'Final Pass Rate', 
                       'Commonsense Constraint Micro Pass Rate',
                       'Commonsense Constraint Macro Pass Rate',
                       'Hard Constraint Micro Pass Rate',
                       'Hard Constraint Macro Pass Rate']:
                if key in line:
                    try:
                        value = float(line.split(':')[1].strip().rstrip('%'))
                        metrics[key] = value
                    except:
                        pass
    
    return metrics


def generate_main_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate main results comparison table."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance comparison on TravelPlanner validation set}
\label{tab:main_results}
\begin{tabular}{l|c|c|c|c}
\toprule
\textbf{Method} & \textbf{Delivery} & \textbf{Final Pass} & \textbf{CS Micro} & \textbf{HC Micro} \\
\midrule
"""
    
    for method, metrics in results.items():
        delivery = metrics.get('Delivery Rate', 0)
        final_pass = metrics.get('Final Pass Rate', 0)
        cs_micro = metrics.get('Commonsense Constraint Micro Pass Rate', 0)
        hc_micro = metrics.get('Hard Constraint Micro Pass Rate', 0)
        
        latex += f"{method} & {delivery:.1f}\\% & {final_pass:.1f}\\% & {cs_micro:.1f}\\% & {hc_micro:.1f}\\% \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_token_consumption_table(token_stats: Dict[str, Dict]) -> str:
    """Generate token consumption comparison table."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Token consumption analysis per query}
\label{tab:token_consumption}
\begin{tabular}{l|r|r|r}
\toprule
\textbf{Method} & \textbf{Avg Tokens} & \textbf{Total Tokens} & \textbf{Est. Cost (DeepSeek)} \\
\midrule
"""
    
    for method, stats in token_stats.items():
        if 'total_tokens' not in stats:
            continue
        tt = stats['total_tokens']
        avg = tt.get('mean', 0)
        total = tt.get('sum', 0)
        cost = (total / 1000) * 0.0002  # DeepSeek rate
        
        latex += f"{method} & {avg:,.0f} & {total:,.0f} & \\${cost:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_step_breakdown_table(step_contribution: Dict[str, float]) -> str:
    """Generate step-by-step token breakdown table."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Token distribution by processing step (SMT pipeline)}
\label{tab:step_breakdown}
\begin{tabular}{l|r}
\toprule
\textbf{Processing Step} & \textbf{\% of Total} \\
\midrule
"""
    
    for step, pct in sorted(step_contribution.items(), key=lambda x: -x[1]):
        latex += f"{step} & {pct:.1f}\\% \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_constraint_breakdown_table(detailed_scores: Dict) -> str:
    """Generate constraint breakdown table by difficulty and days."""
    latex = r"""
\begin{table*}[htbp]
\centering
\caption{Constraint satisfaction breakdown by difficulty level and trip duration}
\label{tab:constraint_breakdown}
\begin{tabular}{l|l|ccc|ccc}
\toprule
\textbf{Constraint Type} & \textbf{Constraint} & \multicolumn{3}{c|}{\textbf{Easy}} & \multicolumn{3}{c}{\textbf{Medium}} \\
 & & 3-day & 5-day & 7-day & 3-day & 5-day & 7-day \\
\midrule
"""
    
    # Placeholder - would need actual detailed scores to fill in
    constraints = [
        "Reasonable City Route",
        "Diverse Restaurants",
        "Diverse Attractions",
        "Minimum Nights Stay",
        "Non-conf. Transportation",
        "Within Current City",
        "Within Sandbox",
        "Complete Information",
    ]
    
    for constraint in constraints:
        latex += f"Commonsense & {constraint} & -- & -- & -- & -- & -- & -- \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


def generate_all_tables(experiment_dir: Path, output_dir: Path):
    """Generate all LaTeX tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    results = {}
    
    # Two-stage results
    two_stage_file = experiment_dir / "two_stage_eval_results.txt"
    if two_stage_file.exists():
        results['Two-Stage (DeepSeek)'] = parse_eval_results(two_stage_file)
    
    # SMT hybrid results
    smt_file = experiment_dir / "smt_hybrid_eval_results.txt"
    if smt_file.exists():
        results['SMT Hybrid'] = parse_eval_results(smt_file)
    
    # Add baseline reference
    results['TravelPlanner Baseline'] = {
        'Delivery Rate': 90.0,
        'Final Pass Rate': 6.0,
        'Commonsense Constraint Micro Pass Rate': 65.0,
        'Hard Constraint Micro Pass Rate': 25.0,
    }
    
    # Generate main results table
    main_table = generate_main_results_table(results)
    (output_dir / "main_results.tex").write_text(main_table)
    print(f"Generated {output_dir / 'main_results.tex'}")
    
    # Load token statistics
    token_stats = {}
    for stats_file in experiment_dir.glob("**/token_stats.jsonl"):
        records = []
        for line in stats_file.read_text().strip().split('\n'):
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
        
        if records:
            import statistics
            total_tokens = [r.get('total_prompt_tokens', 0) for r in records if r.get('total_prompt_tokens')]
            if total_tokens:
                token_stats[stats_file.parent.name] = {
                    'total_tokens': {
                        'mean': statistics.mean(total_tokens),
                        'sum': sum(total_tokens),
                    },
                    'num_queries': len(records),
                }
    
    if token_stats:
        token_table = generate_token_consumption_table(token_stats)
        (output_dir / "token_consumption.tex").write_text(token_table)
        print(f"Generated {output_dir / 'token_consumption.tex'}")
    
    # Generate step breakdown if available
    for name, stats in token_stats.items():
        if 'step_contribution' in stats:
            step_table = generate_step_breakdown_table(stats['step_contribution'])
            (output_dir / "step_breakdown.tex").write_text(step_table)
            print(f"Generated {output_dir / 'step_breakdown.tex'}")
            break
    
    # Generate all tables combined file
    all_tables = "% Auto-generated LaTeX tables for TravelPlanner SMT paper\n\n"
    for tex_file in output_dir.glob("*.tex"):
        all_tables += f"% === {tex_file.name} ===\n"
        all_tables += tex_file.read_text()
        all_tables += "\n\n"
    
    (output_dir / "all_tables.tex").write_text(all_tables)
    print(f"Generated {output_dir / 'all_tables.tex'}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
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
    
    output_dir = args.output_dir or (args.experiment_dir / "latex_tables")
    generate_all_tables(args.experiment_dir, output_dir)


if __name__ == "__main__":
    main()
