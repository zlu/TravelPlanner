#!/usr/bin/env python3
"""
Token Statistics Analysis Tool

Analyzes token consumption across different experiments and identifies
optimization opportunities for the travel planner pipeline.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


def load_token_stats(stats_file: Path) -> List[Dict[str, Any]]:
    """Load token stats from a JSONL file."""
    records = []
    if stats_file.exists():
        with stats_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def analyze_step_tokens(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Analyze token consumption by step type."""
    step_stats = {}
    
    for rec in records:
        step_tokens = rec.get("step_to_code_prompt_tokens", {})
        for step_name, tokens in step_tokens.items():
            if step_name not in step_stats:
                step_stats[step_name] = []
            step_stats[step_name].append(tokens)
    
    result = {}
    for step_name, token_list in step_stats.items():
        if token_list:
            result[step_name] = {
                "mean": statistics.mean(token_list),
                "median": statistics.median(token_list),
                "min": min(token_list),
                "max": max(token_list),
                "std": statistics.stdev(token_list) if len(token_list) > 1 else 0,
                "count": len(token_list),
            }
    
    return result


def analyze_experiment(experiment_dir: Path) -> Dict[str, Any]:
    """Analyze token stats for an experiment."""
    # Look for token_stats.jsonl in common locations
    possible_paths = [
        experiment_dir / "token_stats.jsonl",
        experiment_dir / "smt_planner" / "token_stats.jsonl",
    ]
    
    stats_file = None
    for p in possible_paths:
        if p.exists():
            stats_file = p
            break
    
    if not stats_file:
        return {"error": f"No token_stats.jsonl found in {experiment_dir}"}
    
    records = load_token_stats(stats_file)
    if not records:
        return {"error": f"No records in {stats_file}"}
    
    # Aggregate statistics
    total_tokens = [r.get("total_prompt_tokens", 0) for r in records if r.get("total_prompt_tokens")]
    constraint_tokens = [r.get("constraint_to_step_prompt_tokens", 0) for r in records if r.get("constraint_to_step_prompt_tokens")]
    
    analysis = {
        "experiment": str(experiment_dir),
        "num_queries": len(records),
        "stats_file": str(stats_file),
    }
    
    if total_tokens:
        analysis["total_tokens"] = {
            "mean": statistics.mean(total_tokens),
            "median": statistics.median(total_tokens),
            "min": min(total_tokens),
            "max": max(total_tokens),
            "sum": sum(total_tokens),
        }
    
    if constraint_tokens:
        analysis["constraint_tokens"] = {
            "mean": statistics.mean(constraint_tokens),
            "median": statistics.median(constraint_tokens),
            "min": min(constraint_tokens),
            "max": max(constraint_tokens),
            "sum": sum(constraint_tokens),
        }
    
    # Step-by-step analysis
    step_analysis = analyze_step_tokens(records)
    if step_analysis:
        analysis["step_breakdown"] = step_analysis
        
        # Calculate step contribution percentages
        total_step_tokens = sum(s["mean"] for s in step_analysis.values())
        if total_step_tokens > 0:
            analysis["step_contribution"] = {
                step_name: round(s["mean"] / total_step_tokens * 100, 1)
                for step_name, s in sorted(step_analysis.items(), key=lambda x: -x[1]["mean"])
            }
    
    return analysis


def compare_experiments(experiments: List[Path]) -> Dict[str, Any]:
    """Compare token consumption across experiments."""
    results = {}
    
    for exp_dir in experiments:
        exp_name = exp_dir.name
        results[exp_name] = analyze_experiment(exp_dir)
    
    # Calculate savings
    if len(results) >= 2:
        baselines = {}
        for name, analysis in results.items():
            if "total_tokens" in analysis:
                baselines[name] = analysis["total_tokens"]["mean"]
        
        if baselines:
            max_tokens = max(baselines.values())
            for name, tokens in baselines.items():
                results[name]["vs_highest"] = {
                    "reduction_pct": round((max_tokens - tokens) / max_tokens * 100, 1),
                    "tokens_saved_per_query": round(max_tokens - tokens, 0),
                }
    
    return results


def estimate_cost(total_tokens: int, model: str = "deepseek-chat") -> float:
    """Estimate API cost based on token count."""
    # Cost per 1K tokens (approximate)
    pricing = {
        "deepseek-chat": 0.0002,  # Very cheap
        "gpt-3.5-turbo": 0.0015,
        "gpt-4": 0.03,
        "gpt-4o": 0.005,
    }
    rate = pricing.get(model, 0.001)
    return (total_tokens / 1000) * rate


def generate_report(analysis: Dict[str, Any], model: str = "deepseek-chat") -> str:
    """Generate a human-readable report."""
    lines = ["# Token Statistics Analysis Report\n"]
    
    for exp_name, data in analysis.items():
        lines.append(f"\n## Experiment: {exp_name}\n")
        
        if "error" in data:
            lines.append(f"**Error**: {data['error']}\n")
            continue
        
        lines.append(f"- Queries analyzed: {data.get('num_queries', 'N/A')}")
        
        if "total_tokens" in data:
            tt = data["total_tokens"]
            lines.append(f"\n### Total Tokens per Query")
            lines.append(f"- Mean: {tt['mean']:,.0f}")
            lines.append(f"- Median: {tt['median']:,.0f}")
            lines.append(f"- Range: {tt['min']:,.0f} - {tt['max']:,.0f}")
            lines.append(f"- Total across all queries: {tt['sum']:,.0f}")
            
            # Cost estimate
            cost = estimate_cost(tt['sum'], model)
            lines.append(f"\n### Estimated Cost ({model})")
            lines.append(f"- Total: ${cost:.4f}")
            lines.append(f"- Per query: ${cost / data.get('num_queries', 1):.6f}")
        
        if "step_contribution" in data:
            lines.append(f"\n### Token Contribution by Step")
            lines.append("| Step | % of Total |")
            lines.append("|------|-----------|")
            for step, pct in data["step_contribution"].items():
                lines.append(f"| {step} | {pct}% |")
        
        if "vs_highest" in data:
            vh = data["vs_highest"]
            lines.append(f"\n### Compared to Baseline")
            lines.append(f"- Reduction: {vh['reduction_pct']}%")
            lines.append(f"- Tokens saved per query: {vh['tokens_saved_per_query']:,.0f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze token statistics")
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=Path,
        help="Experiment directories to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output file for markdown report",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name for cost estimation",
    )
    args = parser.parse_args()
    
    # Default to analyzing common experiment directories
    if not args.experiments:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "smt_token_output" / "hybrid_two_stage_smt",
            repo_root / "smt_token_output" / "smt_planner",
            repo_root / "evaluation" / "smt_token_output" / "section_4_2_smt_smoke_policy",
        ]
        args.experiments = [c for c in candidates if c.exists()]
    
    if not args.experiments:
        print("No experiment directories found or specified.")
        return
    
    # Run analysis
    analysis = compare_experiments(args.experiments)
    
    # Output JSON
    if args.output:
        with args.output.open("w") as f:
            json.dump(analysis, f, indent=2)
        print(f"JSON results saved to {args.output}")
    else:
        print(json.dumps(analysis, indent=2))
    
    # Generate report
    report = generate_report(analysis, args.model)
    if args.report:
        with args.report.open("w") as f:
            f.write(report)
        print(f"Report saved to {args.report}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
