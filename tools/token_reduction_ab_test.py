#!/usr/bin/env python3
"""
Token Reduction A/B Test - Compare baseline vs optimized token consumption.

Configurations tested:
1. baseline: No compression, full data
2. tool_compression: Tool output column filtering + row limits
3. profile_compression: User profile core/full modes
4. full_optimization: All optimizations enabled

This script analyzes existing experiment outputs to compare token usage
and pass rates across different configurations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken or fallback to word count."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


def analyze_compression_savings(
    original_data: str,
    compressed_data: str,
) -> Dict:
    """Analyze token savings from compression."""
    orig_tokens = count_tokens(original_data)
    comp_tokens = count_tokens(compressed_data)
    
    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "tokens_saved": orig_tokens - comp_tokens,
        "reduction_percent": (1 - comp_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0,
    }


def simulate_tool_compression(
    tool_name: str,
    data: List[Dict],
    max_rows: int = 30,
    columns: List[str] = None,
) -> Tuple[str, str]:
    """Simulate tool output compression and return before/after."""
    original = json.dumps(data, indent=2)
    
    # Apply row limit
    compressed_data = data[:max_rows]
    
    # Apply column filter
    if columns:
        compressed_data = [
            {k: v for k, v in row.items() if k in columns}
            for row in compressed_data
        ]
    
    compressed = json.dumps(compressed_data, indent=2)
    
    return original, compressed


def load_experiment_results(exp_dir: Path) -> Dict:
    """Load experiment results from directory."""
    results = {
        "token_stats": [],
        "eval_results": None,
        "plans": 0,
    }
    
    # Load token stats
    stats_file = exp_dir / "smt_planner" / "token_stats.jsonl"
    if stats_file.exists():
        for line in stats_file.read_text().strip().split("\n"):
            if line:
                try:
                    results["token_stats"].append(json.loads(line))
                except:
                    pass
    
    # Count plans
    val_dir = exp_dir / "validation"
    if val_dir.exists():
        results["plans"] = len(list(val_dir.glob("generated_plan_*.json")))
    
    return results


def compare_configurations() -> Dict:
    """Compare token usage across different experiment configurations."""
    
    base_dir = REPO_ROOT / "evaluation/smt_token_output"
    
    configs = {
        "baseline": base_dir / "section_4_2_smt_smoke_baseline",
        "optimized": base_dir / "section_4_2_smt_smoke_policy",
    }
    
    comparison = {}
    
    for name, exp_dir in configs.items():
        if not exp_dir.exists():
            comparison[name] = {"error": f"Directory not found: {exp_dir}"}
            continue
        
        results = load_experiment_results(exp_dir)
        
        # Calculate token statistics
        stats = results["token_stats"]
        if stats:
            totals = [s.get("total_prompt_tokens", 0) for s in stats if s.get("total_prompt_tokens")]
            comparison[name] = {
                "num_queries": len(stats),
                "num_plans": results["plans"],
                "avg_tokens": sum(totals) / len(totals) if totals else 0,
                "total_tokens": sum(totals),
                "min_tokens": min(totals) if totals else 0,
                "max_tokens": max(totals) if totals else 0,
            }
        else:
            comparison[name] = {
                "num_queries": 0,
                "num_plans": results["plans"],
                "avg_tokens": 0,
            }
    
    # Calculate deltas
    if "baseline" in comparison and "optimized" in comparison:
        baseline = comparison.get("baseline", {})
        optimized = comparison.get("optimized", {})
        
        if baseline.get("avg_tokens") and optimized.get("avg_tokens"):
            comparison["delta"] = {
                "tokens_saved": baseline["avg_tokens"] - optimized["avg_tokens"],
                "reduction_percent": (1 - optimized["avg_tokens"] / baseline["avg_tokens"]) * 100,
            }
    
    return comparison


def estimate_token_reduction_potential() -> Dict:
    """Estimate potential token reduction from different strategies."""
    
    # Typical data sizes from experiments
    typical_data = {
        "flights": {"rows": 50, "columns": 15},  # All columns
        "accommodations": {"rows": 40, "columns": 18},
        "restaurants": {"rows": 60, "columns": 20},
        "attractions": {"rows": 30, "columns": 10},
    }
    
    # Reduced sizes with optimization
    reduced_data = {
        "flights": {"rows": 40, "columns": 7},  # Key columns only
        "accommodations": {"rows": 30, "columns": 7},
        "restaurants": {"rows": 30, "columns": 5},
        "attractions": {"rows": 30, "columns": 3},
    }
    
    # Estimate tokens (rough: 10 tokens per cell)
    tokens_per_cell = 10
    
    estimates = {}
    
    for tool in typical_data:
        orig = typical_data[tool]
        reduced = reduced_data[tool]
        
        orig_tokens = orig["rows"] * orig["columns"] * tokens_per_cell
        reduced_tokens = reduced["rows"] * reduced["columns"] * tokens_per_cell
        
        estimates[tool] = {
            "original_tokens": orig_tokens,
            "reduced_tokens": reduced_tokens,
            "tokens_saved": orig_tokens - reduced_tokens,
            "reduction_percent": (1 - reduced_tokens / orig_tokens) * 100,
        }
    
    # Total
    total_orig = sum(e["original_tokens"] for e in estimates.values())
    total_reduced = sum(e["reduced_tokens"] for e in estimates.values())
    
    estimates["total"] = {
        "original_tokens": total_orig,
        "reduced_tokens": total_reduced,
        "tokens_saved": total_orig - total_reduced,
        "reduction_percent": (1 - total_reduced / total_orig) * 100,
    }
    
    return estimates


def print_report(comparison: Dict, estimates: Dict) -> None:
    """Print comparison report."""
    
    print("=" * 60)
    print("TOKEN REDUCTION A/B TEST REPORT")
    print("=" * 60)
    print()
    
    # Configuration comparison
    print("=== Configuration Comparison ===")
    for name, stats in comparison.items():
        if name == "delta":
            continue
        print(f"\n{name.upper()}:")
        if "error" in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Queries: {stats.get('num_queries', 0)}")
            print(f"  Plans: {stats.get('num_plans', 0)}")
            print(f"  Avg tokens/query: {stats.get('avg_tokens', 0):,.0f}")
    
    if "delta" in comparison:
        delta = comparison["delta"]
        print(f"\nDELTA (baseline → optimized):")
        print(f"  Tokens saved: {delta.get('tokens_saved', 0):,.0f}")
        print(f"  Reduction: {delta.get('reduction_percent', 0):.1f}%")
    
    print()
    print("=== Token Reduction Potential (Estimates) ===")
    for tool, est in estimates.items():
        if tool == "total":
            print(f"\nTOTAL:")
        else:
            print(f"\n{tool}:")
        print(f"  Original: {est['original_tokens']:,} tokens")
        print(f"  Reduced: {est['reduced_tokens']:,} tokens")
        print(f"  Savings: {est['tokens_saved']:,} ({est['reduction_percent']:.0f}%)")
    
    print()
    print("=== Recommendations ===")
    print("1. Enable tool output compression (TOOL_MAX_ROWS, TOOL_COLUMN_ALLOWLIST)")
    print("2. Use user profile core mode (USER_PROFILE_MODE=core)")
    print("3. Apply database pre-filtering for multi-city queries")
    print("4. Target: 30%+ token reduction while maintaining pass rate")


def main():
    comparison = compare_configurations()
    estimates = estimate_token_reduction_potential()
    print_report(comparison, estimates)


if __name__ == "__main__":
    main()
