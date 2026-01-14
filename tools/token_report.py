#!/usr/bin/env python3
"""
Token Report Generator - Analyze token consumption for paper.

Aggregates token statistics from:
1. token_stats.jsonl - Per-query token counts by step
2. token_audit/*.json - Per-query tool call details with compression stats
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_token_stats(stats_file: Path) -> List[Dict]:
    """Load token stats from JSONL file."""
    records = []
    if not stats_file.exists():
        return records
    
    for line in stats_file.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    
    return records


def load_token_audits(audit_dir: Path) -> List[Dict]:
    """Load token audits from JSON files."""
    records = []
    if not audit_dir.exists():
        return records
    
    for f in sorted(audit_dir.glob("token_audit_*.json")):
        try:
            data = json.loads(f.read_text())
            records.append(data)
        except Exception:
            pass
    
    return records


def analyze_token_stats(records: List[Dict]) -> Dict:
    """Analyze token stats and generate summary."""
    if not records:
        return {"error": "No token stats found"}
    
    # Aggregate by step type
    step_totals = defaultdict(list)
    
    for rec in records:
        # Constraint to step
        c2s = rec.get("constraint_to_step_prompt_tokens")
        if c2s:
            step_totals["constraint_to_step"].append(c2s)
        
        # Query to JSON
        q2j = rec.get("query_to_json_prompt_tokens")
        if q2j:
            step_totals["query_to_json"].append(q2j)
        
        # Step to code (multiple sub-steps)
        s2c = rec.get("step_to_code_prompt_tokens", {})
        if isinstance(s2c, dict):
            for step_name, tokens in s2c.items():
                if tokens:
                    step_totals[f"step_to_code_{step_name}"].append(tokens)
        
        # Total
        total = rec.get("total_prompt_tokens")
        if total:
            step_totals["total"].append(total)
    
    # Calculate statistics
    stats = {}
    for step, values in step_totals.items():
        if values:
            stats[step] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values),
            }
    
    return {
        "num_queries": len(records),
        "by_step": stats,
    }


def analyze_token_audits(records: List[Dict]) -> Dict:
    """Analyze token audits for compression effectiveness."""
    if not records:
        return {"error": "No token audits found"}
    
    # Tool usage stats
    tool_calls = defaultdict(int)
    cache_stats = {"hits": 0, "misses": 0}
    
    # Compression stats
    raw_tokens_total = 0
    compressed_tokens_total = 0
    
    for rec in records:
        cache_stats["hits"] += rec.get("cache_hits", 0)
        cache_stats["misses"] += rec.get("cache_misses", 0)
        
        notebook = rec.get("notebook_tokens", {})
        raw_tokens_total += notebook.get("raw_tokens", 0)
        compressed_tokens_total += notebook.get("compressed_tokens", 0)
        
        for entry in rec.get("entries", []):
            tool = entry.get("tool", "unknown")
            tool_calls[tool] += 1
    
    # Calculate compression ratio
    compression_ratio = 0
    if raw_tokens_total > 0:
        compression_ratio = 1 - (compressed_tokens_total / raw_tokens_total)
    
    return {
        "num_queries": len(records),
        "tool_calls": dict(tool_calls),
        "cache_stats": cache_stats,
        "compression": {
            "raw_tokens_total": raw_tokens_total,
            "compressed_tokens_total": compressed_tokens_total,
            "reduction_ratio": compression_ratio,
            "tokens_saved": raw_tokens_total - compressed_tokens_total,
        },
    }


def generate_report(output_dir: Path, model_name: str = "deepseek") -> Dict:
    """Generate comprehensive token report."""
    
    # Find all token stats files
    stats_files = list(output_dir.rglob("token_stats.jsonl"))
    audit_dirs = list(output_dir.rglob("token_audit"))
    
    report = {
        "output_dir": str(output_dir),
        "model": model_name,
        "stats_files_found": len(stats_files),
        "audit_dirs_found": len(audit_dirs),
        "token_stats": {},
        "token_audits": {},
        "summary": {},
    }
    
    # Analyze token stats
    all_stats_records = []
    for sf in stats_files:
        records = load_token_stats(sf)
        all_stats_records.extend(records)
        report["token_stats"][str(sf.relative_to(output_dir))] = {
            "count": len(records),
        }
    
    if all_stats_records:
        report["summary"]["token_stats"] = analyze_token_stats(all_stats_records)
    
    # Analyze token audits
    all_audit_records = []
    for ad in audit_dirs:
        validation_dir = ad / "validation"
        if validation_dir.exists():
            records = load_token_audits(validation_dir)
            all_audit_records.extend(records)
            report["token_audits"][str(ad.relative_to(output_dir))] = {
                "count": len(records),
            }
    
    if all_audit_records:
        report["summary"]["token_audits"] = analyze_token_audits(all_audit_records)
    
    # Generate cost estimates
    if "token_stats" in report["summary"]:
        stats = report["summary"]["token_stats"]
        if "by_step" in stats and "total" in stats["by_step"]:
            total_tokens = stats["by_step"]["total"]["total"]
            num_queries = stats["by_step"]["total"]["count"]
            avg_tokens = total_tokens / num_queries if num_queries > 0 else 0
            
            # Cost per 1M tokens (approximate)
            costs = {
                "deepseek": 0.14,  # $0.14 per 1M input tokens
                "gpt-3.5": 0.50,
                "gpt-4o": 2.50,
            }
            
            report["summary"]["cost_estimate"] = {
                "total_tokens": total_tokens,
                "avg_tokens_per_query": round(avg_tokens, 0),
                "cost_per_180_queries": {
                    model: round(total_tokens / 1_000_000 * cost * (180 / num_queries), 4)
                    for model, cost in costs.items()
                },
            }
    
    return report


def print_report(report: Dict) -> None:
    """Print formatted report."""
    print("=" * 60)
    print("TOKEN CONSUMPTION REPORT")
    print("=" * 60)
    print(f"Output directory: {report['output_dir']}")
    print(f"Stats files found: {report['stats_files_found']}")
    print(f"Audit dirs found: {report['audit_dirs_found']}")
    print()
    
    summary = report.get("summary", {})
    
    # Token stats summary
    if "token_stats" in summary:
        stats = summary["token_stats"]
        print("=== Token Stats by Step ===")
        print(f"Total queries analyzed: {stats.get('num_queries', 0)}")
        print()
        
        by_step = stats.get("by_step", {})
        if by_step:
            print(f"{'Step':<30} {'Mean':>10} {'Min':>10} {'Max':>10}")
            print("-" * 60)
            for step, values in sorted(by_step.items()):
                print(f"{step:<30} {values['mean']:>10.0f} {values['min']:>10} {values['max']:>10}")
        print()
    
    # Token audit summary
    if "token_audits" in summary:
        audit = summary["token_audits"]
        print("=== Token Compression Analysis ===")
        print(f"Queries audited: {audit.get('num_queries', 0)}")
        
        comp = audit.get("compression", {})
        if comp:
            print(f"Raw tokens total: {comp.get('raw_tokens_total', 0):,}")
            print(f"Compressed tokens: {comp.get('compressed_tokens_total', 0):,}")
            print(f"Tokens saved: {comp.get('tokens_saved', 0):,}")
            print(f"Compression ratio: {comp.get('reduction_ratio', 0):.1%}")
        
        cache = audit.get("cache_stats", {})
        if cache:
            total = cache.get("hits", 0) + cache.get("misses", 0)
            hit_rate = cache["hits"] / total * 100 if total > 0 else 0
            print(f"Cache hit rate: {hit_rate:.1f}% ({cache['hits']}/{total})")
        print()
    
    # Cost estimates
    if "cost_estimate" in summary:
        est = summary["cost_estimate"]
        print("=== Cost Estimates (180 queries) ===")
        print(f"Avg tokens per query: {est.get('avg_tokens_per_query', 0):,.0f}")
        for model, cost in est.get("cost_per_180_queries", {}).items():
            print(f"  {model}: ${cost:.4f}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate token consumption report")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "evaluation/smt_token_output/section_4_2_smt_smoke_policy",
        help="Directory containing token stats",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Output JSON report path",
    )
    
    args = parser.parse_args()
    
    report = generate_report(args.output_dir)
    print_report(report)
    
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to: {args.json_out}")


if __name__ == "__main__":
    main()
