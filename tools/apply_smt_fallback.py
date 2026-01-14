#!/usr/bin/env python3
"""
Apply SMT→LLM fallback strategy to combine best results.

For each query:
1. Use SMT plan if available (guaranteed correct)
2. Fall back to LLM two-stage plan if SMT failed/timed out
3. Generate combined eval file
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.smt_fallback import record_method_stats


def apply_fallback(
    input_dir: Path,
    output_path: Path,
    model_name: str = "deepseek:deepseek-chat",
) -> dict:
    """
    Apply fallback strategy and create combined eval file.
    
    Returns statistics about method usage.
    """
    validation_dir = input_dir / "validation"
    
    stats = {
        "smt_success": 0,
        "llm_fallback": 0,
        "both_failed": 0,
        "total": 0,
    }
    
    output = []
    method_log = []
    
    for idx in range(1, 181):
        plan_path = validation_dir / f"generated_plan_{idx}.json"
        stats["total"] += 1
        
        plan = None
        method = "failed"
        
        try:
            with open(plan_path) as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                last = data[-1]
                
                # Priority 1: SMT parsed results
                smt_key = f"{model_name}_two-stage_smt_parsed_results"
                smt_plan = last.get(smt_key)
                
                if smt_plan and isinstance(smt_plan, list) and len(smt_plan) > 0:
                    plan = smt_plan
                    method = "smt"
                    stats["smt_success"] += 1
                else:
                    # Priority 2: LLM two-stage parsed results
                    llm_key = f"{model_name}_two-stage_parsed_results"
                    llm_plan = last.get(llm_key)
                    
                    if llm_plan and isinstance(llm_plan, list) and len(llm_plan) > 0:
                        plan = llm_plan
                        method = "llm_fallback"
                        stats["llm_fallback"] += 1
                    else:
                        stats["both_failed"] += 1
        except Exception as e:
            stats["both_failed"] += 1
            print(f"Error processing query {idx}: {e}")
        
        output.append({"idx": idx, "plan": plan if plan else [], "method": method})
        method_log.append({"idx": idx, "method": method})
    
    # Write combined eval file
    with open(output_path, 'w') as f:
        for item in output:
            # Remove method from eval output (just idx and plan)
            f.write(json.dumps({"idx": item["idx"], "plan": item["plan"]}) + '\n')
    
    # Write method log
    method_log_path = output_path.parent / "fallback_method_log.jsonl"
    with open(method_log_path, 'w') as f:
        for item in method_log:
            f.write(json.dumps(item) + '\n')
    
    # Calculate percentages
    if stats["total"] > 0:
        stats["smt_rate"] = round(100 * stats["smt_success"] / stats["total"], 1)
        stats["fallback_rate"] = round(100 * stats["llm_fallback"] / stats["total"], 1)
        stats["delivery_rate"] = round(100 * (stats["smt_success"] + stats["llm_fallback"]) / stats["total"], 1)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply SMT→LLM fallback strategy")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=REPO_ROOT / "evaluation/smt_token_output/section_4_2_smt_smoke_policy",
        help="Directory containing generated_plan_*.json files",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output path for combined eval JSONL",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek:deepseek-chat",
        help="Model name prefix for plan keys",
    )
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = args.input_dir / "combined_fallback_eval.jsonl"
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Model name: {args.model_name}")
    print()
    
    stats = apply_fallback(args.input_dir, args.output_path, args.model_name)
    
    print("=== Fallback Statistics ===")
    print(f"Total queries: {stats['total']}")
    print(f"SMT success: {stats['smt_success']} ({stats.get('smt_rate', 0)}%)")
    print(f"LLM fallback: {stats['llm_fallback']} ({stats.get('fallback_rate', 0)}%)")
    print(f"Both failed: {stats['both_failed']}")
    print(f"Delivery rate: {stats.get('delivery_rate', 0)}%")
    print()
    print(f"Combined eval file: {args.output_path}")


if __name__ == "__main__":
    main()
