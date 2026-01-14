"""
SMT Fallback Strategy - Use LLM plans when SMT times out.

This provides a hybrid approach:
1. Try SMT solver first (guaranteed correct if succeeds)
2. Fall back to LLM plan if SMT times out
3. Record which method was used for analysis
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def should_use_fallback() -> bool:
    """Check if fallback to LLM is enabled."""
    return os.getenv("SMT_FALLBACK_ENABLED", "1") == "1"


def get_llm_plan(generated_plan_path: Path, model_name: str = "deepseek:deepseek-chat") -> Optional[Dict]:
    """
    Extract LLM-generated plan from the generated_plan file.
    
    Returns the parsed plan or None if not available.
    """
    if not generated_plan_path.exists():
        return None
    
    try:
        data = json.loads(generated_plan_path.read_text())
        if not isinstance(data, list) or not data:
            return None
        
        last = data[-1]
        
        # Try to get two-stage parsed results
        key = f"{model_name}_two-stage_parsed_results"
        plan = last.get(key)
        
        if plan:
            return {
                "plan": plan,
                "method": "llm_two_stage",
                "model": model_name,
            }
        
        return None
    except Exception:
        return None


def merge_with_fallback(
    smt_result: Optional[Dict],
    llm_result: Optional[Dict],
    smt_status: str,
) -> Tuple[Optional[Dict], str]:
    """
    Merge SMT and LLM results based on SMT status.
    
    Returns:
        Tuple of (final_plan, method_used)
    """
    if smt_status == "sat" and smt_result:
        return smt_result, "smt"
    
    if should_use_fallback() and llm_result:
        return llm_result["plan"], "llm_fallback"
    
    return None, "failed"


def record_method_stats(
    output_path: Path,
    index: int,
    smt_status: str,
    method_used: str,
    smt_time: float = 0,
) -> None:
    """Record which method was used for each query."""
    stats_file = output_path / "method_stats.jsonl"
    
    record = {
        "index": index,
        "smt_status": smt_status,
        "method_used": method_used,
        "smt_time_sec": round(smt_time, 2),
    }
    
    with stats_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def analyze_fallback_usage(stats_file: Path) -> Dict[str, Any]:
    """Analyze how often fallback was used."""
    if not stats_file.exists():
        return {}
    
    stats = {"smt": 0, "llm_fallback": 0, "failed": 0, "total": 0}
    smt_times = []
    
    for line in stats_file.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            record = json.loads(line)
            method = record.get("method_used", "failed")
            stats[method] = stats.get(method, 0) + 1
            stats["total"] += 1
            
            if record.get("smt_time_sec"):
                smt_times.append(record["smt_time_sec"])
        except:
            pass
    
    if smt_times:
        stats["avg_smt_time"] = sum(smt_times) / len(smt_times)
    
    if stats["total"] > 0:
        stats["smt_success_rate"] = stats["smt"] / stats["total"] * 100
        stats["fallback_rate"] = stats["llm_fallback"] / stats["total"] * 100
    
    return stats


# Configuration for paper
FALLBACK_CONFIG = {
    "enabled": should_use_fallback(),
    "priority": ["smt", "llm_two_stage"],
    "description": "SMT-first with LLM fallback for timeout cases",
}
