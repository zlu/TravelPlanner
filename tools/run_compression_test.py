#!/usr/bin/env python3
"""
Quick A/B test for token compression strategies.

Tests a small batch of queries (10) with compression enabled,
compares pass rate to ensure no negative impact.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    name: str,
    env_vars: dict,
    start_idx: int = 1,
    max_items: int = 10,
) -> dict:
    """Run an experiment with specific environment variables."""
    
    output_dir = REPO_ROOT / f"evaluation/smt_token_output/compression_test_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base environment
    env = os.environ.copy()
    env.update({
        "TWO_STAGE_SMT": "1",
        "TWO_STAGE_SMT_ONLY": "1",
        "SMT_PLANNER_FULL_DB": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_DATASETS_CACHE": str(REPO_ROOT / ".cache/hf"),
        "SMT_PLANNER_SET_TYPE": "validation",
        "SMT_PLANNER_OUTPUT_ROOT": str(output_dir / "smt_planner"),
    })
    env.update(env_vars)
    
    # Ensure OPENAI_API_KEY is set from DEEPSEEK_API_KEY
    if "DEEPSEEK_API_KEY" in env and "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = env["DEEPSEEK_API_KEY"]
    
    cmd = [
        "conda", "run", "-n", "cerebos", "python",
        str(REPO_ROOT / "agents/tool_agents.py"),
        "--set_type", "validation",
        "--start_idx", str(start_idx),
        "--max_items", str(max_items),
        "--output_dir", str(output_dir),
        "--model_name", "deepseek:deepseek-chat",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"Config: {env_vars}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        success = False
        print("Experiment timed out!")
    except Exception as e:
        success = False
        print(f"Experiment failed: {e}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Count results
    val_dir = output_dir / "validation"
    plans = list(val_dir.glob("generated_plan_*.json")) if val_dir.exists() else []
    
    return {
        "name": name,
        "success": success,
        "elapsed_sec": elapsed,
        "num_plans": len(plans),
        "output_dir": str(output_dir),
    }


def evaluate_results(output_dir: Path, model_name: str = "deepseek:deepseek-chat") -> dict:
    """Count valid SMT plans and estimate pass rate."""
    val_dir = output_dir / "validation"
    if not val_dir.exists():
        return {"error": "No validation directory"}
    
    valid_smt = 0
    total = 0
    
    for f in sorted(val_dir.glob("generated_plan_*.json")):
        if "from_nested" in f.name:
            continue
        total += 1
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data:
                parsed = data[-1].get(f"{model_name}_two-stage_smt_parsed_results")
                if parsed and isinstance(parsed, list) and len(parsed) > 0:
                    valid_smt += 1
        except:
            pass
    
    return {
        "total": total,
        "valid_smt": valid_smt,
        "smt_rate": valid_smt / total * 100 if total > 0 else 0,
    }


def main():
    print("=" * 60)
    print("TOKEN COMPRESSION A/B TEST")
    print("=" * 60)
    print(f"Testing 10 queries each to verify no pass rate impact")
    print()
    
    experiments = [
        # Baseline: no compression
        ("baseline", {}),
        
        # User profile compression (safest)
        ("profile_core", {"USER_PROFILE_MODE": "core"}),
        
        # Tool output compression (already enabled by default)
        # ("tool_compress", {"TOKEN_REDUCTION_ENABLED": "1"}),
    ]
    
    results = []
    
    for name, env_vars in experiments:
        result = run_experiment(name, env_vars, start_idx=1, max_items=10)
        
        # Evaluate
        eval_result = evaluate_results(Path(result["output_dir"]))
        result.update(eval_result)
        results.append(result)
        
        print(f"\n{name}: {eval_result.get('valid_smt', 0)}/{eval_result.get('total', 0)} valid SMT plans")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Plans generated: {r.get('total', 0)}")
        print(f"  Valid SMT: {r.get('valid_smt', 0)}")
        print(f"  SMT Rate: {r.get('smt_rate', 0):.1f}%")
        print(f"  Time: {r.get('elapsed_sec', 0):.0f}s")
        
        if baseline and r["name"] != "baseline":
            delta = r.get("smt_rate", 0) - baseline.get("smt_rate", 0)
            status = "✓ SAFE" if delta >= -5 else "⚠ CHECK"
            print(f"  Delta vs baseline: {delta:+.1f}% {status}")


if __name__ == "__main__":
    main()
