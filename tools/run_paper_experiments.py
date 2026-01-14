#!/usr/bin/env python3
"""
Run all experiments needed for the TravelPlanner SMT paper.

This script orchestrates:
1. Two-stage tool agent baseline
2. SMT-only planning
3. Hybrid (Two-stage + SMT) planning
4. Ablation studies for token optimization

Usage:
    python tools/run_paper_experiments.py --experiment all
    python tools/run_paper_experiments.py --experiment smt_only --start_idx 1 --max_items 10
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run_command(cmd: List[str], env: Optional[dict] = None, timeout: int = 3600) -> bool:
    """Run a command and return success status."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=full_env,
            timeout=timeout,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s")
        return False
    except Exception as e:
        log(f"Error running command: {e}")
        return False


def run_two_stage_baseline(
    set_type: str = "validation",
    model_name: str = "deepseek:deepseek-chat",
    start_idx: int = 1,
    max_items: int = 180,
    output_dir: str = "evaluation/smt_token_output/two_stage_baseline",
) -> bool:
    """Run two-stage tool agent baseline."""
    log(f"=== Running Two-Stage Baseline ===")
    
    env = {
        "OPENAI_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "PYTHONUNBUFFERED": "1",
        "HF_DATASETS_CACHE": str(REPO_ROOT / ".cache/hf"),
        "TOKEN_REDUCTION": "1",
    }
    
    cmd = [
        "python", "agents/tool_agents.py",
        "--set_type", set_type,
        "--model_name", model_name,
        "--start_idx", str(start_idx),
        "--max_items", str(max_items),
        "--output_dir", output_dir,
    ]
    
    return run_command(cmd, env)


def run_smt_only(
    set_type: str = "validation",
    start_idx: int = 1,
    max_items: int = 180,
    output_dir: str = "smt_token_output/smt_only",
) -> bool:
    """Run SMT-only planning (no two-stage tool retrieval)."""
    log(f"=== Running SMT-Only Planning ===")
    
    env = {
        "OPENAI_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "PYTHONUNBUFFERED": "1",
        "HF_DATASETS_CACHE": str(REPO_ROOT / ".cache/hf"),
        "SMT_PLANNER_FULL_DB": "1",  # Use full database
    }
    
    cmd = [
        "python", "tools/hybrid_two_stage_smt.py",
        "--set_type", set_type,
        "--start_idx", str(start_idx),
        "--max_items", str(max_items),
        "--output_root", output_dir,
        "--full_db",  # Use full database for SMT-only
    ]
    
    return run_command(cmd, env)


def run_hybrid_smt(
    set_type: str = "validation",
    start_idx: int = 1,
    max_items: int = 180,
    output_dir: str = "evaluation/smt_token_output/hybrid_smt",
) -> bool:
    """Run hybrid two-stage + SMT planning."""
    log(f"=== Running Hybrid Two-Stage SMT ===")
    
    env = {
        "OPENAI_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "PYTHONUNBUFFERED": "1",
        "HF_DATASETS_CACHE": str(REPO_ROOT / ".cache/hf"),
        "TWO_STAGE_SMT": "1",
        "TWO_STAGE_SMT_ONLY": "1",
    }
    
    cmd = [
        "python", "agents/tool_agents.py",
        "--set_type", set_type,
        "--model_name", "deepseek:deepseek-chat",
        "--start_idx", str(start_idx),
        "--max_items", str(max_items),
        "--output_dir", output_dir,
    ]
    
    return run_command(cmd, env)


def run_token_optimization_ablation(
    set_type: str = "validation",
    start_idx: int = 1,
    max_items: int = 30,
) -> bool:
    """Run ablation study for token optimization strategies."""
    log(f"=== Running Token Optimization Ablation ===")
    
    base_output = "smt_token_output/ablation"
    results = {}
    
    # Configuration variants
    configs = [
        {
            "name": "baseline_no_reduction",
            "env": {"TOKEN_REDUCTION": "0", "USER_PROFILE_MODE": "off"},
        },
        {
            "name": "with_reduction",
            "env": {"TOKEN_REDUCTION": "1", "USER_PROFILE_MODE": "off"},
        },
        {
            "name": "with_user_profile_core",
            "env": {"TOKEN_REDUCTION": "1", "USER_PROFILE_MODE": "core"},
        },
        {
            "name": "with_user_profile_full",
            "env": {"TOKEN_REDUCTION": "1", "USER_PROFILE_MODE": "full"},
        },
    ]
    
    for config in configs:
        name = config["name"]
        output_dir = f"{base_output}/{name}"
        
        log(f"Running ablation: {name}")
        
        env = {
            "OPENAI_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            "PYTHONUNBUFFERED": "1",
            "HF_DATASETS_CACHE": str(REPO_ROOT / ".cache/hf"),
            **config["env"],
        }
        
        cmd = [
            "python", "agents/tool_agents.py",
            "--set_type", set_type,
            "--model_name", "deepseek:deepseek-chat",
            "--start_idx", str(start_idx),
            "--max_items", str(max_items),
            "--output_dir", output_dir,
        ]
        
        success = run_command(cmd, env)
        results[name] = {"success": success, "output_dir": output_dir}
    
    # Save ablation results
    results_file = REPO_ROOT / base_output / "ablation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)
    
    log(f"Ablation results saved to {results_file}")
    return all(r["success"] for r in results.values())


def run_evaluation(eval_file: Path, set_type: str = "validation") -> dict:
    """Run evaluation on a results file."""
    log(f"Running evaluation on {eval_file}")
    
    cmd = [
        "python", "evaluation/eval.py",
        "--set_type", set_type,
        "--evaluation_file_path", str(eval_file),
    ]
    
    env = {
        "PYTHONPATH": str(REPO_ROOT),
    }
    
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    
    # Parse metrics from output
    metrics = {}
    for line in result.stdout.split('\n'):
        for key in ['Delivery Rate', 'Final Pass Rate', 
                   'Commonsense Constraint Micro Pass Rate',
                   'Hard Constraint Micro Pass Rate']:
            if key in line and ':' in line:
                try:
                    value = float(line.split(':')[1].strip().rstrip('%'))
                    metrics[key] = value
                except:
                    pass
    
    return metrics


def generate_all_figures():
    """Generate all paper figures."""
    log("=== Generating Paper Figures ===")
    
    cmd = [
        "python", "tools/generate_paper_figures.py",
        "--experiment_dir", "evaluation/smt_token_output/section_4_2_smt_smoke_policy",
    ]
    
    return run_command(cmd)


def generate_token_report():
    """Generate token analysis report."""
    log("=== Generating Token Analysis Report ===")
    
    cmd = [
        "python", "tools/analyze_token_stats.py",
        "--report", "evaluation/smt_token_output/token_analysis_report.md",
        "--output", "evaluation/smt_token_output/token_analysis.json",
    ]
    
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument(
        "--experiment",
        choices=["all", "two_stage", "smt_only", "hybrid", "ablation", "eval", "figures"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=180)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    log(f"Starting experiment: {args.experiment}")
    log(f"Set type: {args.set_type}, Start: {args.start_idx}, Max items: {args.max_items}")
    
    results = {}
    
    if args.experiment in ["all", "two_stage"]:
        results["two_stage"] = run_two_stage_baseline(
            args.set_type, "deepseek:deepseek-chat",
            args.start_idx, args.max_items,
            args.output_dir or "evaluation/smt_token_output/two_stage_baseline",
        )
    
    if args.experiment in ["all", "smt_only"]:
        results["smt_only"] = run_smt_only(
            args.set_type, args.start_idx, args.max_items,
            args.output_dir or "smt_token_output/smt_only",
        )
    
    if args.experiment in ["all", "hybrid"]:
        results["hybrid"] = run_hybrid_smt(
            args.set_type, args.start_idx, args.max_items,
            args.output_dir or "evaluation/smt_token_output/hybrid_smt",
        )
    
    if args.experiment in ["all", "ablation"]:
        results["ablation"] = run_token_optimization_ablation(
            args.set_type, args.start_idx, min(args.max_items, 30),
        )
    
    if args.experiment in ["all", "figures"]:
        results["figures"] = generate_all_figures()
        results["token_report"] = generate_token_report()
    
    # Summary
    log("\n=== Experiment Summary ===")
    for name, success in results.items():
        status = "✅" if success else "❌"
        log(f"  {name}: {status}")
    
    if all(results.values()):
        log("\nAll experiments completed successfully!")
        return 0
    else:
        log("\nSome experiments failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
