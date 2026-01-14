#!/usr/bin/env python3
"""
Aggregate SMT experiment outputs into evaluation JSONL.

This script handles the specific experiment directory structure:
  evaluation/smt_token_output/{experiment}/smt_planner/output/{set_type}/gpt_nl/{idx}/plans/plan.txt
"""
import argparse
import ast
import json
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset


def parse_plan_txt(plan_text: str, query_json: dict) -> List[dict]:
    """Parse SMT plan.txt into per-day plan list matching eval.py format."""
    dates_field = query_json.get("date", [])
    if isinstance(dates_field, str):
        try:
            dates_field = ast.literal_eval(dates_field)
        except Exception:
            dates_field = [dates_field]
    query_json = dict(query_json)
    query_json["date"] = dates_field

    lines = [ln.strip() for ln in plan_text.strip().split("\n") if ln.strip()]
    if len(lines) < 6:
        return None

    def parse_list(line: str) -> list:
        try:
            content = line.split(":", 1)[1].strip().rstrip(",")
            return ast.literal_eval(content)
        except Exception:
            return []

    destinations = parse_list(lines[0])
    dates = parse_list(lines[1])
    transportation = parse_list(lines[2])
    restaurants = parse_list(lines[3])
    attractions = parse_list(lines[4])
    accommodations = parse_list(lines[5])

    org = query_json.get("org", "-")
    dest_chain = [org] + destinations + [org]
    accommodations.append("-")

    days = int(query_json.get("days", 3))
    plan_json = []
    city_index = 0

    for i in range(days):
        day_plan = {"days": i + 1}

        # Determine current city and transportation
        if city_index < len(dates) and i < len(query_json["date"]) and dates[city_index] == query_json["date"][i]:
            day_plan["current_city"] = f"from {dest_chain[city_index]} to {dest_chain[city_index + 1]}"
            day_plan["transportation"] = transportation[city_index] if city_index < len(transportation) else "-"
            city_index += 1
        else:
            day_plan["current_city"] = dest_chain[city_index] if city_index < len(dest_chain) else "-"
            day_plan["transportation"] = "-"

        # Meals (3 per day)
        base = 3 * i
        day_plan["breakfast"] = restaurants[base] if base < len(restaurants) else "-"
        day_plan["lunch"] = restaurants[base + 1] if base + 1 < len(restaurants) else "-"
        day_plan["dinner"] = restaurants[base + 2] if base + 2 < len(restaurants) else "-"

        # Attraction
        day_plan["attraction"] = attractions[i] if i < len(attractions) else "-"

        # Accommodation
        acc_idx = max(city_index - 1, 0)
        day_plan["accommodation"] = accommodations[acc_idx] if acc_idx < len(accommodations) else "-"

        plan_json.append(day_plan)

    return plan_json


def load_query_data(set_type: str) -> List[dict]:
    """Load query data from HuggingFace or local cache."""
    try:
        dataset = load_dataset(
            "osunlp/TravelPlanner",
            set_type,
            download_mode="reuse_cache_if_exists",
        )[set_type]
        return list(dataset)
    except Exception as e:
        print(f"Warning: Could not load HF dataset: {e}")
        # Try local queries file
        local_path = Path("validation_queries.json")
        if local_path.exists():
            data = json.loads(local_path.read_text())
            if isinstance(data, dict):
                return data.get(set_type, data.get("validation", []))
            return data if isinstance(data, list) else []
        return []


def aggregate_experiment(
    experiment_dir: Path,
    set_type: str,
    output_path: Path,
    total: Optional[int] = None,
) -> dict:
    """
    Aggregate SMT experiment outputs into evaluation JSONL.
    
    Returns statistics dict with sat/unsat/missing counts.
    """
    smt_base = experiment_dir / "smt_planner" / "output" / set_type / "gpt_nl"
    
    # Load query data for context
    query_data = load_query_data(set_type)
    
    # Determine total count
    if total is None:
        if set_type == "validation":
            total = 180
        elif set_type == "train":
            total = 45
        elif set_type == "test":
            total = 1000
        else:
            total = len(query_data) if query_data else 180
    
    stats = {"sat": 0, "unsat": 0, "error": 0, "missing": 0, "total": total}
    records = []
    
    for idx in range(1, total + 1):
        plan_dir = smt_base / str(idx) / "plans"
        plan_file = plan_dir / "plan.txt"
        query_file = plan_dir / "query.json"
        status_file = plan_dir / "status.json"
        
        plan = None
        query_text = ""
        
        # Get query text
        if idx <= len(query_data):
            query_text = query_data[idx - 1].get("query", "")
        
        if plan_file.exists() and query_file.exists():
            try:
                plan_text = plan_file.read_text().strip()
                query_json = json.loads(query_file.read_text())
                plan = parse_plan_txt(plan_text, query_json)
                if plan:
                    stats["sat"] += 1
                else:
                    stats["error"] += 1
            except Exception as e:
                print(f"Error parsing plan {idx}: {e}")
                stats["error"] += 1
        elif status_file.exists():
            try:
                status = json.loads(status_file.read_text())
                if status.get("status") == "unsat":
                    stats["unsat"] += 1
                else:
                    stats["error"] += 1
            except Exception:
                stats["missing"] += 1
        else:
            stats["missing"] += 1
        
        records.append({"idx": idx, "query": query_text, "plan": plan})
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Stats: SAT={stats['sat']}, UNSAT={stats['unsat']}, ERROR={stats['error']}, MISSING={stats['missing']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate SMT experiment outputs.")
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("evaluation/smt_token_output/section_4_2_smt_smoke_policy"),
        help="Path to experiment directory.",
    )
    parser.add_argument("--set_type", choices=["train", "validation", "test"], default="validation")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to {experiment_dir}/smt_eval.jsonl",
    )
    parser.add_argument("--total", type=int, default=None, help="Total number of queries to process.")
    args = parser.parse_args()
    
    output_path = args.output_path or (args.experiment_dir / "smt_eval.jsonl")
    
    stats = aggregate_experiment(
        args.experiment_dir,
        args.set_type,
        output_path,
        args.total,
    )
    
    # Also save stats
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
