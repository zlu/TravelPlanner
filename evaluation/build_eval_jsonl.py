import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def _read_json_safely(path: Path):
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw)
        return obj


def _build_plan_field(model_name: str, mode: str, strategy: str) -> str:
    if mode == "two-stage":
        return f"{model_name}_{mode}_parsed_results"
    suffix = f"_{strategy}" if strategy else ""
    return f"{model_name}{suffix}_{mode}_parsed_results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval jsonl from generated_plan files.")
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--plan_field", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if args.total:
        total = args.total
    else:
        download_mode = os.getenv("DATASET_DOWNLOAD_MODE", "reuse_cache_if_exists")
        total = None
        try:
            dataset = load_dataset(
                "osunlp/TravelPlanner",
                args.set_type,
                download_mode=download_mode,
            )[args.set_type]
            total = len(dataset)
        except Exception:
            input_root = Path(args.input_dir) / args.set_type
            indices = []
            for path in input_root.glob("generated_plan_*.json"):
                try:
                    idx = int(path.stem.split("_")[-1])
                    indices.append(idx)
                except ValueError:
                    continue
            if indices:
                total = max(indices)
        if total is None:
            local_path = Path("validation_queries.json")
            if local_path.exists():
                try:
                    payload = json.loads(local_path.read_text())
                    if isinstance(payload, dict):
                        items = payload.get(args.set_type) or payload.get("validation") or []
                    elif isinstance(payload, list):
                        items = payload
                    else:
                        items = []
                    if isinstance(items, list) and items:
                        total = len(items)
                except Exception:
                    pass
    if total is None:
        raise RuntimeError("Unable to determine dataset size or available plan files.")
    end_idx = total
    if args.max_items:
        end_idx = min(total, args.start_idx - 1 + args.max_items)

    plan_field = args.plan_field or _build_plan_field(
        args.model_name, args.mode, args.strategy
    )

    input_root = Path(args.input_dir) / args.set_type
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(args.start_idx, end_idx + 1):
            plan = None
            plan_path = input_root / f"generated_plan_{idx}.json"
            if plan_path.exists():
                payload = _read_json_safely(plan_path)
                if isinstance(payload, list) and payload:
                    plan = payload[-1].get(plan_field)
            line = {"idx": idx, "plan": plan}
            f.write(json.dumps(line, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
