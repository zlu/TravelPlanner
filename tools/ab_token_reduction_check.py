import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "agents"))

# Use a dummy tiktoken by default to avoid network fetches for encodings.
if os.getenv("FORCE_DUMMY_TIKTOKEN", "1") != "0":
    class _DummyEncoder:
        def encode(self, text):
            return list(text.encode("utf-8"))
    class _DummyTikToken:
        def encoding_for_model(self, name):
            return _DummyEncoder()
    sys.modules["tiktoken"] = _DummyTikToken()

from agents.tool_agents import ReactAgent  # noqa: E402


def load_queries(path: Path, limit: int):
    data = json.loads(path.read_text())
    queries = [item["query"] for item in data.get("validation", [])]
    return queries[:limit]


def run_variant(queries, output_dir: Path, token_reduction_enabled: bool, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKEN_REDUCTION"] = "1" if token_reduction_enabled else "0"

    tools_list = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    agent = ReactAgent(
        None,
        tools=tools_list,
        max_steps=30,
        react_llm_name=model_name,
        planner_llm_name=model_name,
    )

    results = []
    for idx, query in enumerate(queries, start=1):
        plan, scratchpad, action_log = agent.run(query)
        plan_text = plan or ""

        entry_dir = output_dir / str(idx)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / "query.txt").write_text(query)
        (entry_dir / "plan.txt").write_text(plan_text)
        (entry_dir / "scratchpad.txt").write_text(scratchpad or "")
        (entry_dir / "action_log.json").write_text(json.dumps(action_log, indent=2))

        results.append(
            {
                "index": idx,
                "plan_len": len(plan_text),
                "scratchpad_len": len(scratchpad or ""),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test token reduction on first N queries.")
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "gpt-3.5-turbo-1106"))
    parser.add_argument("--output_dir", default="smt_token_output/ab_token_reduction")
    args = parser.parse_args()

    queries_path = REPO_ROOT / "validation_queries.json"
    queries = load_queries(queries_path, args.limit)

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = REPO_ROOT / output_base

    baseline_dir = output_base / "baseline"
    reduced_dir = output_base / "reduced"

    baseline = run_variant(queries, baseline_dir, False, args.model_name)
    reduced = run_variant(queries, reduced_dir, True, args.model_name)

    diffs = []
    for idx in range(1, len(queries) + 1):
        base_plan = (baseline_dir / str(idx) / "plan.txt").read_text()
        reduced_plan = (reduced_dir / str(idx) / "plan.txt").read_text()
        diffs.append(
            {
                "index": idx,
                "same_plan": base_plan == reduced_plan,
                "baseline_plan_len": len(base_plan),
                "reduced_plan_len": len(reduced_plan),
            }
        )

    summary = {
        "model_name": args.model_name,
        "limit": args.limit,
        "baseline": baseline,
        "reduced": reduced,
        "diffs": diffs,
    }
    summary_path = output_base / "ab_compare.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote comparison to {summary_path}")


if __name__ == "__main__":
    main()
