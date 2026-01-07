"""
Aggregate SMT planner outputs into a single JSONL for evaluation.

It scans ../LLM_Formal_Travel_Planner/output/{set_type}/gpt_nl/*/plans/plan.txt
and emits a JSONL with {"idx": <int>, "query": <str>, "plan": <list-of-day-dicts>}.
Parsing follows the upstream convert_json.py logic.
"""
import argparse
import json
import ast
from pathlib import Path


def parse_plan(plan_text: str, query_json: dict):
    """
    Parse plan.txt into per-day plan list as expected by eval.py.
    """
    lines = [ln.strip() for ln in plan_text.strip().split("\n") if ln.strip()]
    if len(lines) < 6:
        return []
    def parse_list(line, prefix):
        # line like "Destination cities: ['Myrtle Beach'],"
        try:
            content = line.split(":", 1)[1].strip().rstrip(",")
            return ast.literal_eval(content)
        except Exception:
            return []
    destinations = parse_list(lines[0], "Destination cities")
    dates = parse_list(lines[1], "Transportation dates")
    transportation = parse_list(lines[2], "Transportation methods between cities")
    restaurants = parse_list(lines[3], "Restaurants")
    attractions = parse_list(lines[4], "Attractions")
    accommodations = parse_list(lines[5], "Accommodations")

    # Build destinations list with origin at start/end.
    destinations = [query_json["org"]] + destinations + [query_json["org"]]
    accommodations.append("-")  # for the return leg

    plan_json = [{}, {}, {}, {}, {}, {}, {}]
    city_index = 0
    days = query_json.get("days", 0)
    for i in range(days):
        plan_json[i]["day"] = i + 1
        # current city
        if city_index < len(dates) and i < len(query_json["date"]) and dates[city_index] == query_json["date"][i]:
            plan_json[i]["current_city"] = f"from {destinations[city_index]} to {destinations[city_index+1]}"
            plan_json[i]["transportation"] = transportation[city_index] if city_index < len(transportation) else "-"
            city_index += 1
        else:
            plan_json[i]["current_city"] = destinations[city_index]
            plan_json[i]["transportation"] = "-"
        # meals
        base = 3 * i
        plan_json[i]["breakfast"] = restaurants[base] if base < len(restaurants) else "-"
        plan_json[i]["lunch"] = restaurants[base + 1] if base + 1 < len(restaurants) else "-"
        plan_json[i]["dinner"] = restaurants[base + 2] if base + 2 < len(restaurants) else "-"
        # attraction
        plan_json[i]["attraction"] = attractions[i] if i < len(attractions) else "-"
        # accommodation
        acc_idx = max(city_index - 1, 0)
        plan_json[i]["accommodation"] = accommodations[acc_idx] if acc_idx < len(accommodations) else "-"
    return plan_json[:days]


def aggregate(set_type: str, output_path: Path):
    base = Path(__file__).resolve().parents[1] / "smt_output" / set_type / "gpt_nl"
    records = []
    if not base.exists():
        raise FileNotFoundError(f"SMT output dir not found: {base}")

    def dummy_plan():
        return [{"day": i+1,
                 "current_city": "-",
                 "transportation": "-",
                 "breakfast": "-",
                 "lunch": "-",
                 "dinner": "-",
                 "attraction": "-",
                 "accommodation": "-"} for i in range(7)]

    # Determine expected count per split (align with upstream scripts).
    expected_by_split = {"train": 45, "validation": 180, "test": 1000}
    target_count = expected_by_split.get(set_type, 0)

    # Iterate over all possible indices based on directories present.
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    if not dirs:
        raise FileNotFoundError(f"No plan directories found under {base}")
    max_idx = max(int(d.name) for d in dirs if d.name.isdigit())
    if target_count < max_idx:
        target_count = max_idx
    if target_count == 0:
        target_count = max_idx

    for i in range(1, target_count + 1):
        plan_dir = base / str(i)
        plan_file = plan_dir / "plans" / "plan.txt"
        query_file = plan_dir / "plans" / "query.json"
        query_txt_file = plan_dir / "plans" / "query.txt"

        if plan_file.exists() and query_file.exists():
            plan_text = plan_file.read_text().strip()
            query_json = json.loads(query_file.read_text())
            plan_struct = parse_plan(plan_text, query_json)
            raw_query = query_txt_file.read_text().strip() if query_txt_file.exists() else query_json.get("query", "")
        else:
            # Pad missing entries with dummy plan and best-effort query text.
            raw_query = ""
            if query_txt_file.exists():
                raw_query = query_txt_file.read_text().strip()
            elif query_file.exists():
                try:
                    raw_query = json.loads(query_file.read_text()).get("query", "")
                except Exception:
                    raw_query = ""
            else:
                # Try to pull from a parallel dataset if available
                try:
                    import json as _json
                    with open(Path(__file__).resolve().parents[1] / "validation_queries.json") as f:
                        qdata = _json.load(f).get("validation", [])
                        if i - 1 < len(qdata):
                            raw_query = qdata[i-1].get("query", "")
                except Exception:
                    raw_query = ""
            plan_struct = dummy_plan()

        records.append({"idx": i, "query": raw_query, "plan": plan_struct})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate SMT plan outputs to JSONL.")
    parser.add_argument("--set_type", choices=["train", "validation", "test"], default="validation")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("smt_aggregated.jsonl"),
        help="Where to write the aggregated JSONL.",
    )
    args = parser.parse_args()
    aggregate(args.set_type, args.output_path)


if __name__ == "__main__":
    main()
