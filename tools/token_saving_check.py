import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for lightweight environments
    tiktoken = None

import pandas as pd

from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.attractions.apis import Attractions
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from utils.token_reduction import compress_tool_output


def _get_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        return None


def count_tokens(text: str, encoder) -> int:
    if encoder is None:
        return len(text.encode("utf-8"))
    return len(encoder.encode(text))


def serialize_raw(data):
    if isinstance(data, pd.DataFrame):
        return data.to_string(index=False)
    return str(data)


def serialize_compact(data) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def run_case(origin, destination, flight_date, city, distance_origin, distance_destination):
    encoder = _get_encoder()

    original_cwd = Path.cwd()
    os.chdir(str(REPO_ROOT / "tools"))

    flights = Flights()
    accommodations = Accommodations()
    restaurants = Restaurants()
    attractions = Attractions()
    distance = GoogleDistanceMatrix()

    tools = [
        ("flights", flights.run(origin, destination, flight_date)),
        ("accommodations", accommodations.run(city)),
        ("restaurants", restaurants.run(city)),
        ("attractions", attractions.run(city)),
        ("googleDistanceMatrix", distance.run(distance_origin, distance_destination, "driving")),
    ]

    tool_reports = []
    raw_entries = []
    compact_entries = []

    for tool_name, raw_data in tools:
        compressed = compress_tool_output(tool_name, raw_data)
        raw_text = serialize_raw(raw_data)
        compact_text = serialize_compact(compressed)

        tool_reports.append(
            {
                "tool": tool_name,
                "raw_tokens": count_tokens(raw_text, encoder),
                "compressed_tokens": count_tokens(compact_text, encoder),
                "raw_preview": raw_text[:200],
                "compressed_preview": compact_text[:200],
            }
        )

        raw_entries.append(
            {"Short Description": f"{tool_name} sample", "Content": raw_text}
        )
        compact_entries.append(
            {"Short Description": f"{tool_name} sample", "Content": compressed}
        )

    raw_notebook = serialize_compact(raw_entries)
    compact_notebook = serialize_compact(compact_entries)

    summary = {
        "case": {
            "origin": origin,
            "destination": destination,
            "flight_date": flight_date,
            "city": city,
            "distance_origin": distance_origin,
            "distance_destination": distance_destination,
        },
        "notebook_tokens": {
            "raw_tokens": count_tokens(raw_notebook, encoder),
            "compressed_tokens": count_tokens(compact_notebook, encoder),
        },
        "tool_reports": tool_reports,
    }
    os.chdir(original_cwd)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Token-saving smoke test for tool outputs.")
    parser.add_argument("--origin", default="Seattle")
    parser.add_argument("--destination", default="New York")
    parser.add_argument("--flight_date", default="2022-04-01")
    parser.add_argument("--city", default="New York")
    parser.add_argument("--distance_origin", default="Detroit")
    parser.add_argument("--distance_destination", default="Norfolk")
    parser.add_argument("--output_dir", default="smt_token_output")
    args = parser.parse_args()

    report = run_case(
        args.origin,
        args.destination,
        args.flight_date,
        args.city,
        args.distance_origin,
        args.distance_destination,
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "token_saving_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote token saving report to {report_path}")


if __name__ == "__main__":
    main()
