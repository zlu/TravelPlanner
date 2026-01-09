import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.user_profile import build_user_profile, format_profile


def _prune_day_sections(text: str, keep_days: int) -> str:
    if keep_days <= 0:
        return text
    pattern = re.compile(r"(?m)^Day\\s+\\d+\\s*:")
    matches = list(pattern.finditer(text))
    if len(matches) <= keep_days:
        return text

    preamble = text[:matches[0].start()]
    keep_start = matches[-keep_days].start()
    removed = text[matches[0].start():keep_start]
    kept = text[keep_start:]

    summaries = []
    for chunk in re.split(r"(?m)(?=^Day\\s+\\d+\\s*:)", removed):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        header = lines[0]
        current_city = next((line for line in lines if line.startswith("Current City:")), "")
        accommodation = next((line for line in lines if line.startswith("Accommodation:")), "")
        summary_parts = [header]
        if current_city:
            summary_parts.append(current_city)
        if accommodation:
            summary_parts.append(accommodation)
        summaries.append(" | ".join(summary_parts))

    summary_block = ""
    if summaries:
        summary_block = "Earlier days (summary):\\n" + "\\n".join(summaries) + "\\n\\n"

    return preamble + summary_block + kept


def main() -> None:
    query_item = {
        "org": "Cleveland",
        "dest": "Florida",
        "days": 5,
        "visiting_city_number": 2,
        "date": ["2022-03-02", "2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06"],
        "people_number": 6,
        "local_constraint": {
            "house rule": "pets",
            "cuisine": "italian",
            "room type": "entire room",
            "transportation": "no self-driving",
        },
        "budget": 13900,
    }
    query = (
        "Plan a 5-day trip from Cleveland to Florida for 6 people, "
        "budget $13,900, pets allowed, entire room, no self-driving."
    )

    profile = build_user_profile(query_item)
    print("CORE PROFILE JSON:")
    print(format_profile(profile, include_secondary=False))
    print("\\nFULL PROFILE JSON:")
    print(format_profile(profile, include_secondary=True))
    print("\\nQUERY CONTEXT (core mode):")
    print(f"Core constraints (JSON): {format_profile(profile, include_secondary=False)}")
    print("\\nQUERY CONTEXT (full mode):")
    print(
        "Core constraints (JSON): "
        + format_profile(profile, include_secondary=True)
        + "\\nOriginal query: "
        + query
    )

    sample_plan = \"\"\"Travel Plan:
Day 1:
Current City: Cleveland
Transportation: Flight A
Accommodation: Hotel A

Day 2:
Current City: Florida
Transportation: -
Accommodation: Hotel B

Day 3:
Current City: Florida
Transportation: -
Accommodation: Hotel C

Day 4:
Current City: Florida
Transportation: -
Accommodation: Hotel D
\"\"\"
    print(\"\\nPRUNED PLAN (keep last 2 days):\")
    print(_prune_day_sections(sample_plan, keep_days=2))


if __name__ == \"__main__\":
    main()
