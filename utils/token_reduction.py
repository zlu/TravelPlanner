import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pandas import DataFrame

DEFAULT_MAX_ROWS = 30
DEFAULT_PREVIEW_ROWS = 3

TOOL_COLUMN_ALLOWLIST: Dict[str, List[str]] = {
    "flights": [
        "Flight Number",
        "Price",
        "DepTime",
        "ArrTime",
        "FlightDate",
        "OriginCityName",
        "DestCityName",
    ],
    "accommodations": [
        "NAME",
        "price",
        "room type",
        "house_rules",
        "minimum nights",
        "maximum occupancy",
        "city",
    ],
    "restaurants": [
        "Name",
        "Average Cost",
        "Cuisines",
        "Aggregate Rating",
        "City",
    ],
    "attractions": [
        "Name",
        "Address",
        "City",
    ],
}

TOOL_MAX_ROWS: Dict[str, int] = {
    "flights": 40,
    "accommodations": 30,
    "restaurants": 30,
    "attractions": 30,
}


def _apply_allowlist_override() -> None:
    override = None
    raw = os.getenv("TOOL_COLUMN_ALLOWLIST")
    path = os.getenv("TOOL_COLUMN_ALLOWLIST_PATH")
    if raw:
        try:
            override = json.loads(raw)
        except json.JSONDecodeError:
            override = None
    elif path:
        try:
            override = json.loads(Path(path).read_text())
        except Exception:
            override = None
    if not isinstance(override, dict):
        return
    mode = os.getenv("TOOL_COLUMN_ALLOWLIST_MODE", "merge").lower()
    if mode == "replace":
        TOOL_COLUMN_ALLOWLIST.clear()
    for tool, columns in override.items():
        if isinstance(columns, list):
            TOOL_COLUMN_ALLOWLIST[tool] = columns


_apply_allowlist_override()


def compress_tool_output(tool_name: str, data: Any, max_rows: Optional[int] = None) -> Any:
    if isinstance(data, DataFrame):
        allowlist = TOOL_COLUMN_ALLOWLIST.get(tool_name, [])
        if allowlist:
            columns = [col for col in allowlist if col in data.columns]
            if columns:
                data = data[columns]
        limit = TOOL_MAX_ROWS.get(tool_name, DEFAULT_MAX_ROWS) if max_rows is None else max_rows
        if limit is not None:
            data = data.head(limit)
        return data.to_dict(orient="records")
    return data


def summarize_tool_output(tool_name: str, data: Any, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> str:
    label = tool_name or "result"
    if isinstance(data, list):
        preview = data[:preview_rows]
        return f"{label} results: {len(data)} rows. Preview: {preview}"
    if isinstance(data, dict):
        keys = list(data.keys())
        return f"{label} result: keys={keys}"
    return f"{label} result: {data}"
