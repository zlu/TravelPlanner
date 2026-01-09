import json
import os
from typing import Any, Dict, List


DEFAULT_COMMONSENSE = [
    "valid travel order",
    "no duplicated attractions",
    "reasonable transit time",
    "meal diversity",
]

DEFAULT_ENVIRONMENT = [
    "tool availability (flights/hotels/restaurants)",
]

DEFAULT_RELAX_ORDER = [
    "cuisine",
    "room type",
    "house rule",
    "transportation",
]


def _is_set(value: Any) -> bool:
    return value not in (None, "", "null")


def _normalize_local(local: Any) -> Dict[str, Any]:
    if isinstance(local, dict):
        return local
    return {}


def _get_relax_order() -> List[str]:
    raw = os.getenv("CONSTRAINT_RELAX_ORDER")
    if not raw:
        return DEFAULT_RELAX_ORDER
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_constraint_policy(query_item: Dict[str, Any]) -> Dict[str, Any]:
    local = _normalize_local(query_item.get("local_constraint"))
    hard_constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    conflicts: List[str] = []

    for key in ("org", "dest", "days", "date", "people_number", "budget"):
        value = query_item.get(key)
        if _is_set(value):
            hard_constraints[key] = value

    for key in ("transportation", "room type", "house rule"):
        value = local.get(key)
        if _is_set(value):
            hard_constraints[key] = value

    cuisine = local.get("cuisine")
    if _is_set(cuisine):
        preferences["cuisine"] = cuisine

    if os.getenv("CONSTRAINT_BUDGET_CHECK", "0") == "1":
        try:
            from utils.budget_estimation import budget_calc

            dates = query_item.get("date") or []
            if isinstance(dates, str):
                dates = [dates]
            budgets = budget_calc(
                query_item.get("org"),
                query_item.get("dest"),
                int(query_item.get("days", 0)),
                dates,
                people_number=query_item.get("people_number"),
                local_constraint=local,
            )
            min_budget = budgets.get("lowest")
            budget = query_item.get("budget")
            if _is_set(budget) and min_budget is not None and budget < min_budget:
                conflicts.append(f"budget<{int(min_budget)}")
        except Exception as exc:
            conflicts.append(f"budget_check_failed:{exc}")

    policy = {
        "priority_order": ["hard", "commonsense", "environment", "preferences"],
        "hard_constraints": hard_constraints,
        "commonsense_rules": DEFAULT_COMMONSENSE,
        "environment_rules": DEFAULT_ENVIRONMENT,
        "preferences": preferences,
        "relax_order": _get_relax_order(),
        "conflicts": conflicts,
    }
    return policy


def format_constraint_policy(policy: Dict[str, Any]) -> str:
    return json.dumps(policy, ensure_ascii=True, sort_keys=True)
