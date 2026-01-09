import json
from typing import Any, Dict


CORE_LOCAL_KEYS = {"house rule", "room type", "transportation"}
SECONDARY_LOCAL_KEYS = {"cuisine"}


def _is_set(value: Any) -> bool:
    return value not in (None, "", "null")


def build_user_profile(item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a layered user profile with core constraints and secondary preferences.
    """
    core = {
        "org": item.get("org"),
        "dest": item.get("dest"),
        "days": item.get("days"),
        "date": item.get("date"),
        "visiting_city_number": item.get("visiting_city_number"),
        "people_number": item.get("people_number"),
        "budget": item.get("budget"),
    }

    local = item.get("local_constraint") or {}
    if not isinstance(local, dict):
        local = {}
    core_constraints = {k: v for k, v in core.items() if _is_set(v)}
    secondary_prefs: Dict[str, Any] = {}

    for key, value in local.items():
        if not _is_set(value):
            continue
        if key in CORE_LOCAL_KEYS:
            core_constraints[key] = value
        elif key in SECONDARY_LOCAL_KEYS:
            secondary_prefs[key] = value
        else:
            secondary_prefs[key] = value

    return {
        "core_constraints": core_constraints,
        "secondary_prefs": secondary_prefs,
    }


def format_profile(profile: Dict[str, Dict[str, Any]], include_secondary: bool) -> str:
    """
    Return a compact JSON string for prompt injection.
    """
    payload = {"core_constraints": profile.get("core_constraints", {})}
    if include_secondary:
        payload["secondary_prefs"] = profile.get("secondary_prefs", {})
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)
