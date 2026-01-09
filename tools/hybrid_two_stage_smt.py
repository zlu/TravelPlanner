import argparse
import builtins
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from agents.smt_runner import _load_formal_module, _maybe_patch_deepseek, _patch_data_paths, _shim_optional_dependencies
from utils.user_profile import build_user_profile, format_profile


_ENCODING = None
_RUNTIME_CACHE: Dict[str, Dict[str, object]] = {}


def _resolve_smt_repo(smt_repo: Path | None) -> Path:
    if smt_repo:
        return Path(smt_repo)
    env_repo = os.getenv("SMT_REPO")
    if env_repo:
        return Path(env_repo)
    candidate = REPO_ROOT.parent / "LLM_Formal_Travel_Planner"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "Cannot locate LLM_Formal_Travel_Planner. Set SMT_REPO or pass smt_repo."
    )


def _get_runtime(repo_root: Path) -> Dict[str, object]:
    key = str(repo_root.resolve())
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]

    _shim_optional_dependencies()
    module = _load_formal_module(repo_root)
    _maybe_patch_deepseek(module)
    _patch_generate_as_plan(module)
    _patch_steps_response(module)

    data_root = REPO_ROOT / "database"
    flights_df = pd.read_csv(data_root / "flights/clean_Flights_2022.csv").dropna()
    accommodations_df = pd.read_csv(
        data_root / "accommodations/clean_accommodations_2022.csv"
    ).dropna()
    restaurants_df = pd.read_csv(
        data_root / "restaurants/clean_restaurant_2022.csv"
    ).dropna()
    attractions_df = pd.read_csv(
        data_root / "attractions/attractions.csv"
    ).dropna()
    distance_df = pd.read_csv(
        data_root / "googleDistanceMatrix/distance.csv"
    ).dropna()
    state_city_map = load_state_city_map(
        data_root / "background/citySet_with_states.txt"
    )

    runtime = {
        "module": module,
        "data_root": data_root,
        "flights_df": flights_df,
        "accommodations_df": accommodations_df,
        "restaurants_df": restaurants_df,
        "attractions_df": attractions_df,
        "distance_df": distance_df,
        "state_city_map": state_city_map,
    }
    _RUNTIME_CACHE[key] = runtime
    return runtime


def _count_tokens(text: str) -> Tuple[int, str]:
    global _ENCODING
    if tiktoken is None:
        return len(text.split()), "whitespace"
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODING.encode(text)), "tiktoken:cl100k_base"


def _build_prompt_query(query: str, query_json: dict | None) -> str:
    mode = os.getenv("USER_PROFILE_MODE", "off").lower()
    if mode not in {"core", "full"}:
        return query
    if not query_json:
        return query
    profile = build_user_profile(query_json)
    include_secondary = mode == "full"
    profile_text = format_profile(profile, include_secondary)
    if mode == "core":
        return f"Core constraints (JSON): {profile_text}"
    return f"Core constraints (JSON): {profile_text}\nOriginal query: {query}"


def load_state_city_map(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        city, state = line.split("\t")
        mapping.setdefault(state, []).append(city)
    return mapping


def rank_cities_by_flights(
    flights_df: pd.DataFrame, origin: str, date: str, cities: List[str]
) -> List[str]:
    counts = []
    for city in cities:
        mask = (
            (flights_df["OriginCityName"] == origin)
            & (flights_df["DestCityName"] == city)
            & (flights_df["FlightDate"] == date)
        )
        counts.append((city, int(mask.sum())))
    counts.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in counts]


def safe_filter(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    filtered = df[mask]
    return filtered if not filtered.empty else df


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_city_sets(
    data_root: Path, selected_cities: List[str], dest_state: str, origin: str
) -> None:
    background = data_root / "background"
    background.mkdir(parents=True, exist_ok=True)
    city_set = sorted(set(selected_cities + [origin]))
    (background / "citySet.txt").write_text("\n".join(city_set))
    lines = [f"{city}\t{dest_state}" for city in selected_cities]
    (background / "citySet_with_states.txt").write_text("\n".join(lines))


@contextmanager
def _patch_prompt_open(prompt_root: Path):
    original_open = builtins.open

    def open_with_prompts(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and "r" in mode:
            path = Path(file)
            if not path.is_absolute() and path.as_posix().startswith("prompts/"):
                file = prompt_root / path
        return original_open(file, mode, *args, **kwargs)

    builtins.open = open_with_prompts
    try:
        yield
    finally:
        builtins.open = original_open


def _patch_generate_as_plan(module):
    np = module.np

    def safe_seq_value(values, idx, default="-"):
        if idx is None:
            return default
        try:
            idx_int = int(idx)
        except Exception:
            return default
        try:
            if idx_int < 0 or idx_int >= len(values):
                return default
        except Exception:
            return default
        try:
            return values[idx_int]
        except Exception:
            return default

    def safe_col_value(data, column, idx, default="-"):
        if isinstance(data, str):
            return default
        try:
            values = np.array(data[column])
        except Exception:
            return default
        return safe_seq_value(values, idx, default)

    def safe_generate_as_plan(s, variables, query):
        CitySearch = module.Cities()
        FlightSearch = module.Flights()
        AttractionSearch = module.Attractions()
        DistanceSearch = module.GoogleDistanceMatrix()
        AccommodationSearch = module.Accommodations()
        RestaurantSearch = module.Restaurants()

        cities = []
        transportation = []
        departure_dates = []
        transportation_info = []
        restaurant_city_list = []
        attraction_city_list = []
        accommodation_city_list = []

        dates = query.get("date", [])
        if isinstance(dates, str):
            dates = [dates]

        if query["visiting_city_number"] == 1:
            cities = [query["dest"]]
            cities_list = [query["dest"]]
        else:
            cities_list = CitySearch.run(query["dest"], query["org"], dates)
            if isinstance(cities_list, str):
                cities_list = [query["dest"]]
            elif query["org"] in cities_list:
                cities_list.remove(query["org"])
            for city in variables["city"]:
                city_index = s.model()[city].as_long()
                cities.append(safe_seq_value(cities_list, city_index, query["dest"]))

        for i, flight in enumerate(variables["flight"]):
            if bool(s.model()[flight]):
                transportation.append("flight")
            elif bool(s.model()[variables["self-driving"][i]]):
                transportation.append("self-driving")
            else:
                transportation.append("taxi")

        for date_index in variables["departure_dates"]:
            date_idx = s.model()[date_index].as_long()
            departure_dates.append(safe_seq_value(dates, date_idx, dates[0] if dates else "-"))

        dest_cities = [query["org"]] + cities + [query["org"]]
        for i, index in enumerate(variables["flight_index"]):
            if transportation[i] == "flight":
                flight_index = s.model()[index].as_long()
                flight_list = FlightSearch.run(dest_cities[i], dest_cities[i + 1], departure_dates[i])
                flight_number = safe_col_value(flight_list, "Flight Number", flight_index)
                origin_city = safe_col_value(flight_list, "OriginCityName", flight_index)
                dest_city = safe_col_value(flight_list, "DestCityName", flight_index)
                dep_time = safe_col_value(flight_list, "DepTime", flight_index)
                arr_time = safe_col_value(flight_list, "ArrTime", flight_index)
                if flight_number == "-":
                    transportation_info.append("-")
                else:
                    flight_info = (
                        "Flight Number: {}, from {} to {}, Departure Time: {}, Arrival Time: {}"
                    ).format(flight_number, origin_city, dest_city, dep_time, arr_time)
                    transportation_info.append(flight_info)
            elif transportation[i] == "self-driving":
                transportation_info.append(
                    "Self-" + DistanceSearch.run(dest_cities[i], dest_cities[i + 1], mode="driving")
                )
            else:
                transportation_info.append(
                    DistanceSearch.run(dest_cities[i], dest_cities[i + 1], mode="taxi")
                )

        for i, which_city in enumerate(variables["restaurant_in_which_city"]):
            city_index = s.model()[which_city].as_long()
            if int(city_index) == -1:
                restaurant_city_list.append("-")
            else:
                city = safe_seq_value(cities_list, city_index, "-")
                restaurant_list = RestaurantSearch.run(city)
                restaurant_index = s.model()[variables["restaurant_index"][i]].as_long()
                restaurant = safe_col_value(restaurant_list, "Name", restaurant_index)
                if restaurant == "-" or city == "-":
                    restaurant_city_list.append("-")
                else:
                    restaurant_city_list.append(restaurant + ", " + city)

        for i, which_city in enumerate(variables["attraction_in_which_city"]):
            city_index = s.model()[which_city].as_long()
            if int(city_index) == -1:
                attraction_city_list.append("-")
            else:
                city = safe_seq_value(cities_list, city_index, "-")
                attraction_list = AttractionSearch.run(city)
                attraction_index = s.model()[variables["attraction_index"][i]].as_long()
                attraction = safe_col_value(attraction_list, "Name", attraction_index)
                if attraction == "-" or city == "-":
                    attraction_city_list.append("-")
                else:
                    attraction_city_list.append(attraction + ", " + city)

        for i, city in enumerate(cities):
            accommodation_list = AccommodationSearch.run(city)
            accommodation_index = s.model()[variables["accommodation_index"][i]].as_long()
            accommodation = safe_col_value(accommodation_list, "NAME", accommodation_index)
            if accommodation == "-":
                accommodation_city_list.append("-")
            else:
                accommodation_city_list.append(accommodation + ", " + city)

        return (
            f"Destination cities: {cities},\n"
            f"Transportation dates: {departure_dates},\n"
            f"Transportation methods between cities: {transportation_info},\n"
            f"Restaurants (3 meals per day): {restaurant_city_list},\n"
            f"Attractions (1 per day): {attraction_city_list},\n"
            f"Accommodations (1 per city): {accommodation_city_list}"
        )

    module.generate_as_plan = safe_generate_as_plan


def _patch_steps_response(module):
    base_response = module.GPT_response

    def normalize_steps(text: str) -> str:
        blocks = text.split("\n\n")
        cleaned = []
        for block in blocks:
            lines = [line.rstrip() for line in block.splitlines()]
            if not lines:
                continue
            header = lines[0].rstrip()
            rest = "\n".join(lines[1:])
            cleaned.append(header + ("\n" + rest if rest else ""))
        return "\n\n".join(cleaned)

    def wrapped(prompt, *args, **kwargs):
        text = base_response(prompt, *args, **kwargs)
        if isinstance(prompt, str) and "Steps:" in prompt and isinstance(text, str):
            return normalize_steps(text)
        return text

    module.GPT_response = wrapped
    if hasattr(module, "openai_func"):
        module.openai_func.GPT_response = wrapped


def build_filtered_database(
    data_root: Path,
    query_json: dict,
    flights_df: pd.DataFrame,
    accommodations_df: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    attractions_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    state_city_map: Dict[str, List[str]],
    top_k_cities: int,
) -> Dict[str, int]:
    org = query_json["org"]
    dest = query_json["dest"]
    days = int(query_json["days"])
    visit_n = int(query_json.get("visiting_city_number", 1))
    dates = query_json.get("date")
    if isinstance(dates, str):
        dates = [dates]
    elif not isinstance(dates, (list, tuple)):
        dates = [str(dates)]

    if visit_n > 1 and dest in state_city_map:
        candidate_cities = state_city_map[dest]
        ranked = rank_cities_by_flights(flights_df, org, dates[0], candidate_cities)
        selected = ranked[: max(top_k_cities, visit_n)]
        dest_state = dest
    else:
        selected = [dest]
        dest_state = dest

    cities_all = sorted(set(selected + [org]))

    flights_mask = (
        flights_df["OriginCityName"].isin(cities_all)
        & flights_df["DestCityName"].isin(cities_all)
        & flights_df["FlightDate"].isin(dates)
    )
    acc_mask = accommodations_df["city"].isin(selected)
    rest_mask = restaurants_df["City"].isin(selected)
    attr_mask = attractions_df["City"].isin(selected)
    dist_mask = distance_df["origin"].isin(cities_all) & distance_df["destination"].isin(cities_all)

    flights_filtered = safe_filter(flights_df, flights_mask, "flights")
    acc_filtered = safe_filter(accommodations_df, acc_mask, "accommodations")
    rest_filtered = safe_filter(restaurants_df, rest_mask, "restaurants")
    attr_filtered = safe_filter(attractions_df, attr_mask, "attractions")
    dist_filtered = safe_filter(distance_df, dist_mask, "distance")

    write_csv(data_root / "flights/clean_Flights_2022.csv", flights_filtered)
    write_csv(data_root / "accommodations/clean_accommodations_2022.csv", acc_filtered)
    write_csv(data_root / "restaurants/clean_restaurant_2022.csv", rest_filtered)
    write_csv(data_root / "attractions/attractions.csv", attr_filtered)
    write_csv(data_root / "googleDistanceMatrix/distance.csv", dist_filtered)

    write_city_sets(data_root, selected, dest_state, org)

    return {
        "selected_cities": len(selected),
        "flights": len(flights_filtered),
        "accommodations": len(acc_filtered),
        "restaurants": len(rest_filtered),
        "attractions": len(attr_filtered),
        "distance": len(dist_filtered),
    }


def _load_local_dataset(set_type: str) -> List[dict]:
    path = REPO_ROOT / "validation_queries.json"
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        items = data.get(set_type) or data.get("validation") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    if not isinstance(items, list):
        raise ValueError(f"Unsupported dataset format in {path}")
    return items


def _load_cached_query_json(
    output_root: Path, repo_root: Path, set_type: str, index: int
) -> dict | None:
    candidates = [
        output_root / "output",
        repo_root / "output",
    ]
    for base in candidates:
        path = base / set_type / "gpt_nl" / str(index) / "plans" / "query.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def _parse_query_json(module, prompt_root: Path, query: str, model_version: str) -> Tuple[dict, int, str]:
    prompt = (prompt_root / "prompts/query_to_json.txt").read_text()
    token_count, token_kind = _count_tokens(prompt + "{" + query + "}\n" + "JSON:\n")
    raw = module.GPT_response(prompt + "{" + query + "}\n" + "JSON:\n", model_version)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError(f"Could not parse JSON from response: {raw}")
        cleaned = match.group(0)
    return json.loads(cleaned), token_count, token_kind


def _safe_pipeline(
    module,
    query: str,
    prompt_query: str,
    query_json: dict,
    mode: str,
    index: int,
    model_version: str,
    repo_root: Path,
    output_root: Path,
    query_json_tokens: int | None = None,
    query_json_tokenizer: str | None = None,
) -> None:
    path = output_root / "output" / mode / "gpt_nl" / str(index)
    (path / "codes").mkdir(parents=True, exist_ok=True)
    (path / "plans").mkdir(parents=True, exist_ok=True)
    path_str = f"{path}/"

    constraint_to_step_prompt = (repo_root / "prompts/constraint_to_step_nl.txt").read_text()
    step_to_code_destination_cities_prompt = (repo_root / "prompts/step_to_code_destination_cities.txt").read_text()
    step_to_code_departure_dates_prompt = (repo_root / "prompts/step_to_code_departure_dates.txt").read_text()
    step_to_code_transportation_methods_prompt = (
        repo_root / "prompts/step_to_code_transportation_methods.txt"
    ).read_text()
    step_to_code_flight_prompt = (repo_root / "prompts/step_to_code_flight.txt").read_text()
    step_to_code_driving_prompt = (repo_root / "prompts/step_to_code_driving.txt").read_text()
    step_to_code_restaurant_prompt = (repo_root / "prompts/step_to_code_restaurant.txt").read_text()
    step_to_code_attraction_prompt = (repo_root / "prompts/step_to_code_attraction.txt").read_text()
    step_to_code_accommodation_prompt = (
        repo_root / "prompts/step_to_code_accommodation.txt"
    ).read_text()
    step_to_code_budget_prompt = (repo_root / "prompts/step_to_code_budget.txt").read_text()

    CitySearch = module.Cities()
    FlightSearch = module.Flights()
    AttractionSearch = module.Attractions()
    DistanceSearch = module.GoogleDistanceMatrix()
    AccommodationSearch = module.Accommodations()
    RestaurantSearch = module.Restaurants()
    s = module.Optimize()
    variables: dict = {}
    times: List[float] = []
    codes = ""
    success = False
    status = "unknown"
    token_stats = {
        "index": index,
        "tokenizer": query_json_tokenizer,
        "query_to_json_prompt_tokens": query_json_tokens,
        "constraint_to_step_prompt_tokens": None,
        "step_to_code_prompt_tokens": {},
        "total_prompt_tokens": 0,
    }
    if query_json_tokens:
        token_stats["total_prompt_tokens"] += query_json_tokens

    step_to_code_prompts = {
        "Destination cities": step_to_code_destination_cities_prompt,
        "Departure dates": step_to_code_departure_dates_prompt,
        "Transportation methods": step_to_code_transportation_methods_prompt,
        "Flight information": step_to_code_flight_prompt,
        "Driving information": step_to_code_driving_prompt,
        "Restaurant information": step_to_code_restaurant_prompt,
        "Attraction information": step_to_code_attraction_prompt,
        "Accommodation information": step_to_code_accommodation_prompt,
        "Budget": step_to_code_budget_prompt,
    }

    try:
        with (path / "plans" / "query.txt").open("w") as f:
            f.write(query)
        with (path / "plans" / "query.json").open("w") as f:
            json.dump(query_json, f)

        start = time.time()
        constraint_prompt = constraint_to_step_prompt + prompt_query + "\n" + "Steps:\n"
        constraint_tokens, token_kind = _count_tokens(constraint_prompt)
        token_stats["tokenizer"] = token_kind
        token_stats["constraint_to_step_prompt_tokens"] = constraint_tokens
        token_stats["total_prompt_tokens"] += constraint_tokens
        steps_text = module.GPT_response(constraint_prompt, model_version)
        times.append(time.time() - start)

        with (path / "plans" / "steps.txt").open("w") as f:
            f.write(steps_text)

        step_chunks = [chunk.strip() for chunk in steps_text.split("\n\n") if chunk.strip()]
        for step in step_chunks:
            lines_list = step.splitlines()
            if not lines_list:
                continue
            header = lines_list[0]
            lines = "\n".join(lines_list[1:])
            prompt = ""
            step_key = ""
            for key in step_to_code_prompts:
                if key in header:
                    prompt = step_to_code_prompts[key]
                    step_key = key
                    break
            if not prompt:
                continue

            start = time.time()
            step_prompt = prompt + lines
            step_tokens, token_kind = _count_tokens(step_prompt)
            token_stats["tokenizer"] = token_kind
            token_stats["step_to_code_prompt_tokens"][step_key] = step_tokens
            token_stats["total_prompt_tokens"] += step_tokens
            code = module.GPT_response(step_prompt, model_version)
            times.append(time.time() - start)

            code = code.replace("```python", "").replace("```", "").replace("\\_", "_")
            filtered_lines = []
            for line in code.splitlines():
                if line.strip().startswith("##########"):
                    continue
                filtered_lines.append(line)
            code = "\n".join(filtered_lines)
            if step_key != "Destination cities":
                if query_json["days"] == 3:
                    indent = "    "
                elif query_json["days"] == 5:
                    indent = "            "
                else:
                    indent = "                "
                code = indent + code.replace("\n", "\n" + indent)
            codes += code + "\n"
            with (path / "codes" / f"{step_key}.txt").open("w") as f:
                f.write(code)

        solve_prompt = (repo_root / f"prompts/solve_{query_json['days']}.txt").read_text()
        codes += solve_prompt

        scope = dict(module.__dict__)
        scope.update(
            {
                "CitySearch": CitySearch,
                "FlightSearch": FlightSearch,
                "AttractionSearch": AttractionSearch,
                "DistanceSearch": DistanceSearch,
                "AccommodationSearch": AccommodationSearch,
                "RestaurantSearch": RestaurantSearch,
                "s": s,
                "variables": variables,
                "query_json": query_json,
                "path": path_str,
                "success": success,
            }
        )
        start = time.time()
        exec(codes, scope)
        times.append(time.time() - start)
        error_path = path / "plans" / "error.txt"
        if error_path.exists():
            error_path.unlink()
        if (path / "plans" / "plan.txt").exists():
            status = "sat"
        else:
            status = "unsat"
    except Exception as exc:
        with (path / "codes" / "codes.txt").open("w") as f:
            f.write(codes)
        with (path / "plans" / "error.txt").open("w") as f:
            f.write(str(exc))
        status = "error"
    if status in {"sat", "unsat"}:
        error_path = path / "plans" / "error.txt"
        if error_path.exists():
            error_path.unlink()
    with (path / "plans" / "time.txt").open("w") as f:
        for line in times:
            f.write(f"{line}\n")
    with (path / "plans" / "status.json").open("w") as f:
        json.dump({"index": index, "status": status}, f)
    with (output_root / "status.jsonl").open("a") as f:
        f.write(json.dumps({"index": index, "status": status}) + "\n")
    with (path / "plans" / "token_stats.json").open("w") as f:
        json.dump(token_stats, f, indent=2)
    with (output_root / "token_stats.jsonl").open("a") as f:
        f.write(json.dumps(token_stats) + "\n")


def run_single_query(
    query: str,
    query_json: dict | None,
    *,
    index: int,
    set_type: str,
    output_root: Path,
    smt_repo: Path | None = None,
    model_version: str = "gpt-4o",
    full_db: bool = False,
    top_k_cities: int = 6,
) -> Dict[str, object]:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_root = _resolve_smt_repo(smt_repo)
    runtime = _get_runtime(repo_root)
    module = runtime["module"]

    query_tokens = None
    token_kind = None
    if query_json is None:
        query_json, query_tokens, token_kind = _parse_query_json(
            module, repo_root, query, model_version
        )
    prompt_query = _build_prompt_query(query, query_json)

    if full_db:
        _patch_data_paths(runtime["data_root"])
    else:
        query_root = output_root / "filtered_db" / str(index) / "database"
        counts = build_filtered_database(
            query_root,
            query_json,
            runtime["flights_df"],
            runtime["accommodations_df"],
            runtime["restaurants_df"],
            runtime["attractions_df"],
            runtime["distance_df"],
            runtime["state_city_map"],
            top_k_cities,
        )
        (output_root / "filtered_db" / str(index) / "metadata.json").write_text(
            json.dumps(counts, indent=2)
        )
        _patch_data_paths(query_root)

    _safe_pipeline(
        module,
        query,
        prompt_query,
        query_json,
        set_type,
        index,
        model_version,
        repo_root,
        output_root,
        query_tokens,
        token_kind,
    )

    plan_root = output_root / "output" / set_type / "gpt_nl" / str(index) / "plans"
    status = None
    if (plan_root / "status.json").exists():
        status = json.loads((plan_root / "status.json").read_text()).get("status")
    plan_text = None
    if (plan_root / "plan.txt").exists():
        plan_text = (plan_root / "plan.txt").read_text()
    error_text = None
    if (plan_root / "error.txt").exists():
        error_text = (plan_root / "error.txt").read_text()

    return {
        "status": status,
        "plan": plan_text,
        "error": error_text,
        "output_dir": str(plan_root.parent),
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid two-stage (tools -> SMT) runner.")
    parser.add_argument("--set_type", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output_root", default="smt_token_output/hybrid_two_stage_smt")
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=5)
    parser.add_argument("--top_k_cities", type=int, default=6)
    parser.add_argument("--full_db", action="store_true", help="Use full database instead of filtered slices.")
    parser.add_argument("--smt_repo", type=Path, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        dataset = _load_local_dataset(args.set_type)
    except Exception:
        dataset = load_dataset(
            "osunlp/TravelPlanner",
            args.set_type,
            download_mode="reuse_cache_if_exists",
        )[args.set_type]

    data_root = REPO_ROOT / "database"
    flights_df = pd.read_csv(data_root / "flights/clean_Flights_2022.csv").dropna()
    accommodations_df = pd.read_csv(data_root / "accommodations/clean_accommodations_2022.csv").dropna()
    restaurants_df = pd.read_csv(data_root / "restaurants/clean_restaurant_2022.csv").dropna()
    attractions_df = pd.read_csv(data_root / "attractions/attractions.csv").dropna()
    distance_df = pd.read_csv(data_root / "googleDistanceMatrix/distance.csv").dropna()
    state_city_map = load_state_city_map(data_root / "background/citySet_with_states.txt")

    _shim_optional_dependencies()
    repo_root = args.smt_repo
    if repo_root is None:
        repo_root = REPO_ROOT.parent / "LLM_Formal_Travel_Planner"
    repo_root = Path(repo_root)
    module = _load_formal_module(repo_root)
    _maybe_patch_deepseek(module)
    _patch_generate_as_plan(module)
    _patch_steps_response(module)

    start = args.start_idx
    end = min(len(dataset), (start - 1) + args.max_items)
    for idx in range(start, end + 1):
        item = dataset[idx - 1]
        query = item["query"]
        if "org" in item:
            query_json = {
                "org": item["org"],
                "dest": item["dest"],
                "days": item["days"],
                "visiting_city_number": item["visiting_city_number"],
                "date": item["date"],
                "people_number": item["people_number"],
                "local_constraint": item["local_constraint"],
                "budget": item["budget"],
            }
            query_tokens = None
            token_kind = None
        else:
            cached = _load_cached_query_json(output_root, repo_root, args.set_type, idx)
            if cached:
                query_json = cached
                query_tokens = None
                token_kind = None
            else:
                query_json, query_tokens, token_kind = _parse_query_json(
                    module, repo_root, query, "gpt-4o"
                )
        prompt_query = _build_prompt_query(query, query_json)

        if args.full_db:
            _patch_data_paths(REPO_ROOT / "database")
        else:
            query_root = output_root / "filtered_db" / str(idx) / "database"
            counts = build_filtered_database(
                query_root,
                query_json,
                flights_df,
                accommodations_df,
                restaurants_df,
                attractions_df,
                distance_df,
                state_city_map,
                args.top_k_cities,
            )
            (output_root / "filtered_db" / str(idx) / "metadata.json").write_text(
                json.dumps(counts, indent=2)
            )

            _patch_data_paths(query_root)

        _safe_pipeline(
            module,
            query,
            prompt_query,
            query_json,
            args.set_type,
            idx,
            "gpt-4o",
            repo_root,
            output_root,
            query_tokens,
            token_kind,
        )


if __name__ == "__main__":
    main()
