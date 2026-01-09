"""
Wrapper to run the formal SMT-based planner from `../LLM_Formal_Travel_Planner`.

It loads the upstream pipeline (test_travelplanner.py) via importlib, runs it for
TravelPlanner queries, and leaves the generated plans under the upstream repo's
`output/{set_type}/{model_name}/{index}/plans/plan.txt`. This keeps the SMT
logic and prompts in one place while letting us trigger runs from this repo.
"""
import argparse
import importlib.util
import os
import sys
import types
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset
from typing import Optional, List
import json

# Minimal OpenAI shim if the package is unavailable.
try:
    import openai  # type: ignore
except ImportError:
    import requests
    import types
    def _make_openai_shim():
        class ChatCompletion:
            @classmethod
            def create(cls, model, messages, temperature=0.0, top_p=1, frequency_penalty=0, presence_penalty=0):
                base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
                key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ImportError("openai shim: set DEEPSEEK_API_KEY or OPENAI_API_KEY")
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                headers = {"Authorization": f"Bearer {key}"}
                resp = requests.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=200)
                resp.raise_for_status()
                data = resp.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = str(data)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])
        shim = types.SimpleNamespace(ChatCompletion=ChatCompletion, api_key=None, api_base=None)
        sys.modules["openai"] = shim
        openai = shim  # type: ignore
    _make_openai_shim()


# Load environment variables from a local .env (supports DeepSeek creds by default).
load_dotenv()

# Default to DeepSeek model tag if not provided (aligns with downstream file naming).
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "deepseek:deepseek-chat")


def _load_formal_module(repo_root: Path):
    """
    Dynamically import the SMT pipeline module without modifying our codebase.
    """
    target = repo_root / "test_travelplanner.py"
    if not target.exists():
        raise FileNotFoundError(f"Expected SMT runner at {target}")

    spec = importlib.util.spec_from_file_location("formal_travelplanner", target)
    module = importlib.util.module_from_spec(spec)

    # test_travelplanner.py changes cwd (current working directory) to its own directory on import; record and restore.
    cwd = Path.cwd()
    original_sys_path: List[str] = sys.path.copy()
    # Ensure the SMT repo root is on sys.path so relative imports like openai_func resolve.
    sys.path.insert(0, str(repo_root))
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        os.chdir(cwd)
        sys.path = original_sys_path
    return module


def _shim_optional_dependencies():
    """
    Provide lightweight shims for optional deps (anthropic, mistralai) to prevent import errors
    when only DeepSeek/OpenAI paths are used.
    """
    if "anthropic" not in sys.modules:
        fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key=None: None)
        sys.modules["anthropic"] = fake_anthropic
    # Shim mistralai modules
    if "mistralai" not in sys.modules:
        sys.modules["mistralai"] = types.SimpleNamespace()
    if "mistralai.client" not in sys.modules:
        class FakeMistralClient:
            def __init__(self, api_key=None):
                self.api_key = api_key

            def chat(self, *args, **kwargs):
                raise RuntimeError("Mistral client shim: install mistralai to use Mixtral_response.")
        sys.modules["mistralai.client"] = types.SimpleNamespace(MistralClient=FakeMistralClient)
    if "mistralai.models.chat_completion" not in sys.modules:
        class FakeChatMessage:
            def __init__(self, role=None, content=None):
                self.role = role
                self.content = content
        sys.modules["mistralai.models.chat_completion"] = types.SimpleNamespace(ChatMessage=FakeChatMessage)

    # Shim tiktoken only if it's truly unavailable.
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        class _DummyEncoder:
            def encode(self, text):
                return list(text.encode("utf-8"))
        class _DummyTikToken:
            def encoding_for_model(self, name):
                return _DummyEncoder()
        sys.modules["tiktoken"] = _DummyTikToken()

    # Shim langchain ChatOpenAI and get_openai_callback if missing.
    if "langchain" not in sys.modules:
        import types as _types
        sys.modules["langchain"] = _types.ModuleType("langchain")
    if "langchain.chat_models" not in sys.modules:
        class ChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                return None
        mod = types.ModuleType("langchain.chat_models")
        mod.ChatOpenAI = ChatOpenAI
        sys.modules["langchain.chat_models"] = mod
    if "langchain.callbacks" not in sys.modules:
        class _DummyCallback:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
        def get_openai_callback():
            return _DummyCallback()
        mod = types.ModuleType("langchain.callbacks")
        mod.get_openai_callback = get_openai_callback
        sys.modules["langchain.callbacks"] = mod
    if "langchain.llms" not in sys.modules:
        mod = types.ModuleType("langchain.llms")
        sys.modules["langchain.llms"] = mod
    if "langchain.llms.base" not in sys.modules:
        class BaseLLM:
            pass
        mod = types.ModuleType("langchain.llms.base")
        mod.BaseLLM = BaseLLM
        sys.modules["langchain.llms.base"] = mod
    if "langchain.prompts" not in sys.modules:
        class PromptTemplate:
            def __init__(self, *args, **kwargs):
                pass
        mod = types.ModuleType("langchain.prompts")
        mod.PromptTemplate = PromptTemplate
        sys.modules["langchain.prompts"] = mod
    if "langchain.schema" not in sys.modules:
        class _Msg:
            def __init__(self, content=None, role=None):
                self.content = content
                self.role = role
        mod = types.ModuleType("langchain.schema")
        mod.AIMessage = _Msg
        mod.HumanMessage = _Msg
        mod.SystemMessage = _Msg
        sys.modules["langchain.schema"] = mod
    if "langchain_google_genai" not in sys.modules:
        class ChatGoogleGenerativeAI:
            def __init__(self, *args, **kwargs):
                pass
        mod = types.ModuleType("langchain_google_genai")
        mod.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
        sys.modules["langchain_google_genai"] = mod

    # Shim z3 only if truly missing.
    try:
        import z3  # type: ignore
    except ImportError:
        class _Dummy:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return self
            def __iter__(self): return iter([])
        def Int(name): return _Dummy()
        def Real(name): return _Dummy()
        def IntVal(v): return v
        def RealVal(v): return v
        def BoolVal(v): return bool(v)
        def Optimize(): return _Dummy()
        def Solver(): return _Dummy()
        def If(a,b,c): return b if a else c
        def And(*args): return all(args)
        def Or(*args): return any(args)
        def Not(x): return not x
        def Datatype(name): return _Dummy()
        sys.modules["z3"] = types.SimpleNamespace(
            Int=Int, Real=Real, IntVal=IntVal, RealVal=RealVal, BoolVal=BoolVal,
            Optimize=Optimize, Solver=Solver, If=If, And=And, Or=Or, Not=Not, Datatype=Datatype
        )


def _patch_data_paths(data_root: Path, max_city_candidates: Optional[int] = None):
    """
    Force upstream tool loaders to read from our TravelPlanner database directory.
    """
    try:
        import tools.cities.apis as cities
        orig_cities_init = cities.Cities.__init__
        orig_cities_run = cities.Cities.run
        def cities_init(self, path=None):
            path = str(data_root / "background/citySet_with_states.txt")
            return orig_cities_init(self, path=path)
        def cities_run(self, state, *args, **kwargs):
            result = orig_cities_run(self, state, *args, **kwargs)
            if (
                max_city_candidates
                and isinstance(result, list)
                and len(result) > max_city_candidates
            ):
                return result[:max_city_candidates]
            return result
        cities.Cities.__init__ = cities_init
        cities.Cities.run = cities_run
    except Exception:
        pass

    try:
        import tools.accommodations.apis as accommodations
        orig_acc_init = accommodations.Accommodations.__init__
        def acc_init(self, path=None):
            path = str(data_root / "accommodations/clean_accommodations_2022.csv")
            return orig_acc_init(self, path=path)
        accommodations.Accommodations.__init__ = acc_init
    except Exception:
        pass

    try:
        import tools.restaurants.apis as restaurants
        orig_rest_init = restaurants.Restaurants.__init__
        def rest_init(self, path=None):
            path = str(data_root / "restaurants/clean_restaurant_2022.csv")
            return orig_rest_init(self, path=path)
        restaurants.Restaurants.__init__ = rest_init
    except Exception:
        pass

    try:
        import tools.flights.apis as flights
        orig_flights_init = flights.Flights.__init__
        def flights_init(self, path=None):
            path = str(data_root / "flights/clean_Flights_2022.csv")
            return orig_flights_init(self, path=path)
        flights.Flights.__init__ = flights_init
    except Exception:
        pass

    try:
        import tools.attractions.apis as attractions
        orig_attr_init = attractions.Attractions.__init__
        def attr_init(self, path=None):
            path = str(data_root / "attractions/attractions.csv")
            return orig_attr_init(self, path=path)
        attractions.Attractions.__init__ = attr_init
    except Exception:
        pass

    try:
        import tools.googleDistanceMatrix.apis as gdm
        import pandas as pd
        def gdm_init(self, subscription_key: str = ""):
            self.gplaces_api_key = subscription_key
            self.data = pd.read_csv(data_root / "googleDistanceMatrix/distance.csv")
            print("GoogleDistanceMatrix loaded.")
        gdm.GoogleDistanceMatrix.__init__ = gdm_init
    except Exception:
        pass


def _load_queries(set_type: str, dataset_path: Optional[Path]) -> List[dict]:
    """
    Load queries either from a local dataset file (json/jsonl) or from HuggingFace (osunlp/TravelPlanner).
    """
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        items: List[dict] = []
        if path.suffix in [".jsonl", ".jsonlines"]:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        elif path.suffix == ".json":
            with path.open() as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Allow {"validation": [...]} shaped files.
                    data = data.get(set_type, [])
                if not isinstance(data, list):
                    raise ValueError(f"Unsupported JSON format in {path}")
                items = data
        else:
            raise ValueError(f"Unsupported dataset file type: {path}")
        return items

    try:
        return load_dataset("osunlp/TravelPlanner", set_type)["validation" if set_type == "validation" else set_type]  # type: ignore[index]
    except Exception as e:
        raise RuntimeError(
            "Failed to load dataset from HuggingFace (osunlp/TravelPlanner). "
            "Set --dataset_path to a local json/jsonl with a 'query' field, "
            "or ensure HF access works."
        ) from e


def _maybe_patch_deepseek(module):
    """
    If DEEPSEEK_API_KEY is set, redirect GPT_response calls to DeepSeek's API
    (OpenAI-compatible). This avoids editing the upstream repo.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return

    try:
        import openai  # type: ignore
    except Exception:
        # fall back to shim
        import importlib
        openai = importlib.import_module("openai")

    # Configure OpenAI client to talk to DeepSeek using .env values.
    openai.api_key = deepseek_key
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "200"))

    def ds_response(messages, _model_name=None):
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages},
            ],
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            timeout=timeout,
        )
        return resp.choices[0].message.content

    # Patch both module-level and the imported openai_func reference.
    module.GPT_response = ds_response
    if hasattr(module, "openai_func"):
        module.openai_func.GPT_response = ds_response
    print("Patched GPT_response to use DeepSeek (model=%s)" % model_name)


def run_smt(
    set_type: str = "validation",
    model_name: str = DEFAULT_MODEL_NAME,
    max_items: Optional[int] = None,
    smt_repo: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    skip_existing: bool = True,
    start_idx: int = 1,
    max_city_candidates: Optional[int] = None,
):
    # Resolve SMT repo location: CLI arg > env > common sibling defaults.
    if smt_repo:
        repo_root = Path(smt_repo)
    elif os.getenv("SMT_REPO"):
        repo_root = Path(os.getenv("SMT_REPO"))  # type: ignore[arg-type]
    else:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] / "LLM_Formal_Travel_Planner",  # inside current repo (older default)
            here.parents[2] / "LLM_Formal_Travel_Planner",  # sibling to current repo
        ]
        repo_root = None
        for candidate in candidates:
            if candidate.is_dir():
                repo_root = candidate
                break
        if repo_root is None:
            raise FileNotFoundError(
                f"Cannot locate LLM_Formal_Travel_Planner. Tried: {candidates}. "
                f"Please pass --smt_repo or set SMT_REPO (e.g., {candidates[-1]})."
            )

    _shim_optional_dependencies()
    module = _load_formal_module(repo_root)
    # Ensure data files resolve from this repo's database folder.
    data_root = Path(__file__).resolve().parents[1] / "database"
    _patch_data_paths(data_root, max_city_candidates)
    _maybe_patch_deepseek(module)

    dataset = _load_queries(set_type, dataset_path)

    repo_root_path = Path(__file__).resolve().parents[1]
    output_base = (
        Path(output_root)
        if output_root is not None
        else repo_root_path / "smt_output"
    )
    if not output_base.is_absolute():
        output_base = (repo_root_path / output_base).resolve()

    # Monkeypatch pipeline to handle arbitrary step formatting more robustly.
    def patched_pipeline(query, mode, model, index, model_version=None):
        import json
        import time
        import os
        import pdb  # noqa: F401
        from z3 import Optimize  # type: ignore
        from tools.cities.apis import Cities
        from tools.flights.apis import Flights
        from tools.attractions.apis import Attractions
        from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
        from tools.accommodations.apis import Accommodations
        from tools.restaurants.apis import Restaurants

        output_root = output_base / mode / "gpt_nl" / str(index)
        os.makedirs(output_root / "codes", exist_ok=True)
        os.makedirs(output_root / "plans", exist_ok=True)

        with open("prompts/query_to_json.txt", "r") as file:
            query_to_json_prompt = file.read()
        with open("prompts/constraint_to_step_nl.txt", "r") as file:
            constraint_to_step_prompt = file.read()
        with open("prompts/step_to_code_destination_cities.txt", "r") as file:
            step_to_code_destination_cities_prompt = file.read()
        with open("prompts/step_to_code_departure_dates.txt", "r") as file:
            step_to_code_departure_dates_prompt = file.read()
        with open("prompts/step_to_code_transportation_methods.txt", "r") as file:
            step_to_code_transportation_methods_prompt = file.read()
        with open("prompts/step_to_code_flight.txt", "r") as file:
            step_to_code_flight_prompt = file.read()
        with open("prompts/step_to_code_driving.txt", "r") as file:
            step_to_code_driving_prompt = file.read()
        with open("prompts/step_to_code_restaurant.txt", "r") as file:
            step_to_code_restaurant_prompt = file.read()
        with open("prompts/step_to_code_attraction.txt", "r") as file:
            step_to_code_attraction_prompt = file.read()
        with open("prompts/step_to_code_accommodation.txt", "r") as file:
            step_to_code_accommodation_prompt = file.read()
        with open("prompts/step_to_code_budget.txt", "r") as file:
            step_to_code_budget_prompt = file.read()

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

        CitySearch = Cities()
        FlightSearch = Flights()
        AttractionSearch = Attractions()
        DistanceSearch = GoogleDistanceMatrix()
        AccommodationSearch = Accommodations()
        RestaurantSearch = Restaurants()
        s = Optimize()
        variables = {}
        times = []
        codes = "from z3 import *\n\n"
        success = False
        path = str(output_root) + "/"

        try:
            if model == "gpt":
                print(f"[{index}] query_to_json start")
                raw = module.GPT_response(query_to_json_prompt + "{" + query + "}\n" + "JSON:\n", model_version)
                print(f"[{index}] query_to_json done")
            elif model == "claude":
                raw = module.Claude_response(query_to_json_prompt + "{" + query + "}\n" + "JSON:\n")
            elif model == "mixtral":
                raw = module.Mixtral_response(query_to_json_prompt + "{" + query + "}\n" + "JSON:\n", "json")
            else:
                raise ValueError(f"Unsupported model {model}")

            cleaned = raw.replace("```json", "").replace("```", "").strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                pass
            else:
                # try to find a json object substring
                import re
                m = re.search(r"\{.*\}", cleaned, re.S)
                if not m:
                    raise ValueError(f"Could not parse JSON from response: {raw}")
                cleaned = m.group(0)
            query_json = json.loads(cleaned)

            with open(output_root / "plans" / "query.txt", "w") as f:
                f.write(query)
            with open(output_root / "plans" / "query.json", "w") as f:
                json.dump(query_json, f)

            start = time.time()
            if model == "gpt":
                print(f"[{index}] constraint_to_step start")
                steps_text = module.GPT_response(constraint_to_step_prompt + query + "\n" + "Steps:\n", model_version)
                print(f"[{index}] constraint_to_step done")
            elif model == "claude":
                steps_text = module.Claude_response(constraint_to_step_prompt + query + "\n" + "Steps:\n")
            elif model == "mixtral":
                steps_text = module.Mixtral_response(constraint_to_step_prompt + query + "\n" + "Steps:\n")
            else:
                raise ValueError(f"Unsupported model {model}")
            times.append(time.time() - start)

            with open(output_root / "plans" / "steps.txt", "w") as f:
                f.write(steps_text)

            step_chunks = [chunk.strip() for chunk in steps_text.split("\n\n") if chunk.strip()]
            step_items = []
            for step in step_chunks:
                lines_list = step.splitlines()
                if not lines_list:
                    continue
                header = lines_list[0]
                lines = "\n".join(lines_list[1:])
                prompt = ""
                step_key = ""
                for key in step_to_code_prompts.keys():
                    if key in header:
                        prompt = step_to_code_prompts[key]
                        step_key = key
                        break
                if not prompt:
                    continue
                step_items.append({"key": step_key, "lines": lines, "prompt": prompt})

            def generate_per_section(step_items_local):
                nonlocal codes
                for item in step_items_local:
                    step_key = item["key"]
                    prompt = item["prompt"]
                    lines = item["lines"]
                    start = time.time()
                    if model == "gpt":
                        code = module.GPT_response(prompt + lines, model_version)
                    elif model == "claude":
                        code = module.Claude_response(prompt + lines)
                    elif model == "mixtral":
                        code = module.Mixtral_response(
                            prompt
                            + "\nRespond with python codes only, do not add \\ in front of symbols like _ or *.\n Follow the indentation of provided examples carefully, indent after for-loops!\n"
                            + lines,
                            "code",
                        )
                    else:
                        raise ValueError(f"Unsupported model {model}")
                    times.append(time.time() - start)

                    code = code.replace("```python", "").replace("```", "").replace("\\_", "_")
                    if step_key != "Destination cities":
                        if query_json["days"] == 3:
                            code = code.replace("\n", "\n    ")
                        elif query_json["days"] == 5:
                            code = code.replace("\n", "\n            ")
                        else:
                            code = code.replace("\n", "\n                ")
                    codes += code + "\n"
                    with open(output_root / "codes" / f"{step_key}.txt", "w") as f:
                        f.write(code)

            combine_steps = os.getenv("COMBINE_STEP_TO_CODE", "0") == "1"
            combined_used = False
            if combine_steps and step_items:
                print(f"[{index}] combined_step_to_code start")
                combined_prompt = (
                    "You will generate code for multiple sections. For each section, follow the PROMPT and STEPS. "
                    "Output all sections in order. You MUST wrap each section with markers exactly as:\n"
                    "########## <Section Name> response##########\n<code>\n########## <Section Name> response ends##########\n"
                    "Do not add any extra text outside the markers.\n\n"
                )
                for item in step_items:
                    combined_prompt += f"=== SECTION: {item['key']} ===\n"
                    combined_prompt += "PROMPT:\n" + item["prompt"] + "\n"
                    combined_prompt += "STEPS:\n" + item["lines"] + "\n\n"

                start = time.time()
                if model == "gpt":
                    combined_code = module.GPT_response(combined_prompt, model_version)
                elif model == "claude":
                    combined_code = module.Claude_response(combined_prompt)
                elif model == "mixtral":
                    combined_code = module.Mixtral_response(combined_prompt, "code")
                else:
                    raise ValueError(f"Unsupported model {model}")
                print(f"[{index}] combined_step_to_code done")
                times.append(time.time() - start)

                def extract_section(output, key):
                    start_tag = f"########## {key} response##########"
                    end_tag = f"########## {key} response ends##########"
                    if start_tag in output and end_tag in output:
                        return output.split(start_tag, 1)[1].split(end_tag, 1)[0]
                    return ""

                # Save raw combined output for debugging when markers are missing.
                with open(output_root / "codes" / "combined_response.txt", "w") as f:
                    f.write(combined_code)

                combined_used = True
                for item in step_items:
                    step_key = item["key"]
                    code = extract_section(combined_code, step_key)
                    if not code:
                        # fallback: if no markers, skip to per-section calls
                        combine_steps = False
                        break
                    code = code.replace("```python", "").replace("```", "").replace("\\_", "_").strip()
                    # Basic syntax check; fallback if any section fails to compile.
                    try:
                        compile(code, "<section>", "exec")
                    except Exception:
                        combine_steps = False
                        break
                    if step_key != "Destination cities":
                        if query_json["days"] == 3:
                            code = code.replace("\n", "\n    ")
                        elif query_json["days"] == 5:
                            code = code.replace("\n", "\n            ")
                        else:
                            code = code.replace("\n", "\n                ")
                    codes += code + "\n"
                    with open(output_root / "codes" / f"{step_key}.txt", "w") as f:
                        f.write(code)

            if not combine_steps and step_items:
                print(f"[{index}] per_section_step_to_code start")
                codes = "from z3 import *\n\n"
                generate_per_section(step_items)
                print(f"[{index}] per_section_step_to_code done")

            with open("prompts/solve_{}.txt".format(query_json["days"]), "r") as f:
                codes += f.read()

            start = time.time()
            print(f"[{index}] exec start")
            try:
                # Persist the full code before execution for debugging.
                with open(output_root / "codes" / "codes.txt", "w") as f:
                    f.write(codes)
                # Execute with the upstream module's namespace so helper fns (e.g., get_arrivals_list) are in scope.
                module.__dict__.update({
                    "CitySearch": CitySearch,
                    "FlightSearch": FlightSearch,
                    "AttractionSearch": AttractionSearch,
                    "DistanceSearch": DistanceSearch,
                    "AccommodationSearch": AccommodationSearch,
                    "RestaurantSearch": RestaurantSearch,
                    "s": s,
                    "variables": variables,
                    "query_json": query_json,
                    "path": path,
                    "success": success,
                })
                exec(codes, module.__dict__)
            except Exception as exec_err:
                # If combined code failed, fall back to per-section generation once.
                if combined_used:
                    codes = "from z3 import *\n\n"
                    generate_per_section(step_items)
                    # Persist fallback code and retry with timeout.
                    with open(output_root / "codes" / "codes.txt", "w") as f:
                        f.write(codes)
                    module.__dict__.update({
                        "CitySearch": CitySearch,
                        "FlightSearch": FlightSearch,
                        "AttractionSearch": AttractionSearch,
                        "DistanceSearch": DistanceSearch,
                        "AccommodationSearch": AccommodationSearch,
                        "RestaurantSearch": RestaurantSearch,
                        "s": s,
                        "variables": variables,
                        "query_json": query_json,
                        "path": path,
                        "success": success,
                    })
                    exec(codes, module.__dict__)
                else:
                    raise exec_err
            times.append(time.time() - start)
            print(f"[{index}] exec done")
        except Exception as e:
            with open(output_root / "codes" / "codes.txt", "w") as f:
                f.write(codes)
            with open(output_root / "plans" / "error.txt", "w") as f:
                f.write(str(e))
            with open(output_root / "plans" / "time.txt", "w") as f:
                for line in times:
                    f.write(f"{line}\n")
            # If no plan was produced, log it.
            plan_file = output_root / "plans" / "plan.txt"
            if not plan_file.exists():
                with open(output_root / "plans" / "error.txt", "a") as f:
                    f.write("\nNo plan generated (unsat or execution failed).")

    module.pipeline = patched_pipeline

    dataset_len = len(dataset)
    if start_idx < 1 or start_idx > dataset_len:
        raise ValueError(f"start_idx {start_idx} out of range (1..{dataset_len})")
    if max_items is None:
        end_idx = dataset_len
    else:
        end_idx = min(dataset_len, (start_idx - 1) + max_items)
    total = end_idx - (start_idx - 1)
    print(f"Running SMT resolver for {total} queries (set={set_type}, model={model_name})")

    orig_cwd = Path.cwd()
    try:
        # Ensure relative file reads (prompts/*) work by running inside the SMT repo.
        os.chdir(repo_root)
        for idx in range(start_idx - 1, end_idx):
            out_dir = output_base / set_type / "gpt_nl" / str(idx + 1) / "plans" / "plan.txt"
            if skip_existing and out_dir.exists():
                continue
            query = dataset[idx]["query"]
            # The upstream pipeline only branches on model in {'gpt','claude','mixtral'}.
            # We pass 'gpt' here to use GPT_response, which we have already patched to DeepSeek.
            pipeline_model = "gpt"
            # test_travelplanner expects 1-based indexing for file paths
            module.pipeline(query, set_type, pipeline_model, idx + 1, "gpt-4o")
    finally:
        os.chdir(orig_cwd)


def main():
    parser = argparse.ArgumentParser(description="Run SMT-based TravelPlanner pipeline.")
    parser.add_argument("--set_type", choices=["train", "validation", "test"], default="validation")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Model flag used by upstream runner (defaults to DeepSeek tag if set in .env or falls back to deepseek:deepseek-chat).",
    )
    parser.add_argument(
        "--max_items", type=int, default=None, help="Limit number of queries for smoke runs."
    )
    parser.add_argument(
        "--smt_repo",
        type=Path,
        default=None,
        help="Path to LLM_Formal_Travel_Planner repo (defaults to ../LLM_Formal_Travel_Planner).",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=os.getenv("DATASET_PATH"),
        help="Local json/jsonl file with queries (optional). If omitted, tries to load osunlp/TravelPlanner from HF.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Base output directory (defaults to ./smt_output).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1,
        help="1-based index to start from when iterating queries (skip earlier).",
    )
    parser.add_argument(
        "--max_city_candidates",
        type=int,
        default=None,
        help="Optional cap on number of candidate cities returned by CitySearch.run (helps avoid combinatorial blowups).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip queries whose plan.txt already exists.",
    )
    args = parser.parse_args()

    run_smt(
        args.set_type,
        args.model_name,
        args.max_items,
        args.smt_repo,
        args.dataset_path,
        output_root=args.output_root,
        skip_existing=args.skip_existing,
        start_idx=args.start_idx,
        max_city_candidates=args.max_city_candidates,
    )


if __name__ == "__main__":
    main()
