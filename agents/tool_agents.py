import re, string, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "tools" / "planner"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
try:
    import tiktoken
except Exception:
    class _DummyEnc:
        def encode(self, text: str):
            return text.split()

    class tiktoken:  # type: ignore[no-redef]
        @staticmethod
        def encoding_for_model(_model: str) -> _DummyEnc:
            return _DummyEnc()
from pandas import DataFrame
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.callbacks import get_openai_callback
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from prompts import zeroshot_react_agent_prompt
from utils.func import load_line_json_data, save_file
from utils.path_utils import normalize_output_dir
from utils.token_reduction import compress_tool_output, summarize_tool_output
from utils.user_profile import build_user_profile, format_profile
from utils.constraint_policy import build_constraint_policy, format_constraint_policy
from tools.planner.apis import SmtPlanner
from agents.aggregate_smt_output import parse_plan as parse_smt_plan
import sys
import json
import openai
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from datasets import load_dataset
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')


pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {"FlightSearch":"flights","AttractionSearch":"attractions","GoogleDistanceMatrix":"googleDistanceMatrix","AccommodationSearch":"accommodation","RestaurantSearch":"restaurants","Planner":"planner","NotebookWrite":"notebook","CitySearch":"cities"}

class CityError(Exception):
    pass

class DateError(Exception):
    pass

def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

class ReactAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name = 'gpt-3.5-turbo-1106',
                 planner_llm_name = 'gpt-3.5-turbo-1106',
                #  logs_path = '../logs/',
                 city_file_path = '../database/background/citySet.txt'
                 ) -> None: 

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode
        self.token_reduction_enabled = os.getenv("TOKEN_REDUCTION", "1") != "0"
        self.shared_tool_cache = os.getenv("SHARED_TOOL_CACHE", "0") == "1"
        self.tool_cache = {}
        self.token_audit_enabled = os.getenv("TOKEN_REDUCTION_AUDIT", "0") == "1"
        self.token_audit_entries = []
        self.token_audit_cache_hits = 0
        self.token_audit_cache_misses = 0

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt

        self.json_log = []

        self.current_observation = ''
        self.current_data = None
        self.user_profile = None
        self.query_context = None
        self.query_item = None
        self.query_index = None
        self.user_profile_mode = os.getenv("USER_PROFILE_MODE", "off").lower()
        self.constraint_policy_mode = os.getenv("CONSTRAINT_POLICY_MODE", "off").lower()
        self.constraint_policy_text = None
        self.notebook_keep_days = _parse_env_int("NOTEBOOK_CONTEXT_WINDOW_DAYS")
        self.notebook_keep_undated = os.getenv("NOTEBOOK_CONTEXT_KEEP_UNDATED", "1") == "1"
        self.notebook_summary = os.getenv("NOTEBOOK_CONTEXT_SUMMARY", "1") == "1"

        if react_llm_name.startswith('ollama:'):
            # Use a local Ollama model via LangChain's ChatOllama wrapper.
            # Example: --model_name ollama:llama3
            ollama_model = react_llm_name.split(":", 1)[1] or "llama3"
            self.max_token_length = 30000
            self.llm = ChatOllama(
                model=ollama_model,
                temperature=0,
            )
        elif 'gpt-3.5' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=1,
                     max_tokens=256,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs={"stop": stop_list})
            
        elif 'gpt-4' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs={"stop": stop_list})
            
        elif react_llm_name in ['mistral-7B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})
            
        elif react_llm_name in ['mixtral']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})
            
        elif react_llm_name in ['ChatGLM3-6B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                     temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})
        
        elif react_llm_name in ['gemini']:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required when using 'gemini' model. Please set it in your .env file.")
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.max_token_length = 30000
        elif react_llm_name.startswith('deepseek:') or react_llm_name.startswith('deepseek-'):
            # DeepSeek models use OpenAI-compatible API
            deepseek_model = react_llm_name.replace('deepseek:', '') if ':' in react_llm_name else react_llm_name
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is required when using DeepSeek models. Please set it in your .env file.")
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                model_name=deepseek_model,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com/v1",
                model_kwargs={"stop": stop_list}
            )


        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []

        # self.log_path = logs_path + datetime.now().strftime('%Y%m%d%H%M%S') + '.out'
        # self.log_file = open(self.log_path, 'a+')

        # print("logs will be stored in " + self.log_path)

        self.city_set = self.load_city(city_set_path=city_file_path)

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def run(self, query, query_item=None, query_index=None, reset=True) -> None:

        self.query = query
        self.query_context = None
        self.user_profile = None
        self.query_item = query_item
        self.query_index = query_index
        self.constraint_policy_text = None
        if self.token_audit_enabled:
            self.token_audit_entries = []
            self.token_audit_cache_hits = 0
            self.token_audit_cache_misses = 0
        if query_item and self.user_profile_mode in {"core", "full"}:
            self.user_profile = build_user_profile(query_item)
            include_secondary = self.user_profile_mode == "full"
            profile_text = format_profile(self.user_profile, include_secondary)
            if self.user_profile_mode == "core":
                self.query_context = f"Core constraints (JSON): {profile_text}"
            else:
                self.query_context = (
                    f"Core constraints (JSON): {profile_text}\n"
                    f"Original query: {query}"
                )
        if query_item and self.constraint_policy_mode != "off":
            policy = build_constraint_policy(query_item)
            policy_text = format_constraint_policy(policy)
            self.constraint_policy_text = f"Constraint policy (JSON): {policy_text}"
        
        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought":"",
                              "action": "", "observation": "", "state":""})

        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()

        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:',"")
        # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')


        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()

        if action == None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action


        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:',"")


        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return


        # action_type, action_arg = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        # self.log_file.write(self.scratchpad.split('\n')[-1]+'\n')

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '

        if action == None or action == '' or action == '\n':
            action_type = None 
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        
        else:
            action_type, action_arg = parse_action(action)
            
            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'
                
                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return
                    
                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            if action_type == 'FlightSearch':
                try:
                    if validate_date_format(action_arg.split(', ')[2]) and validate_city_format(action_arg.split(', ')[0],self.city_set ) and validate_city_format(action_arg.split(', ')[1],self.city_set):
                        tool_key = "flights"
                        self._mask_previous_observation()
                        cached = self._get_cached_tool_result(tool_key, action_arg)
                        if cached is None:
                            raw = self.tools['flights'].run(action_arg.split(', ')[0], action_arg.split(', ')[1], action_arg.split(', ')[2])
                            compressed = compress_tool_output(tool_key, raw)
                            if self.token_reduction_enabled:
                                self.current_data = compressed
                                self.current_observation = summarize_tool_output(tool_key, self.current_data)
                            else:
                                self.current_data = to_string(raw)
                                self.current_observation = self.current_data
                            self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                            self._cache_tool_result(tool_key, action_arg, self.current_data)
                            cached_tag = ""
                        else:
                            self.current_data = cached
                            self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                            cached_tag = "[cached] "
                        if cached is not None and self.token_reduction_enabled:
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        elif cached is not None:
                            self.current_observation = str(self.current_data)
                        self.current_observation = f"{cached_tag}{self.current_observation}"
                        self.scratchpad += self.current_observation 
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except DateError:
                    self.retry_record['flights'] += 1
                    self.current_observation = f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.scratchpad += f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.json_log[-1]['state'] = f'Illegal args. DateError'

                except ValueError as e:
                    self.retry_record['flights'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['flights'] += 1
                    self.current_observation = f'Illegal Flight Search. Please try again.'
                    self.scratchpad += f'Illegal Flight Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AttractionSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        tool_key = "attractions"
                        self._mask_previous_observation()
                        cached = self._get_cached_tool_result(tool_key, action_arg)
                        if cached is None:
                            raw = self.tools['attractions'].run(action_arg)
                            compressed = compress_tool_output(tool_key, raw)
                            if self.token_reduction_enabled:
                                self.current_data = compressed
                                self.current_observation = summarize_tool_output(tool_key, self.current_data)
                            else:
                                self.current_data = to_string(raw)
                                self.current_observation = self.current_data
                            self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                            self._cache_tool_result(tool_key, action_arg, self.current_data)
                            cached_tag = ""
                        else:
                            self.current_data = cached
                            self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                            cached_tag = "[cached] "
                        if cached is not None and self.token_reduction_enabled:
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        elif cached is not None:
                            self.current_observation = str(self.current_data)
                        self.current_observation = f"{cached_tag}{self.current_observation}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    self.retry_record['attractions'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AccommodationSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        tool_key = "accommodations"
                        self._mask_previous_observation()
                        cached = self._get_cached_tool_result(tool_key, action_arg)
                        if cached is None:
                            raw = self.tools['accommodations'].run(action_arg)
                            compressed = compress_tool_output(tool_key, raw)
                            if self.token_reduction_enabled:
                                self.current_data = compressed
                                self.current_observation = summarize_tool_output(tool_key, self.current_data)
                            else:
                                self.current_data = to_string(raw)
                                self.current_observation = self.current_data
                            self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                            self._cache_tool_result(tool_key, action_arg, self.current_data)
                            cached_tag = ""
                        else:
                            self.current_data = cached
                            self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                            cached_tag = "[cached] "
                        if cached is not None and self.token_reduction_enabled:
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        elif cached is not None:
                            self.current_observation = str(self.current_data)
                        self.current_observation = f"{cached_tag}{self.current_observation}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e :
                    self.retry_record['accommodations'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = f'Illegal Accommodation Search. Please try again.'
                    self.scratchpad += f'Illegal Accommodation Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'RestaurantSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        tool_key = "restaurants"
                        self._mask_previous_observation()
                        cached = self._get_cached_tool_result(tool_key, action_arg)
                        if cached is None:
                            raw = self.tools['restaurants'].run(action_arg)
                            compressed = compress_tool_output(tool_key, raw)
                            if self.token_reduction_enabled:
                                self.current_data = compressed
                                self.current_observation = summarize_tool_output(tool_key, self.current_data)
                            else:
                                self.current_data = to_string(raw)
                                self.current_observation = self.current_data
                            self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                            self._cache_tool_result(tool_key, action_arg, self.current_data)
                            cached_tag = ""
                        else:
                            self.current_data = cached
                            self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                            cached_tag = "[cached] "
                        if cached is not None and self.token_reduction_enabled:
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        elif cached is not None:
                            self.current_observation = str(self.current_data)
                        self.current_observation = f"{cached_tag}{self.current_observation}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['restaurants'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'
                    
            elif action_type == "CitySearch":
                try:
                    tool_key = "cities"
                    self._mask_previous_observation()
                    cached = self._get_cached_tool_result(tool_key, action_arg)
                    if cached is None:
                        raw = self.tools['cities'].run(action_arg)
                        compressed = compress_tool_output(tool_key, raw)
                        if self.token_reduction_enabled:
                            self.current_data = compressed
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        else:
                            self.current_data = str(raw)
                            self.current_observation = self.current_data
                        self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                        self._cache_tool_result(tool_key, action_arg, self.current_data)
                        cached_tag = ""
                    else:
                        self.current_data = cached
                        self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                        cached_tag = "[cached] "
                    if cached is not None and self.token_reduction_enabled:
                        self.current_observation = summarize_tool_output(tool_key, self.current_data)
                    elif cached is not None:
                        self.current_observation = str(self.current_data)
                    self.current_observation = f"{cached_tag}{self.current_observation}"
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['cities'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. State Error'

                except Exception as e:
                    print(e)
                    self.retry_record['cities'] += 1
                    self.current_observation = f'Illegal City Search. Please try again.'
                    self.scratchpad += f'Illegal City Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'


            elif action_type == 'GoogleDistanceMatrix':

                try:
                    tool_key = "googleDistanceMatrix"
                    self._mask_previous_observation()
                    cached = self._get_cached_tool_result(tool_key, action_arg)
                    if cached is None:
                        raw = self.tools['googleDistanceMatrix'].run(action_arg.split(', ')[0],action_arg.split(', ')[1],action_arg.split(', ')[2])
                        compressed = compress_tool_output(tool_key, raw)
                        if self.token_reduction_enabled:
                            self.current_data = compressed
                            self.current_observation = summarize_tool_output(tool_key, self.current_data)
                        else:
                            self.current_data = str(raw)
                            self.current_observation = self.current_data
                        self._record_token_audit(tool_key, action_arg, raw, compressed, cached=False)
                        self._cache_tool_result(tool_key, action_arg, self.current_data)
                        cached_tag = ""
                    else:
                        self.current_data = cached
                        self._record_token_audit(tool_key, action_arg, None, cached, cached=True)
                        cached_tag = "[cached] "
                    if cached is not None and self.token_reduction_enabled:
                        self.current_observation = summarize_tool_output(tool_key, self.current_data)
                    elif cached is not None:
                        self.current_observation = str(self.current_data)
                    self.current_observation = f"{cached_tag}{self.current_observation}"
                    self.scratchpad += self.current_observation 
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['googleDistanceMatrix'] += 1
                    self.current_observation = f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.scratchpad += f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            
            
            elif action_type == 'NotebookWrite':
                try:
                    self._mask_previous_observation()
                    self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
                    self.scratchpad  +=  self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['notebook'] += 1
                    self.current_observation = f'{e}'
                    self.scratchpad += f'{e}'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            

            elif action_type == "Planner":
                # try:
                    notebook_entries = self.tools['notebook'].list_all()
                    if self.notebook_keep_days:
                        notebook_entries = _prune_notebook_entries(
                            notebook_entries,
                            self.query_item,
                            self.notebook_keep_days,
                            keep_undated=self.notebook_keep_undated,
                            summarize=self.notebook_summary,
                        )
                    notebook_payload = json.dumps(notebook_entries, ensure_ascii=True, separators=(",", ":"))
                    planner_query = _append_constraint_policy(action_arg, self.constraint_policy_text)
                    self.current_observation = str(
                        self.tools['planner'].run(
                            notebook_payload,
                            planner_query,
                            query_item=self.query_item,
                            query_index=self.query_index,
                        )
                    )
                    self._write_token_audit()
                    self.scratchpad  +=  self.current_observation
                    self.answer = self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

            else:
                self.retry_record['invalidAction'] += 1
                self.current_observation = 'Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / ' \
                                   'AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'

        if action == None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            print(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation

        self.step_n += 1

        # 

        if action_type and action_type == 'Planner' and self.retry_record['planner']==0:
            
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if self.react_name == 'gemini':
                    request = format_step(self.llm.invoke(self._build_agent_prompt(),stop=['\n']).content)
                else:
                    request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                # print(request)
                return request
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            scratchpad = self.scratchpad
            query_text = self.query_context or self.query
            query_text = _append_constraint_policy(query_text, self.constraint_policy_text)
            prompt = self.agent_prompt.format(
                query=query_text,
                scratchpad=scratchpad)
            if len(self.enc.encode(prompt)) > self.max_token_length:
                scratchpad = truncate_scratchpad(scratchpad, n_tokens=max(self.max_token_length - 1000, 500))
                self.scratchpad = scratchpad
            return self.agent_prompt.format(
                query=query_text,
                scratchpad=scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        if not self.shared_tool_cache:
            self.tool_cache = {}

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()
    
    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0

    def _mask_previous_observation(self) -> None:
        if self.current_observation:
            self.scratchpad = self.scratchpad.replace(
                self.current_observation,
                "Masked due to limited length. Make sure the data has been written in Notebook.",
            )

    def _make_cache_key(self, tool_key: str, action_arg: str) -> str:
        return f"{tool_key}:{action_arg}"

    def _get_cached_tool_result(self, tool_key: str, action_arg: str):
        if not self.token_reduction_enabled:
            return None
        return self.tool_cache.get(self._make_cache_key(tool_key, action_arg))

    def _cache_tool_result(self, tool_key: str, action_arg: str, data) -> None:
        if not self.token_reduction_enabled:
            return
        self.tool_cache[self._make_cache_key(tool_key, action_arg)] = data

    def _record_token_audit(self, tool_key: str, action_arg: str, raw_data, compressed, cached: bool) -> None:
        if not self.token_audit_enabled:
            return
        self.token_audit_entries.append(
            {
                "tool": tool_key,
                "action_arg": action_arg,
                "cache_key": self._make_cache_key(tool_key, action_arg),
                "raw_text": to_string(raw_data) if raw_data is not None else None,
                "compressed": compressed,
                "cached": cached,
            }
        )
        if cached:
            self.token_audit_cache_hits += 1
        else:
            self.token_audit_cache_misses += 1

    def _write_token_audit(self) -> None:
        if not self.token_audit_enabled:
            return
        out_root = os.getenv("TOKEN_AUDIT_OUT_DIR")
        if not out_root:
            return
        out_root_path = Path(out_root)
        if not out_root_path.is_absolute():
            out_root_path = (REPO_ROOT / out_root_path).resolve()
        set_type = os.getenv("TOKEN_AUDIT_SET_TYPE", "validation")
        index = self.query_index if self.query_index is not None else "unknown"
        out_dir = out_root_path / set_type
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder = _get_audit_encoder()

        entries = list(self.token_audit_entries)
        first_raw_by_key = {}
        first_compact_by_key = {}
        for entry in entries:
            key = entry["cache_key"]
            if entry.get("raw_text") and key not in first_raw_by_key:
                first_raw_by_key[key] = entry["raw_text"]
            if entry.get("compressed") is not None and key not in first_compact_by_key:
                first_compact_by_key[key] = entry["compressed"]

        raw_entries = []
        compact_entries = []
        for entry in entries:
            raw_text = entry.get("raw_text") or first_raw_by_key.get(entry["cache_key"])
            if raw_text is not None:
                raw_entries.append({"Short Description": f"{entry['tool']} sample", "Content": raw_text})
            if entry.get("compressed") is not None:
                compact_entries.append(
                    {"Short Description": f"{entry['tool']} sample", "Content": entry["compressed"]}
                )

        raw_unique_entries = []
        compact_unique_entries = []
        for key, raw_text in first_raw_by_key.items():
            raw_unique_entries.append({"Short Description": f"{key} sample", "Content": raw_text})
        for key, compressed in first_compact_by_key.items():
            compact_unique_entries.append({"Short Description": f"{key} sample", "Content": compressed})

        raw_tokens = _count_tokens(_serialize_compact(raw_entries), encoder) if raw_entries else None
        compressed_tokens = (
            _count_tokens(_serialize_compact(compact_entries), encoder) if compact_entries else None
        )
        raw_unique_tokens = (
            _count_tokens(_serialize_compact(raw_unique_entries), encoder)
            if raw_unique_entries
            else None
        )
        compressed_unique_tokens = (
            _count_tokens(_serialize_compact(compact_unique_entries), encoder)
            if compact_unique_entries
            else None
        )

        reduction = None
        if raw_tokens:
            reduction = (raw_tokens - (compressed_tokens or 0)) / raw_tokens

        duplicate_savings = None
        if compressed_tokens:
            duplicate_savings = (compressed_tokens - (compressed_unique_tokens or 0)) / compressed_tokens

        summary = {
            "index": index,
            "cache_hits": self.token_audit_cache_hits,
            "cache_misses": self.token_audit_cache_misses,
            "notebook_tokens": {
                "raw_tokens": raw_tokens,
                "compressed_tokens": compressed_tokens,
                "reduction": reduction,
                "compressed_duplicate_savings": duplicate_savings,
            },
            "entries": [
                {
                    "tool": entry["tool"],
                    "action_arg": entry["action_arg"],
                    "cached": entry["cached"],
                }
                for entry in entries
            ],
        }
        (out_dir / f"token_audit_{index}.json").write_text(json.dumps(summary, indent=2))


    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            
            # Avoid instantiating the planner tool twice 
            if tool_name == 'planner' and planner_model_name is not None:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])(model_name=planner_model_name)
            else:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])()
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, 'r').read().strip().split('\n')
        for unit in lines:
            city_set.append(unit)
        return city_set

### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def _get_audit_encoder():
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        return None


def _count_tokens(text: str, encoder) -> int:
    if encoder is None:
        return len(text.encode("utf-8"))
    return len(encoder.encode(text))


def _serialize_compact(data) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _parse_env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _append_constraint_policy(text: str, policy_text: str | None) -> str:
    if not policy_text:
        return text
    if "Constraint policy" in text:
        return text
    return f"{text}\n\n{policy_text}"


_DATE_PATTERN = re.compile(r"\\d{4}-\\d{2}-\\d{2}")


def _summarize_content(content) -> str:
    if isinstance(content, list):
        preview = content[:1]
        return f"[summary] {len(content)} items; first: {preview}"
    if isinstance(content, dict):
        return f"[summary] keys={list(content.keys())}"
    text = str(content)
    return f"[summary] {text[:200]}"


def _summarize_entry(entry: dict) -> dict:
    return {
        "index": entry.get("index"),
        "Short Description": entry.get("Short Description"),
        "Content": _summarize_content(entry.get("Content")),
    }


def _prune_notebook_entries(
    entries: list,
    query_item: dict | None,
    keep_days: int,
    *,
    keep_undated: bool,
    summarize: bool,
) -> list:
    if not query_item or keep_days <= 0:
        return entries
    dates = query_item.get("date") or []
    if isinstance(dates, str):
        dates = [dates]
    if not isinstance(dates, list) or not dates:
        return entries
    keep_dates = set(dates[-keep_days:])
    pruned = []
    for entry in entries:
        entry_text = _serialize_compact(entry)
        entry_dates = set(_DATE_PATTERN.findall(entry_text))
        if entry_dates & keep_dates:
            pruned.append(entry)
            continue
        if not entry_dates and keep_undated:
            pruned.append(entry)
            continue
        if summarize:
            pruned.append(_summarize_entry(entry))
    return pruned


def parse_action(string):
    cleaned = string.strip()
    cleaned = re.sub(r'^Action\s*\d*\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, cleaned)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')



def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observation_indices = [i for i, line in enumerate(lines) if line.startswith('Observation')]
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens and observation_indices:
        ind = observation_indices.pop(0)
        prefix = lines[ind].split(':')[0]
        lines[ind] = f"{prefix}: [truncated to save tokens]"
    return '\n'.join(lines)


def _maybe_parse_plan_json(text: str):
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned or cleaned[0] not in "[{":
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _maybe_parse_smt_plan(text: str, query_item: dict | None):
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("SMT solver") or cleaned.startswith("SMT planner"):
        return None
    if not query_item:
        return None
    query_json = {
        "org": query_item["org"],
        "dest": query_item["dest"],
        "days": query_item["days"],
        "visiting_city_number": query_item["visiting_city_number"],
        "date": query_item["date"],
        "people_number": query_item["people_number"],
        "local_constraint": query_item.get("local_constraint"),
        "budget": query_item["budget"],
    }
    try:
        return parse_smt_plan(cleaned, query_json)
    except Exception:
        return None


def _read_json_safely(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw)
        return obj


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def remove_observation_lines(text, step_n):
    pattern = re.compile(rf'^Observation {step_n}.*', re.MULTILINE)
    return pattern.sub('', text)

def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    if not re.match(pattern, date_str):
        raise DateError
    return True

def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True

def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")
    
    # Initialize an empty dictionary to store the results
    result = {}
    
    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment
                
        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")
        
        # Store the result with the operator included
        result[key] = (operator, value)
        
    return result

def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)

if __name__ == '__main__':

    tools_list = ["notebook","flights","attractions","accommodations","restaurants","googleDistanceMatrix","planner","cities"]
    # model_name = ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral','ChatGLM3-6B-32K'][2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()
    args.output_dir = normalize_output_dir(args.output_dir, base_dir=REPO_ROOT)
    smt_enabled = os.getenv("TWO_STAGE_SMT", "0") == "1"
    smt_only = os.getenv("TWO_STAGE_SMT_ONLY", "0") == "1"
    if "SMT_PLANNER_SET_TYPE" not in os.environ:
        os.environ["SMT_PLANNER_SET_TYPE"] = args.set_type
    if "SMT_PLANNER_OUTPUT_ROOT" not in os.environ:
        os.environ["SMT_PLANNER_OUTPUT_ROOT"] = os.path.join(args.output_dir, "smt_planner")
    if "TOKEN_AUDIT_OUT_DIR" not in os.environ:
        os.environ["TOKEN_AUDIT_OUT_DIR"] = os.path.join(args.output_dir, "token_audit")
    if "TOKEN_AUDIT_SET_TYPE" not in os.environ:
        os.environ["TOKEN_AUDIT_SET_TYPE"] = args.set_type
    smt_planner = SmtPlanner(model_name=args.model_name) if (smt_enabled or smt_only) else None
    if (
        os.environ.get("PLANNER_BACKEND", "").lower() == "smt"
        and "SMT_PLANNER_FULL_DB" not in os.environ
    ):
        os.environ["SMT_PLANNER_FULL_DB"] = "1"
    # Use reuse_cache_if_exists to avoid downloading script if data is already cached
    if args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation', download_mode='reuse_cache_if_exists')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test', download_mode='reuse_cache_if_exists')['test']
    numbers = [i for i in range(1, len(query_data_list) + 1) if i >= args.start_idx]
    if args.max_items:
        numbers = numbers[: args.max_items]
    agent = ReactAgent(None, tools=tools_list,max_steps=30,react_llm_name=args.model_name,planner_llm_name=args.model_name)
    with get_openai_callback() as cb:
        
        for number in tqdm(numbers[:]):
            query_item = query_data_list[number-1]
            query = query_item['query']
              # check if the directory exists
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
                result =  [{}]
            else:
                result = _read_json_safely(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'))
                
            profile = build_user_profile(query_item)
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/user_profile_{number}.json'), 'w') as f:
                json.dump(profile, f, indent=2)

            if smt_planner is not None and smt_only:
                smt_plan_text = smt_planner.run(
                    "",
                    query,
                    query_item=query_item,
                    query_index=number,
                )
                result[-1][f'{args.model_name}_two-stage_smt_results'] = smt_plan_text
                result[-1][f'{args.model_name}_two-stage_smt_parsed_results'] = _maybe_parse_smt_plan(
                    smt_plan_text, query_item
                )
                result[-1]['user_profile'] = profile
                with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
                    json.dump(result, f, indent=4)
                continue

            while True:
                planner_results, scratchpad, action_log  = agent.run(
                    query,
                    query_item=query_item,
                    query_index=number,
                )
                if planner_results != None:
                    break
            
            if planner_results == 'Max Token Length Exceeded.':
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad 
                result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
                action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
            else:
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad 
                result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
                result[-1]['user_profile'] = profile
                if os.getenv("PLANNER_OUTPUT_FORMAT", "").lower() == "json":
                    parsed = _maybe_parse_plan_json(planner_results)
                    if parsed is not None:
                        result[-1][f'{args.model_name}_two-stage_parsed_results'] = parsed
                if smt_planner is not None:
                    smt_plan_text = smt_planner.run(
                        "",
                        query,
                        query_item=query_item,
                        query_index=number,
                    )
                    result[-1][f'{args.model_name}_two-stage_smt_results'] = smt_plan_text
                    result[-1][f'{args.model_name}_two-stage_smt_parsed_results'] = _maybe_parse_smt_plan(
                        smt_plan_text, query_item
                    )

            # write to json file
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
                json.dump(result, f, indent=4)
        
    print(cb)
