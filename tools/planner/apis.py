import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,reflect_prompt,react_reflect_planner_agent_prompt, REFLECTION_HEADER
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.llms.base import BaseLLM
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List, Union, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
PLANNER_CONTEXT_WINDOW_DAYS = os.environ.get('PLANNER_CONTEXT_WINDOW_DAYS')
PLANNER_BACKEND = os.environ.get("PLANNER_BACKEND", "llm").lower()


def _load_smt_runner():
    from tools.hybrid_two_stage_smt import run_single_query
    return run_single_query


class SmtPlanner:
    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.output_root = Path(
            os.getenv("SMT_PLANNER_OUTPUT_ROOT", "smt_token_output/smt_planner")
        )
        repo_override = os.getenv("SMT_PLANNER_REPO")
        self.smt_repo = Path(repo_override) if repo_override else None
        self.model_version = os.getenv("SMT_PLANNER_MODEL", model_name)
        self.full_db = os.getenv("SMT_PLANNER_FULL_DB", "0") == "1"
        self.top_k_cities = int(os.getenv("SMT_PLANNER_TOP_K_CITIES", "6"))
        self.set_type = os.getenv("SMT_PLANNER_SET_TYPE", "validation")

    def run(self, _text, query, *, query_item=None, query_index=None, log_file=None) -> str:
        run_single_query = _load_smt_runner()
        smt_query = query
        if query_item and isinstance(query_item, dict) and query_item.get("query"):
            smt_query = query_item["query"]
        index = int(query_index) if query_index else 1
        query_json = None
        if query_item:
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
        result = run_single_query(
            smt_query,
            query_json,
            index=index,
            set_type=self.set_type,
            output_root=self.output_root,
            smt_repo=self.smt_repo,
            model_version=self.model_version,
            full_db=self.full_db,
            top_k_cities=self.top_k_cities,
        )
        status = result.get("status")
        plan = result.get("plan")
        error = result.get("error")
        if log_file:
            log_file.write(f"\n[SmtPlanner] status={status}\n")
        if status == "sat" and plan:
            return plan
        if status == "unsat":
            return "SMT solver returned unsat."
        if error:
            return f"SMT solver error: {error}"
        return "SMT planner did not return a plan."


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


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.backend = PLANNER_BACKEND
        self.smt_planner = None
        if self.backend == "smt":
            self.smt_planner = SmtPlanner(model_name=model_name)
            print("[Planner] SMT backend enabled.")
            return

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name in  ['mistral-7B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo")
        
        elif model_name in  ['ChatGLM3-6B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo")
            
        elif model_name in ['mixtral']:
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="YOUR/MODEL/PATH")
        elif model_name.startswith('ollama:'):
            # Use local Ollama models via LangChain's ChatOllama wrapper.
            # Example: --model_name ollama:llama3
            ollama_model = model_name.split(":", 1)[1] or "llama3"
            self.llm = ChatOllama(
                model=ollama_model,
                temperature=0,
            )
        elif model_name in ['gemini']:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required when using 'gemini' model. Please set it in your .env file.")
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        elif model_name.startswith('deepseek:') or model_name.startswith('deepseek-'):
            # DeepSeek models use OpenAI-compatible API
            deepseek_model = model_name.replace('deepseek:', '') if ':' in model_name else model_name
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is required when using DeepSeek models. Please set it in your .env file.")
            self.llm = ChatOpenAI(
                model_name=deepseek_model,
                temperature=0,
                max_tokens=4096,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com/v1"
            )
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI models. Please set it in your .env file.")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)


        # Debug logging to make routing explicit when debugging backends.
        try:
            backend_name = type(self.llm).__name__
            extra = getattr(self.llm, "model", None)
        except Exception:
            backend_name = str(type(self.llm))
            extra = None
        print(f"[Planner] LLM backend loaded: {backend_name} (model_name={model_name}, extra={extra})")

    def run(self, text, query, *, query_item=None, query_index=None, log_file=None) -> str:
        if self.backend == "smt" and self.smt_planner is not None:
            return self.smt_planner.run(
                text,
                query,
                query_item=query_item,
                query_index=query_index,
                log_file=log_file,
            )
        if log_file:
            log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        # print(self._build_agent_prompt(text, query))
        if self.model_name in ['gemini']:
            return str(self.llm.invoke(self._build_agent_prompt(text, query)).content)
        else:
            if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
                return 'Max Token Length Exceeded.'
            else:
                return self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content

    def _build_agent_prompt(self, text, query) -> str:
        if PLANNER_CONTEXT_WINDOW_DAYS:
            try:
                keep_days = int(PLANNER_CONTEXT_WINDOW_DAYS)
                text = _prune_day_sections(text, keep_days)
            except ValueError:
                pass
        return self.agent_prompt.format(
            text=text,
            query=query)


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
        summary_block = "Earlier days (summary):\n" + "\n".join(summaries) + "\n\n"

    return preamble + summary_block + kept


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        # Support both OpenAI-hosted and local Ollama models.
        if model_name.startswith('ollama:'):
            ollama_model = model_name.split(":", 1)[1] or "llama3"
            self.react_llm = ChatOllama(
                model=ollama_model,
                temperature=0,
            )
        elif model_name.startswith('deepseek:') or model_name.startswith('deepseek-'):
            # DeepSeek models use OpenAI-compatible API
            deepseek_model = model_name.replace('deepseek:', '') if ':' in model_name else model_name
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is required when using DeepSeek models. Please set it in your .env file.")
            self.react_llm = ChatOpenAI(
                model_name=deepseek_model,
                temperature=0,
                max_tokens=1024,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com/v1",
                model_kwargs={"stop": ["Action", "Thought", "Observation"]},
            )
        else:
            self.react_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": ["Action", "Thought", "Observation"]},
            )
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        if model_name.startswith('ollama:'):
            # Use the same local Ollama model for both react and reflection LLMs.
            ollama_model = model_name.split(":", 1)[1] or "llama3"
            self.react_llm = ChatOllama(
                model=ollama_model,
                temperature=0,
            )
            self.reflect_llm = ChatOllama(
                model=ollama_model,
                temperature=0,
            )
        elif model_name in ['gemini']:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required when using 'gemini' model. Please set it in your .env file.")
            self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        elif model_name.startswith('deepseek:') or model_name.startswith('deepseek-'):
            # DeepSeek models use OpenAI-compatible API
            deepseek_model = model_name.replace('deepseek:', '') if ':' in model_name else model_name
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is required when using DeepSeek models. Please set it in your .env file.")
            self.react_llm = ChatOpenAI(
                model_name=deepseek_model,
                temperature=0,
                max_tokens=1024,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com/v1",
                model_kwargs={"stop": ["Action","Thought","Observation,'\n"]}
            )
            self.reflect_llm = ChatOpenAI(
                model_name=deepseek_model,
                temperature=0,
                max_tokens=1024,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base="https://api.deepseek.com/v1",
                model_kwargs={"stop": ["Action","Thought","Observation,'\n"]}
            )
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI models. Please set it in your .env file.")
            self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
            self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
                else:
                    return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def prompt_reflection(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
                else:
                    return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad,
                            reflections = self.reflections_str)
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    
