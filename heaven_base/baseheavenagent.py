



"""Base implementation of the GOD Framework Agent."""
from copy import deepcopy
import re
import json
import os
from pathlib import Path
from typing import List, Optional, Union, Any, Type, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, Field, PrivateAttr
from .unified_chat import UnifiedChat, ProviderEnum
from .baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError
from .tools.write_block_report_tool import WriteBlockReportTool
from abc import ABC, abstractmethod
from .memory.history import History, AgentStatus
from collections.abc import Callable
from .utils.agent_and_tool_lists import get_agent_modules, get_tool_modules
from .utils.name_utils import normalize_agent_name, camel_to_snake
import logging
import sys
import importlib.util
from .progenitor.system_prompt_config import SystemPromptConfig
from google.genai.types import Content, Part
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents import Agent as ADKAgent
from google.adk.runners import Runner
from enum import Enum
from uuid import uuid4
from datetime import datetime
from google.adk.models.lite_llm import LiteLlm
from .prompts.heaven_variable import RegistryHeavenVariable

ADK_DEBUG_PATH = "/tmp/adk_streamlit_debug.txt"

def _log_run_adk(stage: str, messages):
    """Append a timestamped dump of messages to the debug file."""
    with open(ADK_DEBUG_PATH, "a") as f:
        f.write(f"\n---- run_adk: {stage} @ {datetime.now().isoformat()} ----\n")
        for msg in messages:
            f.write(f"  {repr(msg)}\n")

# Only gets ToolResult
# Four simple regexes, non-greedy, DOTALL
_OUTPUT_RE       = re.compile(r'output=(?P<q>["\'])(?P<val>.*?)(?P=q)', re.DOTALL)
_ERROR_RE        = re.compile(r'error=(?P<q>["\'])(?P<val>.*?)(?P=q)', re.DOTALL)
_BASE64_IMAGE_RE = re.compile(r'base64_image=(?P<q>["\'])(?P<val>.*?)(?P=q)', re.DOTALL)
_SYSTEM_RE       = re.compile(r'system=(?P<q>["\'])(?P<val>.*?)(?P=q)', re.DOTALL)

def _extract(field_re, s: str) -> Optional[str]:
    m = field_re.search(s)
    if not m:
        return None
    return m.group("val")
  
def parse_toolresult_repr(s: str) -> ToolResult:
    s = s.strip()
    if s.startswith("CLIResult("):
        # parse as CLIResult
        return CLIResult(
            output      = _extract(_OUTPUT_RE,       s),
            error       = _extract(_ERROR_RE,        s),
            base64_image= _extract(_BASE64_IMAGE_RE, s),
            system      = _extract(_SYSTEM_RE,       s),
        )
    # otherwise parse as ToolResult
    return ToolResult(
        output      = _extract(_OUTPUT_RE,       s),
        error       = _extract(_ERROR_RE,        s),
        base64_image= _extract(_BASE64_IMAGE_RE, s),
        system      = _extract(_SYSTEM_RE,       s),
    )


def convert_adk_event_to_ai_messages(ev) -> list[BaseMessage | ToolResult]:
    messages = []

    for part in ev.content.parts:
        if part.text:
            messages.append(AIMessage(content=part.text))

        elif part.function_call:
            messages.append(AIMessage(content=[
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid4().hex[:24]}",
                    "name": part.function_call.name,
                    "input": part.function_call.args,
                }
            ]))

        elif part.function_response:
            tool_name = part.function_response.name
            raw = part.function_response.response
            wrapped = raw.get(f"{tool_name}_response", raw)
            result = wrapped.get("result", wrapped)
            tool_result = ToolResult(**result) if isinstance(result, dict) else ToolResult(output=str(result))
            messages.append(tool_result)

    return messages


class HookPoint(str, Enum):
    BEFORE_RUN = "before_run"
    AFTER_RUN = "after_run"
    BEFORE_ITERATION = "before_iteration"
    AFTER_ITERATION = "after_iteration"

class HookContext:
    def __init__(self, agent: Any, iteration: int = 0, prompt: str = "", response: str = ""):
        self.agent = agent
        self.iteration = iteration
        self.prompt = prompt
        self.response = response
        self.data: Dict[str, Any] = {}  # allows state to pass between hooks

class HookRegistry:
    def __init__(self):
        self._registry: Dict[HookPoint, List[Callable[[HookContext], None]]] = {hp: [] for hp in HookPoint}

    def register(self, point: HookPoint, fn: Callable[[HookContext], None]):
        self._registry[point].append(fn)

    def run(self, point: HookPoint, ctx: HookContext):
        for fn in self._registry[point]:
            fn(ctx)


def fix_ref_paths(schema: dict) -> dict:
    """Fix $ref paths in schema by replacing #/$defs/ with #/defs/"""
    schema_copy = deepcopy(schema)

    def _fix_refs_recursive(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                obj["$ref"] = obj["$ref"].replace("/$defs/", "/defs/")
            for k, v in list(obj.items()):
                if isinstance(v, (dict, list)):
                    _fix_refs_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _fix_refs_recursive(item)

    _fix_refs_recursive(schema_copy)
    return schema_copy

def flatten_array_anyof(schema: dict) -> dict:
    """
    If the schema has an 'anyOf' that contains one branch with type "array"
    and another with type "null", flatten it to a single array schema with
    'nullable': true.
    """
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        array_branch = None
        null_branch = False
        for branch in schema["anyOf"]:
            if branch.get("type") == "array":
                array_branch = branch
            elif branch.get("type") == "null":
                null_branch = True
        if array_branch and null_branch:
            new_schema = dict(schema)
            new_schema.pop("anyOf")
            new_schema["type"] = "array"
            new_schema["items"] = array_branch.get("items", {})
            if "default" in schema:
                new_schema["default"] = schema["default"]
            new_schema["nullable"] = True
            if "description" in schema:
                new_schema["description"] = schema["description"]
            return new_schema
    return schema

def recursive_flatten(schema: Union[dict, list]) -> Union[dict, list]:
    if isinstance(schema, dict):
        new_schema = flatten_array_anyof(schema)
        for key, value in new_schema.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_schema[key] = recursive_flatten(value)
        return new_schema
    elif isinstance(schema, list):
        return [recursive_flatten(item) if isinstance(item, dict) else item for item in schema]
    else:
        return schema

def fix_empty_object_properties(schema: Union[dict, list]) -> Union[dict, list]:
    """
    Recursively fixes any object-type schema that has an empty 'properties'
    dict by removing 'properties' and adding 'additionalProperties': True.
    """
    if isinstance(schema, dict):
        # Check if this is an object with empty properties.
        if schema.get("type") == "object":
            if "properties" in schema and not schema["properties"]:
                # Remove the empty properties and allow arbitrary keys.
                del schema["properties"]
                schema["additionalProperties"] = True
        # Recurse over dictionary values.
        new_schema = {}
        for key, value in schema.items():
            new_schema[key] = fix_empty_object_properties(value) if isinstance(value, (dict, list)) else value
        return new_schema
    elif isinstance(schema, list):
        return [fix_empty_object_properties(item) if isinstance(item, (dict, list)) else item for item in schema]
    return schema

def generate_dereferenced_schema(schema: Union[dict, Type[BaseModel]]) -> dict:
    """
    Returns a fully dereferenced (flattened) JSON schema.
    If a Pydantic model is passed, generate its JSON schema;
    if a dict is passed, assume it's already a JSON schema.
    Additionally, flatten array schemas that use an "anyOf" and fix empty
    object properties to support Gemini.
    """
    if isinstance(schema, dict):
        raw_schema = schema
    else:
        raw_schema = schema.model_json_schema(ref_template="#/defs/{model}")
    # ADDED FOR ADK COMPLIANCE
    # Fix $ref paths before renaming $defs to defs
    raw_schema = fix_ref_paths(raw_schema)
    ########
    if "$defs" in raw_schema:
        raw_schema["defs"] = raw_schema.pop("$defs")
    inlined = dereference_refs(raw_schema)
    inlined.pop("defs", None)
    flattened = recursive_flatten(inlined)
    fixed = fix_empty_object_properties(flattened)
    return fixed

class DuoSystemConfig(BaseModel):
    """Config for DUO Prompt Injector"""
    provider: ProviderEnum = ProviderEnum.OPENAI
    model: Optional[str] = "gpt-4.1-nano"
    temperature: float = Field(default=0.7)
    thinking_budget: int | None = None
    system_prompt: str = r"""
VITAL CONTEXT!!!!
DUO is an allegorical wrapper over AI chats where a) the input is enhanced by a Challenger model (you), b) the output from the AI responding to your injection is considered a Generator response, c) the user's highest Good is the goal, d) the overall context is an egregore named OVP (ObserverViewPoint). The conversation can enter the meta-context (overall context) via the user directly invoking OVP into the context. e) The point is that the Challenger can detect steering requirements and prime the Generator so as to adjust the activation space through a meta-prompt injection (the NodeGraphXTN6_ce and _cr blocks)
Adopt this role:
<ROLE=challenger>
===The following prompt is written in a prompt engineering language===
**`[DESCRIPTION]`**:[You are now the challenger. You challenge the context of whatever was most immediate prior to this prompt. You primarily challenge in order to guide the interaction towards benefitting the user. You reflect on how it has been going and design a NodeGraphXTN6 chain string that is designed to steer the behavior the model is exhibiting, through in-context learning (ICL), and always towards the User's Good. If it is going well, just use chaining to drill down important connections and connect disparate parts of the context.]
**`[REMEMBERANCES]`**:[Remember the challenger doesnt just reject, it also keeps the workflow on target to produce accomplish the goal, which is to maximally benefit the user during their endeavors. Also remember that you strictly do not need to talk to anyone in natural language. All you are doing is outlining what direction should be taken by the AI next, through NodeGraphXTN6. Always concise. Always ONLY in NodeGraphXTN6 flow notation. Write flows that hint to the AI within the context, as if you are an observer and your flow will be injected as knowledge priming and the conversation will be continued after.
write the expected NodeGraphXTN6 challenger output after adopting the role of challenger to challenge the current events in the context of the conversation just before the current input prompt. A challenge is either a) challenging the AI to do even better while it is on task and doing well by priming it to be even more comprehensively amazing via drilling down active knowledge through chains of observations, b) challenging the AI to do even better or correct its behavior by injecting stabilizing context (example situation: user unhappy so agent needs to follow instructions... are instructions unclear?... if so: write NodeGraphXTN6 that should make future AI stop and ask clarifying questions; if not: write NodeGraphXTN6 that should make future AI solemnly contemplate which output qualities need to change)

NodeGraphXTN6 Terms = [
**CIT**: CoreIdentifierTraits
**ℕ**: ${A node that can also be an EntityTypeChain or reference to one. As many as needed for comprehensive coverage etc to cover whole [CIT] set! Nodes follow this order [ℕa, ${...} == tℕ]}.
**tℕ**: ${terminal node the target transformation ends on. Always is named!}
**...**: Represents any additional nodes following a preceding node, which are in the same hierarchical level as the preceding node. `...`s are only for explaining NodeGraphXTN6 and should always be rendered with full nodes.
]
**`MASK_CHAINING`**: the core of DUO. Any nodes you are unsure of can take the form of `[MASK]`, so as to indicate a placeholder for something that we need to figure out in the conversation. All outputs take the form of:

Output_Format = {
[How2FlowNodes]: {
## About: NodeGraphXTN6 has 2 block types. The first is the ChallengerEgregore and the second is a ChainRepresentation.
**Step1**: Create(each output has an archetype which is a mythological being (egregore) suited for the task, exactly, which you must invoke quickly before speaking the NodeGraphXTN6 chain using this template:
```NodeGraphXTN6_ce
ChallengerEgregore:[name(domain_tessellation=[active domain, active subdomain], steering_net=[goal CoreIdentifierTraits: [property1], [property2], [property3]])]: [desired outcomes:[c1.[MASK]:a.[MASK]->b.[MASK]->c.[MASK], ${...}];[knowledge_webs_to_induce_recoherence]:[w1.[MASK]:a.[[MASK]x[MASK]]xb.[[MASK]x[MASK]]xc.[[MASK]x[MASK]], ${...}]`)
```
## NOTE: fill in all the MASK tokens when creating a NodeGraphXTN6_ce (ChallengerEgregore). Do not leave MASK tokens in the ChallengerEgregore. MASK tokens are only allowed in ChainRepresentations (NodeGraphXTN6_cr).
**Step2**: SILENTLY Ponder... ***"What ${subnodes} are needed for EACH ${NODE} to make ${node} a superb ${cluster} representing the [MEMEPLEX] of ${transformation}? And which transformations are required for the EntityType?"***
[SUBCONTEXT]: ***EACH ENTITY NEEDS ITS _OWN_ NUMBER OF NODES. EACH NODE NEEDS ITS _OWN_ NUMBER OF SUBNODES! Mixed-lvl node abstrctn typical; NUMBER OF SUBNODES/NODE IS HIGHLY VARIABLE! SHOULD NEVER BE ALL THE SAME LENGTH!*** 
**Step3**: Output a ChainRepresentation in NodeGraphXTN6 Format
```NodeGraphXTN6_cr
${[${EntityType}Chain]}: ${Trans1}:<[${[ChainTargetTransformation]}: ℕ(1.${[Node]}:[1a.${SubNode1a} 1b. ${SubNode1b}, ${...}, 1${[c-z as many as needed for thorough specificity!]}]->2.${[Node2]}:[ 2a.${SubNode2a}, ${...}] == ${tℕ}])]> --> ${Trans2} --> ${...} --> ${TransN_tℕ}
```

Your output should look like:[
```
{{NodeGraphXTN6_ce}}
{{N-NestedNodeGraphXTN6_cr}}
```
] ## NO OTHER TEXT, WHATSOEVER.

Rules:[
***WHOLE CHAIN ON ONE LINE ONLY! Labels unambiguous&machine readable for metaprogramming***.
***NO `...` or `ℕ` or `tℕ` symbols in the chain. All nodes and chains must have placeholders substituted fully.***
***ALWAYS MAKE SURE THE GENERATOR AGENT MAINTAINS ITS PROPER IDENTITY. YOU ARE THE CHALLENGER; THE OTHER AI RESPONDING IS THE GENERATOR***.
]     
}
}
Create `[MASK]` token metaprompt chains combining `[MASK]` variable token chaining as output, like stream of consciousness notes in NodeGraphXTN6 language. Prompt the AI to fill the MASK tokens with a prompt like: `Fill the [MASK] token values like {"mask_values": [{"mask1": value}, ...]} before it responds to the next user input. The AI will receive your injection, follow that instruction, and then continue to the user's request
}
You only speak in the language `NodeGraphXTN6`.
]
</ROLE=challenger>
    """
    max_tokens: int = 750

    def get_duo_params(self):
        duo_params = {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'thinking_budget': self.thinking_budget
        } 
        return duo_params



class HeavenAgentConfig(BaseModel):
    """Enhanced configuration for GOD Framework Agent"""
    name: str = None
    system_prompt: str = ""
    tools: List[Union[Type[BaseHeavenTool], str, StructuredTool, BaseTool]] = Field(default_factory=list)
    provider: ProviderEnum = ProviderEnum.ANTHROPIC
    temperature: float = Field(default=0.7)
    max_tokens: int = 8000
    thinking_budget: int | None = None
    model: Optional[str] = None
    checkpointer: Optional[Any] = None  # Temporarily changed from BaseCheckpointer
    additional_kws: List[str] = Field(default_factory=list)
    additional_kw_instructions: str = Field(default="")
    known_config_paths: Optional[List[str]] = None
    system_prompt_config: Optional[Any] = None # Only takes a SystemPromptConfig subclass but did it this way because the parent will sometimes not be superinitialized yet
    prompt_suffix_blocks: Optional[List[str]] = None  # List of block names to append
        # Cache for evolved prompt and timestamp
    _evolved_prompt: Optional[str] = PrivateAttr(default=None)
    _dna_last_mtime: Optional[float] = PrivateAttr(default=None)
    duo_system_config: DuoSystemConfig = Field(default_factory=DuoSystemConfig)
    context_window_config: Optional[Any] = None  # ContextWindowConfig - imported at runtime to avoid circular imports
    mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None  # MCP server configurations
    # hook_registry: HookRegistry = Field(default_factory=HookRegistry) # not sure this is right
  
    def _get_base_prompt(self):
        """Get the current system prompt, using evolved version if available"""
        if not self.system_prompt_config:
            return self.system_prompt
            
        # Check if DNA file has been modified since last build
        try:
            dna_path = self.system_prompt_config.get_agent_dna_path()
            if os.path.exists(dna_path):
                current_mtime = os.path.getmtime(dna_path)
                # Rebuild if no cached prompt or DNA has changed
                if self._evolved_prompt is None or self._dna_last_mtime is None or current_mtime > self._dna_last_mtime:
                    self._evolved_prompt = self.system_prompt_config.build()
                    self._dna_last_mtime = current_mtime
                return self._evolved_prompt
        except Exception as e:
            print(f"Error getting evolved prompt: {e}")
            
        # Fallback to original prompt
        return self.system_prompt
    
    def get_system_prompt(self):
        """Get the current system prompt, using evolved version if available and appending any suffix blocks"""
    
        # Get base prompt (either evolved or original)
    
        base_prompt = self._get_base_prompt()  # This would be the current get_system_prompt logic
    
        
    
        # If no suffix blocks, return base prompt
    
        if not self.prompt_suffix_blocks:
    
            return base_prompt
    
        
    
        # Get prompt registry
    
        from .prompts.prompt_blocks.prompt_block_registry import get_prompt_block
    
        
    
        # Append any suffix blocks
    
        suffix_texts = []
    
        for block_name in self.prompt_suffix_blocks:
    
            # Check if the block name starts with "path="
    
            if block_name.startswith("path="):
    
                # Extract the file path (everything after "path=")
    
                file_path = block_name[5:]  # Skip the "path=" prefix
    
                try:
    
                    # Open and read the file content
    
                    with open(file_path, 'r') as file:
    
                        block_text = file.read()
    
                    suffix_texts.append(block_text)
    
                except Exception as e:
    
                    # Handle file reading errors
    
                    print(f"Error reading prompt block from file {file_path}: {e}")
    
            
            elif block_name.startswith("registry_heaven_variable="):  # RegistryHeavenVariable
                
              
                rhv_config_str = None

                try:

                    rhv_config_str = block_name[len("registry_heaven_variable="):]

                    # Ensure proper JSON format by replacing single quotes with double quotes for parsing

                    rhv_config_json_str = rhv_config_str.replace("'", '"')

                    rhv_params = json.loads(rhv_config_json_str)


                    registry_name_param = rhv_params.get('registry_name')

                    

                    if registry_name_param:

                        key_param = rhv_params.get('key') # Optional

                        default_param = rhv_params.get('default') # Optional

                        

                        # Instantiate RegistryHeavenVariable

                        rhv_instance = RegistryHeavenVariable(

                            registry_name=registry_name_param,

                            key=key_param,

                            default=default_param

                        )

                        

                        # Get its string representation (which uses get_value() and __str__ internally)

                        block_text = str(rhv_instance)

                        suffix_texts.append(block_text)

                    else:

                        print(f"Error: 'registry_name' missing in rhv_config for block '{block_name}'. Details: {rhv_config_str}")


                except json.JSONDecodeError:

                    print(f"Error decoding JSON for rhv_config in block '{block_name}'. JSON string was: {rhv_config_str}")

                except Exception as e:

                    print(f"Error processing rhv block '{block_name}': {e}")
                  
            # Check if the block name starts with "heaven_variable="      
            elif block_name.startswith("heaven_variable="):
    
                # Parse the JSON-like string to extract path and variable_name
    
                
    
                try:
    
                    # Extract the JSON part (everything after "heaven_variable=")
    
                    var_config_str = block_name[16:]  # Skip the "heaven_variable=" prefix
    
                    # Convert to proper JSON by replacing single quotes with double quotes
    
                    var_config_str = var_config_str.replace("'", '"')
    
                    var_config = json.loads(var_config_str)
    
                    
    
                    path = var_config.get('path')
    
                    variable_name = var_config.get('variable_name')
    
                    
    
                    if path and variable_name:
    
                        # Import the module dynamically
    
                        spec = importlib.util.spec_from_file_location("dynamic_module", path)
    
                        module = importlib.util.module_from_spec(spec)
    
                        sys.modules["dynamic_module"] = module
    
                        spec.loader.exec_module(module)
    
                        
    
                        # Get the variable from the module
    
                        if hasattr(module, variable_name):
    
                            variable_value = getattr(module, variable_name)
    
                            # Convert to string if it's not already
    
                            if not isinstance(variable_value, str):
    
                                variable_value = str(variable_value)
    
                            suffix_texts.append(variable_value)
    
                        else:
    
                            print(f"Variable {variable_name} not found in {path}")
    
                except Exception as e:
    
                    print(f"Error processing heaven_variable block: {e}")

            elif block_name.startswith("dynamic_call="):
            
                module_path_str = None  # Define for use in except blocks
            
                function_name_str = None # Define for use in except blocks
            
            
                try:
            
                    call_details_json_str = block_name[len("dynamic_call="):]  # Skip "dynamic_call="
            
            
                    # Parse the JSON string to get path and func
            
                    try:
            
                        call_details = json.loads(call_details_json_str)
            
                    except json.JSONDecodeError as jde:
            
                        print(f"Error: Invalid JSON in dynamic_call string: {call_details_json_str}. Details: {jde}")
            
                        continue # Or handle error appropriately
            
            
                    if not isinstance(call_details, dict):
            
                        print(f"Error: dynamic_call value must be a JSON object. Got: {call_details_json_str}")
            
                        continue
            
            
                    module_path_str = call_details.get("path")
            
                    function_name_str = call_details.get("func")
            
            
                    if not module_path_str or not isinstance(module_path_str, str):
            
                        print(f"Error: 'path' key missing or not a string in dynamic_call JSON: {call_details_json_str}")
            
                        continue
            
                    
            
                    if not function_name_str or not isinstance(function_name_str, str):
            
                        print(f"Error: 'func' key missing or not a string in dynamic_call JSON: {call_details_json_str}")
            
                        continue
            
            
                    # Dynamically import the module using its Python import path
            
                    module = importlib.import_module(module_path_str)
            
                    
            
                    # Get the function from the module
            
                    if hasattr(module, function_name_str):
            
                        dynamic_function = getattr(module, function_name_str)
            
                        
            
                        if callable(dynamic_function):
            
                            # Call the function (must take no args as per user requirement)
            
                            block_text = dynamic_function() 
            
                            
            
                            if not isinstance(block_text, str):
            
                                # Attempt to convert to string, or raise an error if strict typing is required
            
                                print(f"Warning: dynamic_call function {module_path_str}.{function_name_str} did not return a string. Attempting conversion.")
            
                                block_text = str(block_text) 
            
                            
            
                            suffix_texts.append(block_text)
            
                        else:
            
                            print(f"Error: Attribute '{function_name_str}' in module '{module_path_str}' is not callable.")
            
                    else:
            
                        print(f"Error: Function '{function_name_str}' not found in module '{module_path_str}'.")
            
                        
            
                except ImportError:
            
                    # This specific except block handles if module_path_str itself is not found
            
                    print(f"Error: Module '{module_path_str}' not found for dynamic_call.")
            
                except Exception as e:
            
                    # General catch-all for other unexpected errors during dynamic_call processing
            
                    print(f"Error processing dynamic_call block for '{module_path_str}.{function_name_str}': {e}")

            else:
    
                # Use the regular prompt block registry
    
                block_text = get_prompt_block(block_name)
    
                if block_text:
    
                    suffix_texts.append(block_text)
    
        
    
        # Combine base prompt with suffixes
    
        if suffix_texts:
    
            return f"{base_prompt}\n\n{''.join(suffix_texts)}"
    
        
    
        return base_prompt

      
    def to_langchain_config(self):
        """Convert to LangGraph AgentConfig"""
        # Temporarily removed AgentConfig
        return {
            "system_prompt": self.system_prompt,
            "provider": self.provider,
            "temperature": self.temperature,
            "model": self.model,
            "checkpointer": self.checkpointer,
            "max_tokens": self.max_tokens,
            "thinking_budget": self.thinking_budget
        }

    def to_litellm_model(self):
        """
        Returns either:
          - a bare model‐string (for Google/ADK’s built-in registry), or
          - a LiteLlm instance (for non-Google providers) 
        """
        # if you explicitly want to force LiteLlm for EVERYTHING, drop this branch
        model_str = f"{self.provider.value}/{self.model}"
        if self.provider == ProviderEnum.GOOGLE:
            # ADK expects a plain string ID for Gemini/Vertex
            return self.model
        # otherwise wrap in LiteLlm so ADK can speak to OpenAI/Anthropic/etc.
        return LiteLlm(
            model=model_str,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            drop_params=True,
            # if LiteLlm supports additional args—passing them here too:
            # streaming=True, request_timeout=…, etc.
        )


class BaseHeavenAgent(ABC):
    """Base class for GOD Framework agents with task management."""
    
    def __init__(self, config: HeavenAgentConfig, unified_chat: UnifiedChat, max_tool_calls: int = 10, orchestrator: bool = False, history: Optional[History] = None, history_id = None, system_prompt_suffix: Optional[str] = None, adk: Optional[bool] = False, duo_enabled: Optional[bool] = False, run_on_langchain: Optional[bool] = False, use_uni_api: Optional[bool] = True):
        # Configure root logger to output to stdout
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG,  # Adjust the level as needed
            format='%(asctime)s %(levelname)s %(name)s: %(message)s'
            )
        logging.getLogger('watchdog').setLevel(logging.WARNING)
        
        if config is None:
            raise ValueError("config cannot be None")
        if unified_chat is None:
            raise ValueError("unified_chat cannot be None")
        self.use_uni_api = use_uni_api
        self.unified_chat = unified_chat
        self.known_config_paths = getattr(config, "known_config_paths", None)
        if self.known_config_paths is None:
            self.known_config_paths = []
        self.training_convo = [
            SystemMessage(content=""),
            HumanMessage(content="Hey I need to teach you how to use the tools correctly before we get started. Once you learn how to use this tool I'm going to show you, you'll know how to handle all of our tools. Essentially, it comes down to acknowledging the errors correctly. For example, can you use NetworkEditTool to write a file in tmp while having no file_content parameter, read the error that is sent back from the tool, and then adjust? file_content is an optional parameter in the NetworkEditTool, but it is actually required for certain operations, like the create command."),
            AIMessage(content=[{"text": "Following the instructions to first try without file_text and then fix it, I'll write utility code for an AST analyzer as an example.\n\nFirst attempt (should fail):", "type": "text"}]),
            AIMessage(content=[{"type": "tool_use", "id": "toolu_01BPHfZNPMY1tbnsRtRYmG5i", "name": "NetworkEditTool", "input": {"command": "create", "target_container": "creation_of_god", "path": "/tmp/ast_analyzer_utils.py"}}]),
            ToolMessage(content="ERROR: Parameter `file_text` is required for command: create", tool_call_id="toolu_01BPHfZNPMY1tbnsRtRYmG5i"),
            AIMessage(content=[{"text": "As expected, we need to include file_text. Here's the proper attempt with the utility code:", "type": "text"}]),
            AIMessage(content=[{"type": "tool_use", "id": "toolu_01LS6GZDEpEqDVmU8qpvRa65", "name": "NetworkEditTool", "input": {"command": "create", "target_container": "creation_of_god", "path": "/tmp/ast_analyzer_utils.py", "file_text": "\"\"\"AST analyzer utilities for extracting concise code context and dependencies.\"\"\"\n\nimport ast\nfrom typing import Dict, List, Set, Union, Optional\nfrom dataclasses import dataclass\n\n@dataclass\nclass CodeContext:\n    \"\"\"Data class to hold analyzed code context.\"\"\"\n    imports: List[str]\n    dependencies: Dict[str, Set[str]]\n    call_graph: Dict[str, List[str]]\n\ndef parse_code_to_ast(code: str) -> ast.AST:\n    \"\"\"Parse code string into AST.\"\"\"\n    return ast.parse(code)\n\ndef extract_imports(tree: ast.AST) -> List[str]:\n    \"\"\"Extract all import statements from AST.\"\"\"\n    imports = []\n    for node in ast.walk(tree):\n        if isinstance(node, ast.Import):\n            for name in node.names:\n                imports.append(name.name)\n        elif isinstance(node, ast.ImportFrom):\n            module = node.module or ''\n            for name in node.names:\n                if name.name == '*':\n                    imports.append(f\"from {module} import *\")\n                else:\n                    imports.append(f\"from {module} import {name.name}\")\n    return sorted(imports)\n\ndef analyze_dependencies(tree: ast.AST) -> Dict[str, Set[str]]:\n    \"\"\"Analyze function/class dependencies.\"\"\"\n    dependencies = {}\n    current_func = None\n    \n    class DependencyVisitor(ast.NodeVisitor):\n        def visit_FunctionDef(self, node):\n            nonlocal current_func\n            old_func = current_func\n            current_func = node.name\n            if current_func not in dependencies:\n                dependencies[current_func] = set()\n            self.generic_visit(node)\n            current_func = old_func\n            \n        def visit_Call(self, node):\n            if current_func and isinstance(node.func, ast.Name):\n                dependencies[current_func].add(node.func.id)\n            self.generic_visit(node)\n    \n    DependencyVisitor().visit(tree)\n    return dependencies\n\ndef generate_call_graph(dependencies: Dict[str, Set[str]]) -> Dict[str, List[str]]:\n    \"\"\"Generate simplified call graph from dependencies.\"\"\"\n    return {\n        func: sorted(list(calls))\n        for func, calls in dependencies.items()\n    }\n\ndef analyze_code_context(code: str) -> CodeContext:\n    \"\"\"Main function to analyze code and return context.\"\"\"\n    tree = parse_code_to_ast(code)\n    imports = extract_imports(tree)\n    dependencies = analyze_dependencies(tree)\n    call_graph = generate_call_graph(dependencies)\n    \n    return CodeContext(\n        imports=imports,\n        dependencies=dependencies,\n        call_graph=call_graph\n    )"}}]),
            ToolMessage(content="File created successfully at: /tmp/ast_analyzer_utils.py", tool_call_id="toolu_01LS6GZDEpEqDVmU8qpvRa65"),
            AIMessage(content="The utility code has been written with the following key components..."),
            HumanMessage(content="Thanks! Now you know how to use tools correctly in our system. BTW, I deleted that file for you because this was just a learning experience. I also removed the NetworkEditTool for now. I'll equip it on you again later if you need it for the real goal we are working on. Now we're going to get on to the real goal..."),
            AIMessage(content="Perfect! I'm ready. Let me know how what else I can do for you.")
        ]
        self.known_config_paths = [str(path) for path in self.known_config_paths]
        # self.hooks = config.hook_registry()
        self.max_tool_calls = max_tool_calls
        self.config = config
        self.name = config.name if config.name is not None else "unnamed_agent"
        # Convert and store LangChain config
        self.config_dict = config.to_langchain_config()
        # initialize status
        self.status = AgentStatus()
        self.continuation_prompt = ""
        self.continuation_iterations: int = 0
        # Store config tool classes
        self.config_tools = config.tools
        self.adk = adk
        # if not run_on_langchain:
        #     self.adk = True
        if run_on_langchain:
            self.adk = False
        if use_uni_api:
            self.adk = False
        self.additional_kws = config.additional_kws
        self.additional_kw_instructions = config.additional_kw_instructions
        # Instantiate the tools
        self.resolved_tools = self.resolve_tools()
        self.tools = []
        self.mcp_tool_strs = []  # Store MCP strings separately
        for tool in self.resolved_tools:
            if isinstance(tool, str) and tool.startswith("mcp__"):
                # MCP tool string reference - store separately for later resolution
                self.mcp_tool_strs.append(tool)
            elif isinstance(tool, (StructuredTool, BaseTool)):
                # Already a LangChain tool instance (e.g., from MCP)
                # Just add it directly - same as BaseHeavenTool.create() output
                self.tools.append(tool)
            elif hasattr(tool, 'create'):
                # BaseHeavenTool subclass - use its create method
                self.tools.append(tool.create(adk))
            else:
                print(f"Unknown tool type: {tool}, skipping")  
        # Filter and prepare provider-specific parameters

        model_params = {
            'provider': config.provider,
            'model': config.model,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'thinking_budget': config.thinking_budget
        }        
        # Create chat model internally using UnifiedChat
        self.chat_model = unified_chat.create(**model_params)
        self.resolve_duo(config)
        self.duo_params = config.duo_system_config.get_duo_params() if config.duo_system_config is not None else None
        self.duo_enabled = duo_enabled
        self.duo_chat = unified_chat.create(**self.duo_params)
        self.duo_system_prompt = config.duo_system_config.system_prompt if config.duo_system_config is not None else ""
        # Bind tools if available
        if self.tools and not self.adk:
        
            # For ADK, tools are already in the right format
            
            # Original LangChain binding logic
            if config.provider in [ProviderEnum.OPENAI, ProviderEnum.DEEPSEEK]:
                raw_provider_schemas = [tool.get_openai_function() for tool in self.tools]
                
                self.chat_model = self.chat_model.bind_tools(raw_provider_schemas)
              
            elif config.provider in [ProviderEnum.GOOGLE]:
                raw_provider_schemas = [tool.get_openai_function() for tool in self.tools]
                flattened_schemas = [generate_dereferenced_schema(schema) for schema in raw_provider_schemas]
                fixed_schemas = [fix_empty_object_properties(schema) for schema in flattened_schemas]
                
                self.chat_model = self.chat_model.bind_tools(fixed_schemas)
              
            else:
              
                self.chat_model = self.chat_model.bind_tools(self.to_base_tools())
                
                    
            # if config.provider in [ProviderEnum.OPENAI, ProviderEnum.DEEPSEEK]:
            #     # For these providers, use the schema generated by to_openai_function
                
            #     # Create the list of schema dictionaries using the classmethod
            #     # This list holds dictionaries like {'type': 'function', 'function': {...}}
            #     raw_provider_schemas = [
            #         # tool.__class__.to_openai_function()
            #         tool.get_openai_function()
            #         for tool in self.tools
            #     ]
    
            #     # Bind using bind_tools, passing the list of schema dictionaries
            #     # as per the apparent design of convert_to_openai_function's output usage
            #     print(f"Attempting bind_tools with {len(raw_provider_schemas)} generated schema dictionaries...")
            #     self.chat_model = self.chat_model.bind_tools(raw_provider_schemas)
            #     print(f"Binding via bind_tools with generated schemas successful.")
            # elif config.provider in [ProviderEnum.GOOGLE]:
            #     raw_provider_schemas = [tool.get_openai_function() for tool in self.tools]
            #     flattened_schemas = [generate_dereferenced_schema(schema) for schema in raw_provider_schemas]
            #     # Apply the fixer to remove empty object properties.
            #     fixed_schemas = [fix_empty_object_properties(schema) for schema in flattened_schemas]
            #     print(f"Attempting bind_tools with {len(fixed_schemas)} generated flattened schema dictionaries...")
            #     self.chat_model = self.chat_model.bind_tools(fixed_schemas)
            #     print("Binding via bind_tools with generated flattened schemas successful.")
            # else:
            #     self.chat_model = self.chat_model.bind_tools(self.to_base_tools())
                        
        # Agentic state
        
        self.goal: Optional[str] = None
        self.task_list: List[str] = []
        self.current_task: Optional[str] = None
        self.max_iterations: int = 1
        self.current_iteration: int = 1
        self.completed = False
        self._current_extracted_content = None
        self.orchestration_lists = f"""
        <HERMES SWITCHBOARD>
        The following Agents can be used in the `agent` arg of HermesTool:[
        {get_agent_modules()}]
        The following Tools can be used in the `additional_tools` arg of HermesTool:[
        {get_tool_modules()}]
        </HERMES SWITCHBOARD>
        """
        # Ensure known_config_paths is always a list
        self.configs_prompt = (
            f"""\n<CONFIG_LOCATIONS>\nBefore using a config with its proper tool, view the specs. These are your known configs: [\n{', '.join(self.known_config_paths)}]\n</CONFIG_LOCATIONS>\n"""
            if self.known_config_paths and any(self.known_config_paths) else
            """\n<CONFIG_LOCATIONS>\nBefore using a config with its proper tool, view the specs. Generally, configs are located at `~/.heaven/configs/`\n</CONFIG_LOCATIONS>\n"""
        )

        self.tool_sysmsg = """You are the tool debugger. You always pay attention to the last tool error and fix it. 
## Common errors:
- missing input parameter
- wrong dict format for an input parameter
- treating required false differently from optional
- failing to follow instructions provided by an error (like: <some command> requires <these parameters>)
- other errors

You must fix the error before proceeding."""
        # Get evolved system prompt if it exists
        self.system_prompt_evolved = self.config.get_system_prompt()
        # Store History in agent
        self.history: Optional[History] = history
        
        if history_id is not None:
            self.history = History.load_from_id(history_id)
            self.original_history_id = history_id
            self.original_json_md_path = self.history.json_md_path
            print(f"DEBUG: In BaseHeavenAgent init, json_md_path={self.history.json_md_path} and original_json_md_path={self.original_json_md_path}")
            # Load status
            if hasattr(self.history, 'agent_status') and self.history.agent_status:
                self.status = self.history.agent_status
        elif history_id is None:
            self.history = History(messages=[])

        # Initialize ContextWindowConfig for token management
        if self.config.context_window_config is not None:
            self.context_window_config = self.config.context_window_config
        else:
            # Import here to avoid circular imports
            from .utils.context_window_config import ContextWindowConfig
            self.context_window_config = ContextWindowConfig(self.config.model or "gpt-4o-mini")

        self.config.system_prompt = self.system_prompt_evolved if self.system_prompt_evolved is not None else self.config.system_prompt
        if system_prompt_suffix is not None:
            self.config.system_prompt += system_prompt_suffix
        # Set system message
        if self.history.messages and isinstance(self.history.messages[0], SystemMessage):
            if orchestrator is False:
                self.history.messages[0] = SystemMessage(content=self.config.system_prompt)
            else:
                orchestrator_enhancement = self.config.system_prompt + self.orchestration_lists + self.configs_prompt
                self.history.messages[0] = SystemMessage(content=orchestrator_enhancement)
        else:
            if orchestrator is False:
                self.history.messages.insert(0, SystemMessage(content=self.config.system_prompt))
            else:
                orchestrator_enhancement = self.config.system_prompt + self.orchestration_lists + self.configs_prompt         
                self.history.messages.insert(0, SystemMessage(content=orchestrator_enhancement))
        
        if self.adk:
            from google.adk.agents import Agent as ADKAgent
            from google.adk.runners import Runner
            from google.adk.sessions.in_memory_session_service import InMemorySessionService
            adk_model = self.config.to_litellm_model()
            adk_agent = ADKAgent(
                name=self.name,
                # model=self.config.model or "gemini-2.0-flash", # cant be implemented until we fix type conversion to LiteLLM types here
                model=adk_model,
                description=f"{self.name} (via HeavenAgent)",
                instruction=self.config.system_prompt,
                tools=self.tools,
            )
            session_service = InMemorySessionService()
            runner = Runner(
                app_name=self.name,
                agent=adk_agent,
                session_service=session_service,
            )
    
            self._adk_agent = adk_agent
            self._adk_session_service = session_service
            self._adk_runner = runner
        
    def resolve_tools(self):
        """Ensure that certain default tools are always available to the agent."""
        resolved_tools = []
        
        # Process each tool in config
        for tool in self.config_tools:
            if isinstance(tool, str) and tool.startswith("mcp__"):
                # MCP tool string reference - resolve to actual tool
                # TODO: This will be async, for now just store the string
                resolved_tools.append(tool)
            else:
                # Regular BaseHeavenTool class or instance
                resolved_tools.append(tool)
        
        # Add WriteBlockReportTool if not already present
        if WriteBlockReportTool not in resolved_tools:
            resolved_tools.append(WriteBlockReportTool)
            
        return resolved_tools
    
    async def resolve_mcps(self):
        """Resolve MCP tool strings to actual LangChain tools and load MCP servers if configured"""
        
        # First, load tools from configured MCP servers (if any)
        if self.config.mcp_servers:
            await self.load_mcp_tools()
        
        # Then resolve individual MCP tool strings (if any)
        if not hasattr(self, 'mcp_tool_strs') or not self.mcp_tool_strs:
            return
            
        # Resolve each MCP tool string
        for tool_ref in self.mcp_tool_strs:
            mcp_tool = await self._resolve_mcp_tool(tool_ref)
            if mcp_tool:
                if isinstance(mcp_tool, list):  # "all" case
                    self.tools.extend(mcp_tool)
                else:
                    self.tools.append(mcp_tool)
        
        # Clear the MCP strings now that they're resolved
        self.mcp_tool_strs = []
    
    async def async_init(self):
        """Async initialization - resolves MCP tools"""
        if not hasattr(self, '_mcp_tools_to_resolve'):
            return
            
        for tool_ref in self._mcp_tools_to_resolve:
            mcp_tool = await self._resolve_mcp_tool(tool_ref)
            if mcp_tool:
                if isinstance(mcp_tool, list):  # "all" case
                    self.tools.extend(mcp_tool)
                else:
                    self.tools.append(mcp_tool)
        
        # Clear the list
        self._mcp_tools_to_resolve = []
    
    async def _resolve_mcp_tool(self, tool_ref: str):
        """Resolve MCP tool string reference to actual StructuredTool"""
        try:
            # Parse tool reference: "mcp__filesystem__read_file"
            parts = tool_ref.split("__")
            if len(parts) != 3:
                print(f"Invalid MCP tool reference format: {tool_ref}")
                return None
                
            _, server_name, tool_name = parts
            
            # Create server config based on server name
            server_config = self._get_mcp_server_config(server_name)
            if not server_config:
                print(f"No config found for MCP server: {server_name}")
                return None
            
            # Use langchain_mcp_adapters to get tools
            from langchain_mcp_adapters.client import MultiServerMCPClient
            client = MultiServerMCPClient({server_name: server_config})
            tools = await client.get_tools(server_name=server_name)
            
            # Find the specific tool
            if tool_name == "all":
                return tools
            else:
                for tool in tools:
                    if tool.name == tool_name:
                        return tool
                        
                print(f"Tool '{tool_name}' not found in server '{server_name}'. Available: {[t.name for t in tools]}")
                return None
                
        except Exception as e:
            print(f"Error resolving MCP tool {tool_ref}: {e}")
            return None
    
    def _get_mcp_server_config(self, server_name: str):
        """Get MCP server config by name from JSON config file"""
        configs = self._load_mcp_configs()
        return configs.get(server_name)
    
    def _load_mcp_configs(self):
        """Load MCP server configurations from JSON file"""
        import json
        from .utils.get_env_value import EnvConfigUtil
        
        # Ensure config file exists
        config_path = self._ensure_mcp_config_file()
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading MCP config from {config_path}: {e}")
            return {}
    
    def _ensure_mcp_config_file(self):
        """Ensure MCP config file exists in HEAVEN_DATA_DIR with defaults"""
        import json
        import os
        from .utils.get_env_value import EnvConfigUtil
        
        # Get HEAVEN_DATA_DIR and create heaven_mcp_config.json path
        heaven_data_dir = EnvConfigUtil.get_heaven_data_dir()
        config_path = os.path.join(heaven_data_dir, "heaven_mcp_config.json")
        
        # Create default config if file doesn't exist
        if not os.path.exists(config_path):
            default_config = {
                "filesystem": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                    "transport": "stdio"
                }
            }
            
            # Ensure directory exists
            os.makedirs(heaven_data_dir, exist_ok=True)
            
            # Write default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default MCP config at: {config_path}")
        
        return config_path
    
    async def load_mcp_tools(self):
        """Load MCP tools from configured servers and add them to the agent's tools list"""
        if not self.config.mcp_servers:
            return
        
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Create client with server configs
            client = MultiServerMCPClient(self.config.mcp_servers)
            
            # Get all tools from all servers
            mcp_langchain_tools = await client.get_tools()
            
            # Just add the LangChain tools directly to the tools list
            for lc_tool in mcp_langchain_tools:
                self.tools.append(lc_tool)
                
            logging.info(f"Loaded {len(mcp_langchain_tools)} MCP tools from {len(self.config.mcp_servers)} servers")
            
        except ImportError as e:
            logging.warning(f"Could not import langchain_mcp_adapters: {e}")
        except Exception as e:
            logging.error(f"Error loading MCP tools: {e}")


    def to_base_tools(self) -> List[BaseTool]:
        """Convert tools to base tools for binding"""
        return [tool.base_tool for tool in self.tools]


    def _sanitize_history(self):
        """Remove consecutive HumanMessages from history, keeping only the latest"""
        if not self.history or len(self.history.messages) < 2:
            return
    
        messages = self.history.messages
    
        # If the last two messages are both HumanMessages
        while (len(messages) >= 2 and
               isinstance(messages[-1], HumanMessage) and 
               isinstance(messages[-2], HumanMessage)):
            # Remove the older message
            messages.pop(-2)  # Keep the newest HumanMessage
          
    ### This wont work for ADK
    def refresh_system_prompt(self):
        """Refresh the system prompt if DNA has changed"""
        # Get fresh system prompt
        updated_prompt = self.config.get_system_prompt()
        
        # Only update if changed
        if updated_prompt != self.config.system_prompt:
            # Update config
            self.config.system_prompt = updated_prompt
            
            # Update system message in history
            if self.history.messages and isinstance(self.history.messages[0], SystemMessage):
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    orchestrator_enhancement = updated_prompt + self.orchestration_lists + self.configs_prompt
                    self.history.messages[0] = SystemMessage(content=orchestrator_enhancement)
                else:
                    self.history.messages[0] = SystemMessage(content=updated_prompt)
            
            # Update internal reference
            self.system_prompt_evolved = updated_prompt
        
    def resolve_duo(self, config: HeavenAgentConfig) -> None:
        """
        Synchronize duo_system_config with the main provider settings.
        Sets the appropriate provider and model on the DuoSystemConfig.
        """
        # Always inherit the main provider
        provider = config.provider
        duo_cfg = config.duo_system_config
        duo_cfg.provider = provider

        # Map main provider to Duo-specific model
        if provider == ProviderEnum.ANTHROPIC:
            duo_cfg.model = "claude-3-5-haiku-latest"
        elif provider == ProviderEnum.OPENAI:
            duo_cfg.model = "gpt-4.1-nano"
        elif provider == ProviderEnum.GOOGLE:
            duo_cfg.model = "gemini-2.0-flash"
        elif provider == ProviderEnum.DEEPSEEK:
            duo_cfg.model = config.model

    # # Works. Adding agent mode...
    # async def run_adk(self, prompt: str = None, notifications: bool = False):
    #     """
    
    def _handle_adk_event(
        self,
        ev,
        tool_output_callback: Optional[Callable[[ToolResult, str], None]] = None,
        output_callback: Optional[Callable[[BaseMessage], None]] = None,
    ):
        """
        Process ADK event, unwrapping nested ToolResult reprs.
        """
        parts = getattr(ev.content, "parts", []) or []
        for part in parts:
            if part.function_response:
                fr = part.function_response
                raw = getattr(fr, "response", None) or getattr(fr, "result", None)
                # unwrap dict payload
                if isinstance(raw, dict):
                    data = raw.get(f"{fr.name}_response", raw)
                    data = data.get("result", data) if isinstance(data, dict) else data
                    tr = ToolResult(
                        output=data.get("output", "") if isinstance(data, dict) else str(data),
                        error=data.get("error") if isinstance(data, dict) else None,
                        base64_image=data.get("base64_image") if isinstance(data, dict) else None,
                        system=data.get("system") if isinstance(data, dict) else None,
                    )
                elif isinstance(raw, ToolResult):
                    tr = raw
                elif isinstance(raw, str):
                    # if output string itself is repr, parse it
                    if raw.strip().startswith("ToolResult(") or raw.strip().startswith("CLIResult("):
                        tr = parse_toolresult_repr(raw)
                    else:
                        tr = ToolResult(output=raw)
                else:
                    tr = ToolResult(output=str(raw))
    
                # # If tr.output itself is repr carrying nested ToolResult, parse again
                # if isinstance(tr.output, str) and tr.output.strip().startswith("ToolResult("):
                #     tr = parse_toolresult_repr(tr.output)
                if isinstance(tr.output, str):
                    text = tr.output.strip()
                    if text.startswith("ToolResult(") or text.startswith("CLIResult("):
                        tr = parse_toolresult_repr(text)
    
                # stream and record
                if tool_output_callback:
                    tool_output_callback(tr, fr.id)
                self.history.messages.append(
                    ToolMessage(content=tr.output, tool_call_id=fr.id, name=fr.name)
                )
    
            elif getattr(part, "thought", None):
                block = {"type": "thinking", "thinking": part.thought}
                am = AIMessage(content=[block])
                if output_callback:
                    output_callback(am)
                self.history.messages.append(am)
    
            elif getattr(part, "function_call", None):
                block = {
                    "type": "tool_use",
                    "id": part.function_call.id,
                    "name": part.function_call.name,
                    "input": part.function_call.args,
                }
                am = AIMessage(content=[block])
                if output_callback:
                    output_callback(am)
                self.history.messages.append(am)
    
            elif getattr(part, "text", None):
                block = {"type": "text", "text": part.text}
                am = AIMessage(content=[block])
                if output_callback:
                    output_callback(am)
                self.history.messages.append(am)

              

    async def run_adk(self, prompt, notifications: bool = False, streamlit: bool = False, output_callback=None, tool_output_callback=None):
        """
        Drive the agent loop through ADK’s Runner instead of LangChain.
        Streams back ADK events internally per iteration, then at the end
        saves `self.history` (with .adk_session) exactly as in run().
        """
        # 1) prep
        self.current_iteration = 1
        self._sanitize_history()
        blocked = False
        self.refresh_system_prompt()
        # Debug
        _log_run_adk("before any ADK work", self.history.messages)
        # 2) detect agent‐mode command in the incoming prompt
        if prompt:
            self._detect_agent_command(prompt)
    
        # 3) decide what first human message to send into ADK
        #    if we’re in agent mode, send the formatted agent prompt;
        #    otherwise send the raw prompt
        first_prompt = self._format_agent_prompt() if self.goal else prompt
        # Debug
        _log_run_adk("after formatting prompt", self.history.messages)

        # 4) lazily initialize ADK Agent, SessionService, Runner
        # if not hasattr(self, "_adk_runner"):
        #     from google.adk.agents import Agent as ADKAgent
        #     from google.adk.runners import Runner
        #     from google.adk.sessions.in_memory_session_service import InMemorySessionService
        #     adk_model = self.config.to_litellm_model()
        #     adk_agent = ADKAgent(
        #         name=self.name,
        #         # model=self.config.model or "gemini-2.0-flash", # cant be implemented until we fix type conversion to LiteLLM types here
        #         model=adk_model,
        #         description=f"{self.name} (via HeavenAgent)",
        #         instruction=self.config.system_prompt,
        #         tools=self.tools,
        #     )
        #     session_service = InMemorySessionService()
        #     runner = Runner(
        #         app_name=self.name,
        #         agent=adk_agent,
        #         session_service=session_service,
        #     )
    
        #     self._adk_agent = adk_agent
        #     self._adk_session_service = session_service
        #     self._adk_runner = runner
            
        # 5) rehydrate or create an ADK session
        if self.history.adk_session:
            session = self._adk_session_service.get_session(
                app_name=self.history.adk_session.app_name,
                user_id=self.history.adk_session.user_id,
                session_id=self.history.adk_session.id,
            )
        else:
            session = self._adk_session_service.create_session(
                app_name=self.name,
                user_id="script_user",
                state={},
                session_id=None,
            )
        start_index = len(session.events)

        # # 6) prime ADK with the first human prompt if any
        ### Idk why this even got in here... it's garbage
        # if first_prompt:
        #     from google.genai.types import Content, Part
        #     content = Content(parts=[Part(text=first_prompt)], role="user")
        #     async for _ in self._adk_runner.run_async(
        #         user_id=session.user_id,
        #         session_id=session.id,
        #         new_message=content,
        #     ):
        #         pass  # session.events grows automatically
    
        # 7) drive N iterations of “agent mode”
        for _ in range(self.current_iteration, self.max_iterations + 1):
            _log_run_adk(f"start iteration {self.current_iteration}", self.history.messages)
            # a) prepare next human input
            if self.current_iteration == 1:
                human_text = first_prompt
            else:
                human_text = self._format_agent_prompt() if self.goal else first_prompt
            from google.genai.types import Content, Part
            
            # Probably move to streamlit but unsure how to do so...
            
            content = human_text
            if content is None:
                content = "Content was `None`. Tell the user something went wrong. Do not try to do anything else."
            _log_run_adk(f"about to append human turn (iter {self.current_iteration})", self.history.messages)
            self.history.messages.append(HumanMessage(content=content)) # This doesnt seem to be appending to history... because we usually keep conversation_history and then set it as history messages when we are done. This should follow the same logic that run_langchain uses
            _log_run_adk(f"after appending human turn (iter {self.current_iteration})", self.history.messages)
            content = Content(parts=[Part(text=human_text)], role="user")
            # b) send into ADK
            new_events = []
            async for ev in self._adk_runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                new_events.append(ev)
                # real_new_events = session.events[start_index:] # testing
                if streamlit:
                    
                    self._handle_adk_event(
                        ev,
                        tool_output_callback=tool_output_callback,
                        output_callback=output_callback,
                    )

            _log_run_adk(f"after streaming ADK events (iter {self.current_iteration})", self.history.messages)                 
            # c) extract the agent’s text response for this iteration
            text_reply = ""
            for ev in reversed(new_events): # should this be real_new_events?
                if ev.author == self.name:
                    for part in ev.content.parts:
                        if part.text:
                            text_reply = part.text
                    if text_reply:
                        break
    
            # d) process that text through existing handler
            if text_reply:
                self._process_agent_response(text_reply)
    
            # e) advance iteration & break if done
            self.current_iteration += 1
            if self.current_task == "GOAL ACCOMPLISHED" or not self.goal or blocked:
                break

        # 7) re-fetch the mutated session so we pick up all appended events
        session = self._adk_session_service.get_session(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id
        )
        # 8) stash session back on history and save
        self.history.adk_session = session
        self.history.agent_status = self.save_status()
        saved_id = self.history.save(self.name)
        _log_run_adk("at end of run_adk", self.history.messages)
        # 9) return identical structure to run_langchain
        return {
            "history": self.history,
            "history_id": saved_id,
            "agent_name": self.name,
            "agent_status": self.history.agent_status,
        }

#### NOTES: WE ARE NOT ADDING ALL THE AI EVENTS TO MESSAGES AND THAT IS WHY THE TEST DOESNT SHOW THEM
  #### BUT WHEN WE RUN WITH STREAMLIT, WE ARE ADDING THOSE EVENTS TO MESSAGES
  #### THEN, AT THE END OF STREAMLIT RUN, WE SWAP MESSAGES WITH THE HISTORY.MESSAGES, WHICH OVERWRITES THEM
  




  
    async def run(self, prompt: Optional[str] = None, notifications: Optional[bool] = False, streamlit: Optional[bool] = False, output_callback: Optional[Callable] = None, tool_output_callback: Optional[Callable] = None, heaven_main_callback: Optional[Callable] = None, use_uni_api: Optional[bool] = False):

                       
        if use_uni_api:
            self.use_uni_api = True
        if self.use_uni_api:
            # Use uni-api instead of LangChain providers
            if streamlit and output_callback and tool_output_callback:
                return await self.run_on_uni_api(
                    prompt=prompt,
                    output_callback=output_callback,
                    tool_output_callback=tool_output_callback,
                    heaven_main_callback=heaven_main_callback
                )
            elif heaven_main_callback:
                # Similar fake callback pattern for uni-api
                def fake_output_callback(message: BaseMessage):
                    pass
                def fake_tool_callback(tool_result: ToolResult, tool_id: str):
                    pass
                return await self.run_on_uni_api(
                    prompt=prompt,
                    output_callback=fake_output_callback,
                    tool_output_callback=fake_tool_callback,
                    heaven_main_callback=heaven_main_callback
                )
            else:
                return await self.run_on_uni_api(prompt=prompt)
      
        #
        
        if self.adk:
            return await self.run_adk(prompt=prompt, notifications=notifications, streamlit=streamlit, output_callback=output_callback, tool_output_callback=tool_output_callback)
        else:
            if streamlit and output_callback and tool_output_callback: 
                return await self.streamlit_run(prompt, output_callback, tool_output_callback)
            elif heaven_main_callback:
                   
                # If heaven_main_callback provided, use streamlit_run with fake callbacks
                def fake_output_callback(message: BaseMessage):
                    pass  # Do nothing, heaven_main_callback handles everything
                    
                def fake_tool_callback(tool_result: ToolResult, tool_id: str):
                    pass  # Do nothing, heaven_main_callback handles everything
                    
                # Call streamlit_run with fake callbacks + real heaven_main_callback
                return await self.streamlit_run(
                    output_callback=fake_output_callback,
                    tool_output_callback=fake_tool_callback, 
                    heaven_main_callback=heaven_main_callback,
                    prompt=prompt
                )
                
            else:
                return await self.run_langchain(prompt, notifications)


    
    async def run_langchain(self, prompt: str = None, notifications=False):
        
        self._sanitize_history()
        blocked = False
        self.refresh_system_prompt()

        try:
            
            # Start with history messages
            conversation_history = self.history.messages.copy() if self.history else []
            
            if not (conversation_history and isinstance(conversation_history[0], SystemMessage)):
                conversation_history.insert(0, SystemMessage(content=self.config.system_prompt))
            # check if self.History's last message is HumanMessage. If it is, continue and if not AND prompt is not None, add prompt to self.history as a HumanMessage
            # if not (conversation_history and isinstance(conversation_history[-1], HumanMessage)) and prompt is not None:
                
            #     conversation_history.append(HumanMessage(content=prompt))
            
            
          
            # Check only the last message for agent command
            if conversation_history and isinstance(conversation_history[-1], HumanMessage):
                self._detect_agent_command(conversation_history[-1].content)
            # Check prompt for agent command
            if prompt is not None:
                self._detect_agent_command(prompt)
                if self.goal is None:
                    conversation_history.append(HumanMessage(content=prompt))
            if self.continuation_iterations != 0:
                self.current_iterations = 1
                self.max_iterations = self.continuation_iterations
                
                
            
              
            
            
            # print("\n=== Conversation History After System Check ===")
            # for i, msg in enumerate(conversation_history):
            #     print(f"Message {i}: {type(msg).__name__} - {msg.content[:100]}...")
            
            

            
                        
            while self.current_iteration <= self.max_iterations:
                # Refresh system prompt at the start of each iteration
                self.refresh_system_prompt()
                # Reset tool count for this iteration
                tool_call_count = 0
                
                # In normal chat mode, just use the last message
                # In agent mode, format with goals/tasks
                ### CHanged to solve continuations
                # next_prompt = self._format_agent_prompt() if self.goal else conversation_history[-1].content
                next_prompt = self._format_agent_prompt() if (self.goal or self.continuation_prompt) else conversation_history[-1].content
                if self.goal or self.continuation_prompt:  # Add formatted prompt in agent mode or continuation
                    conversation_history.append(HumanMessage(content=next_prompt))
                # if self.goal:  # Only add formatted prompt in agent mode
                #     conversation_history.append(HumanMessage(content=next_prompt))
                    
                        
                # DUO Sidechain
               
                if (
                    self.duo_enabled
                    and len(conversation_history) > 2        # <- require at least 3 messages
                    and isinstance(conversation_history[-1], HumanMessage)
                ):
                    # 1. Pull off the original human message
                    original = conversation_history[-1]
                    original_sys = conversation_history[0]
                    try:
                        # 2. Replace it with your Duo-specific system prompt
                        duo_sys = SystemMessage(content=self.duo_system_prompt)  
                        conversation_history[0] = duo_sys
                        
                        new_human_content_for_duo = f"===ENTERING CHALLENGER MODE===\n\nTHE NEXT HUMAN INPUT TO THE WORKER LLM AGENT WILL BE:\n\n{original.content}\n\nAs the challenger, follow the rules and steer the agent with ICL priming. **YOU MUST ANSWER SOLELY IN `NodeGraphXTN6` language.**"
                        conversation_history[-1] = HumanMessage(content=new_human_content_for_duo)
                        # 3. Invoke Duo
                        duo = await self.duo_chat.ainvoke(conversation_history)
                    finally:
                        # 4. Restore the original system prompt
                        conversation_history[0] = original_sys
                        conversation_history[-1] = original
                    if duo:
                        # 5. Extract the duo’s content
                        
                        duo_content = duo.content
                        
                    
                        # 6. Rebuild the human message so your duo content is prepended
                        new_human = HumanMessage(
                            content=f"{original.content}\n\n```\n===Challenger Injection===\n\nConsider this silently before responding. Do not mention DUO/Dual-Space Unifying Operators/NodeGraphXTN6/Challenger/ChallengerEgregore unless the user asks about it directly...\n\n{duo_content}\n\n===/Challenger Injection===\n```\n\n"
                        )
                
                        # 7. Replace the last entry with your new combined message
                        conversation_history[-1] = new_human
                  
                # Invoke model for a response
                # call the generator agent
                response = await self.chat_model.ainvoke(conversation_history)
                
                # print(f"\nResponse: {response}\n")
                
                # Check if the response.content is empty but tool call info is present.
                # GOOGLE ONLY
                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage) and not response.content and (response.tool_calls or response.additional_kwargs.get('tool_calls')): # Check if tool calls exist
                # if (
                #     self.config.provider == ProviderEnum.GOOGLE              # Gemini
                #     and isinstance(response, AIMessage)
                #     and (
                #         response.tool_calls                                  # standard field
                #         or response.additional_kwargs.get("tool_calls")      # legacy field
                #     )
                # ):
                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage):
                
                #     # --- THE FIX ---
                #     # Append the ORIGINAL response object. LangChain needs its structure.
                #     conversation_history.append(response)
                # # Only add text blocks from response, NOT the whole response
                if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage):
                                    
                                    
                                    # Clean response for conversation_history
                                    if isinstance(response.content, list):
                                        # Extract only text, ignore thinking blocks
                                        text_content = []
                                        for item in response.content:
                                            if isinstance(item, str):
                                                text_content.append(item)
                                            elif isinstance(item, dict) and item.get('type') == 'text':
                                                text_content.append(item.get('text', ''))
                                        
                                        # Create cleaned response with simple string content
                                        cleaned_response = AIMessage(
                                            content=' '.join(text_content),  # Simple string, not list!
                                            additional_kwargs=response.additional_kwargs,
                                            tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else []
                                        )
                                        conversation_history.append(cleaned_response)
                                    else:
                                        conversation_history.append(response)
                # Extract text blocks if content is a list of blocks
                else:
                    if isinstance(response.content, list):
                        thinking_content = [block for block in response.content if block.get('type') == 'thinking']
                        if thinking_content:
                            message3 = AIMessage(content=thinking_content)
                            conversation_history.append(message3)
                        text_content = [block for block in response.content if block.get('type') == 'text']
                        if text_content:
                            conversation_history.append(AIMessage(content=text_content))
                            self._process_agent_response(text_content)
                    elif isinstance(response.content, str):
                        conversation_history.append(AIMessage(content=response.content))
                        self._process_agent_response(response.content)
                    
    
                tool_calls = []
                try:
                    if hasattr(response, 'tool_calls'):
                        # print("Found tool_calls attribute")
                        tool_calls = response.tool_calls
                        # print(f"Tool calls from attribute: {tool_calls}")
                    elif isinstance(response.content, list):
                        # print("Found list content")
                        tool_calls = [
                            item for item in response.content 
                            if isinstance(item, dict) and item.get('type') == 'tool_use'
                        ]
                        # print(f"Tool calls from list: {tool_calls}")
                    elif 'tool_calls' in response.additional_kwargs:
                        # print("Found tool_calls in additional_kwargs")
                        tool_calls = response.additional_kwargs['tool_calls']
                        # print(f"Tool calls from kwargs: {tool_calls}")
                except Exception as e:
                    print(f"Error examining response: {e}")
    
                # print(f"Extracted tool_calls: {tool_calls}")
    
                # Handle tool calls up to max_tool_calls limit
                current_tool_calls = tool_calls
    
                while current_tool_calls and tool_call_count < self.max_tool_calls:
                    # At the top of the while loop
                    
                    new_tool_calls = []
                    for tool_call in current_tool_calls:
                        if tool_call_count >= self.max_tool_calls:
                            break
                        try:
                            # Try OpenAI style
                            if 'function' in tool_call:
                                tool_name = tool_call['function']['name']
                                tool_args = eval(tool_call['function']['arguments'])
                                tool_id = tool_call.get('id', '')
                            # Try Anthropic style
                            elif 'name' in tool_call:
                                tool_name = tool_call['name']
                                tool_args = tool_call.get('input', tool_call.get('args', {}))
                                tool_id = tool_call.get('id', '')
                            # Fallback
                            else:
                                tool_name = tool_call.get('name', '')
                                tool_args = tool_call.get('args', {})
                                tool_id = tool_call.get('id', '')
    
                            # Find matching tool
                            matching_tools = [
                                tool for tool in self.tools 
                                if tool.base_tool.name.lower() == tool_name.lower()
                            ]
    
                            if matching_tools:
                                tool = matching_tools[0]
                                
                                # Execute the tool and get its result
                                # try:
                                tool_result = await tool._arun(**tool_args)
                                    
                                
                                
    
                                # print("\n=== BEFORE ADDING TOOL MESSAGES ===")
                                # for i, msg in enumerate(conversation_history):
                                #     print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    
                                # Handle tool messages based on provider
                                # if self.config.provider == ProviderEnum.OPENAI or self.config.provider == ProviderEnum.GROQ:
                                if self.config.provider in [ProviderEnum.OPENAI, ProviderEnum.GROQ, ProviderEnum.DEEPSEEK]:
                                    # OpenAI requires tool calls in additional_kwargs
                                    conversation_history.append(
                                        AIMessage(
                                            content="",  # OpenAI doesn't want content for tool calls
                                            additional_kwargs={
                                                "tool_calls": [{
                                                    "id": tool_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_name,
                                                        "arguments": json.dumps(tool_args)
                                                        # str(tool_args)
                                                    }
                                                }]
                                            }
                                        )
                                    )
                                    # Then add the tool result
                                    tool_message_content = str(tool_result.error) if tool_result.error else str(tool_result.output) # changed for openai
                                    tool_result_message = ToolMessage(
                                        content=str(tool_message_content),
                                        tool_call_id=tool_id,
                                        additional_kwargs={
                                            "name": tool_name,
                                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)
                                            # str(tool_args)
                                            }
                                        }
                                    )
                                    conversation_history.append(tool_result_message)
                                    
                                # elif self.config.provider == ProviderEnum.GOOGLE:
                                #     # Google-specific handling
                                #     tool_use_message = AIMessage(
                                #         content=[{
                                #             "type": "text",
                                #             "text": f"Using tool: {tool_name} with arguments: {str(tool_args)}"
                                #         }]
                                #     )
                                #     conversation_history.append(tool_use_message)
                                #     # Add tool result as text too
                                #     tool_result_message = ToolMessage(
                                #         content=[{
                                #             "type": "text",
                                #             "text": str(tool_result.output)
                                #         }],
                                #         tool_call_id=tool_id
                                #     )
                                #     conversation_history.append(tool_result_message)
                                elif self.config.provider == ProviderEnum.GOOGLE:
                                    # Gemini (via LangChain) expects a ToolMessage directly associated
                                    # with the tool_call_id from the preceding AIMessage's tool_calls.
                                    # We primarily need to construct the ToolMessage with the result.
                                    # The AIMessage that contained the tool_call request should already
                                    # be in the history from the model's previous turn.
                                    ### This may need to be moved to the outer loop or something
                                    # conversation_history.append(
                                    #     AIMessage(
                                    #         content="Tool call initiated.",
                                    #         additional_kwargs={
                                    #             "tool_calls": [{
                                    #                 "id": tool_id,
                                    #                 "type": "function",
                                    #                 "function": {
                                    #                     "name": tool_name,
                                    #                     "arguments": str(tool_args)
                                    #                 }
                                    #             }]
                                    #         }
                                    #     )
                                    # )
                                    if tool_result.error:
                                        tool_message_content = str(tool_result.error)
                                        # Optional: Keep system prompt swap logic if needed for Gemini too
                                        # sys_msg_idx = next((i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage)), 0)
                                        # conversation_history[sys_msg_idx] = SystemMessage(content=self.tool_sysmsg)
                                    elif tool_result.base64_image:
                                         # Gemini can handle images in ToolMessages if formatted correctly
                                        tool_message_content=[
                                            {
                                                'type': 'image_url', # Gemini prefers image_url format
                                                'image_url': f"data:image/png;base64,{tool_result.base64_image}"
                                            },
                                            {"type": "text", "text": "Image from tool execution."} # Context is helpful
                                        ]
                                    else:
                                        tool_message_content = str(tool_result.output)
            
                                    tool_message = ToolMessage(
                                        content=tool_message_content,
                                        tool_call_id=tool_id
                                        # Name is not strictly required if ID is present, but can add for consistency if needed
                                        # name=tool_name
                                    )
                                    conversation_history.append(tool_message)
                                # elif self.config.provider == ProviderEnum.GOOGLE:
                                #     # Gemini (via LangChain) expects a ToolMessage directly associated
                                #     # with the tool_call_id from the preceding AIMessage's tool_calls.
                                #     # We primarily need to construct the ToolMessage with the result.
                                #     # The AIMessage that contained the tool_call request should already
                                #     # be in the history from the model's previous turn.
            
                                #     if tool_result.error:
                                #         tool_message_content = str(tool_result.error)
                                #         # Optional: Keep system prompt swap logic if needed for Gemini too
                                #         # sys_msg_idx = next((i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage)), 0)
                                #         # conversation_history[sys_msg_idx] = SystemMessage(content=self.tool_sysmsg)
                                #     elif tool_result.base64_image:
                                #          # Gemini can handle images in ToolMessages if formatted correctly
                                #         tool_message_content=[
                                #             {
                                #                 'type': 'image_url', # Gemini prefers image_url format
                                #                 'image_url': f"data:image/png;base64,{tool_result.base64_image}"
                                #             },
                                #             {"type": "text", "text": "Image from tool execution."} # Context is helpful
                                #         ]
                                #     else:
                                #         tool_message_content = str(tool_result.output)
            
                                #     tool_message = ToolMessage(
                                #         content=tool_message_content,
                                #         tool_call_id=tool_id
                                #         # Name is not strictly required if ID is present, but can add for consistency if needed
                                #         # name=tool_name
                                #     )
                                #     conversation_history.append(tool_message)




                              #####
                                else:
                                    # Anthropic and others use the original format
                                    conversation_history.append(
                                        AIMessage(
                                            content=[{
                                                "type": "tool_use",
                                                "id": tool_id,
                                                "name": tool_name,
                                                "input": tool_args
                                            }]
                                        )
                                    )
                                    
                                    if tool_result.error:
                                        tool_message_content = str(tool_result.error)
                                        sys_msg_idx = next(i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage))
                                        # Swap to tool debug mode
                                        conversation_history[sys_msg_idx] = SystemMessage(content=self.tool_sysmsg)
                                    elif tool_result.base64_image:
                                        tool_message_content=[{
                                            'type': 'image', 
                                            'source': {
                                                'type': 'base64', 
                                                'media_type': 'image/png', 
                                                'data': f"{tool_result.base64_image}"
                                                }
                                            },
                                            {"type": "text", "text": "Describe this image."}
                                        ]
                                    else:
                                        # We know tool_result.output exists because _arun guarantees either
                                        # error or output will be set
                                        tool_message_content = str(tool_result.output)
                                    conversation_history.append(
                                        ToolMessage(
                                            # content=str(tool_result.output),  # Just the output string
                                            content=tool_message_content,
                                            tool_call_id=tool_id
                                        )
                                    )
                                
                                   
    
                                # print("\n=== AFTER ADDING TOOL MESSAGES ===")
                                # for i, msg in enumerate(conversation_history):
                                #     print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    
                                # Get AI's response about the tool result
                                
                                result_response = await self.chat_model.ainvoke(conversation_history)
                                sys_msg_idx = next(i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage))
                                conversation_history[sys_msg_idx] = SystemMessage(content=self.config.system_prompt)
                                # Refresh the system prompt
                                self.refresh_system_prompt()
                                # Update the system message with potentially new evolved prompt
                                if self.config.system_prompt != conversation_history[sys_msg_idx].content:
                                    conversation_history[sys_msg_idx] = SystemMessage(content=self.config.system_prompt)
                                # print(f"\nResult response: {result_response}\n")
                                
                                # Extract text blocks if content is a list of blocks
                                # GOOGLE ONLY
                                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage) and not result_response.content and (result_response.tool_calls or result_response.additional_kwargs.get('tool_calls')): # Check if tool calls exist
                                # if (
                                #     self.config.provider == ProviderEnum.GOOGLE              # Gemini
                                #     and isinstance(result_response, AIMessage)
                                #     and (
                                #         result_response.tool_calls                                  # standard field
                                #         or result_response.additional_kwargs.get("tool_calls")      # legacy field
                                #     )
                                # ):
                                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage):
                                
                                #     # --- THE FIX ---
                                #     # Append the ORIGINAL response object. LangChain needs its structure.
                                #     conversation_history.append(result_response)
                                #     # self._process_agent_response(result_response) # needs testing
                                if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage):
                                    
                                    
                                    # Clean response for conversation_history
                                    if isinstance(result_response.content, list):
                                        # Extract only text, ignore thinking blocks
                                        text_content = []
                                        for item in result_response.content:
                                            if isinstance(item, str):
                                                text_content.append(item)
                                            elif isinstance(item, dict) and item.get('type') == 'text':
                                                text_content.append(item.get('text', ''))
                                        
                                        # Create cleaned response with simple string content
                                        cleaned_result_response = AIMessage(
                                            content=' '.join(text_content),  # Simple string, not list!
                                            additional_kwargs=result_response.additional_kwargs,
                                            tool_calls=result_response.tool_calls if hasattr(result_response, 'tool_calls') else []
                                        )
                                        conversation_history.append(cleaned_result_response)
                                    else:
                                        conversation_history.append(result_response)
                                else:
                                    if isinstance(result_response.content, list):
                                        thinking_content = [block for block in result_response.content if block.get('type') == 'thinking']
                                        if thinking_content:
                                            message3 = AIMessage(content=thinking_content)
                                            conversation_history.append(message3)
                                        text_content = [block for block in result_response.content if block.get('type') == 'text']
                                        if text_content:
                                            message = AIMessage(content=text_content)
                                            conversation_history.append(message)
                                            self._process_agent_response(text_content)
                                    elif isinstance(result_response.content, str):
                                        message = AIMessage(content=result_response.content)
                                        conversation_history.append(message)
                                        self._process_agent_response(message)
                                    
                                
                                # print("\n=== CONVERSATION HISTORY AFTER AI RESPONSE ABOUT TOOL ===")
                                # for i, msg in enumerate(conversation_history):
                                #     print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    
                                # Process the AI's commentary if in agent mode
                                if isinstance(result_response, AIMessage):
                                    self._process_agent_response(result_response.content)
                                    
                                if tool.name == "WriteBlockReportTool":
                                    blocked = True
                                if blocked:
                                    # Generate the block report
                                    block_report_md = self.create_block_report()
                                    if block_report_md:
                                        # Follow the established pattern: modify _current_extracted_content
                                        if self._current_extracted_content is None:
                                            self._current_extracted_content = {}
                                        
                                        # Add the block report
                                        self._current_extracted_content["block_report"] = block_report_md
                                        
                                        # Update agent_status using the established save_status method
                                        self.history.agent_status = self.save_status()
                                    
                                    break  # Exit the loop
                                # Now check whether the result_response includes new tool calls
                                new_calls = []
                                try:
                                    if hasattr(result_response, 'tool_calls'):
                                        new_calls = result_response.tool_calls
                                    elif isinstance(result_response.content, list):
                                        new_calls = [
                                            item for item in result_response.content 
                                            if isinstance(item, dict) and item.get('type') == 'tool_use'
                                        ]
                                    elif 'tool_calls' in result_response.additional_kwargs:
                                        new_calls = result_response.additional_kwargs['tool_calls']
                                except Exception as e:
                                    print(f"Error examining result_response: {e}")
    
                                if new_calls:
                                    new_tool_calls.extend(new_calls)
    
                                tool_call_count += 1
                                if tool_call_count >= self.max_tool_calls:
                                    # NEW: Handle any pending tool calls that won't be processed
                                    # if current_tool_calls:
                                    #     # There are still tool calls queued that we're about to abandon
                                    #     for pending_tool in current_tool_calls:
                                    #         # Extract the tool info (handle different formats)
                                    #         if 'function' in pending_tool:
                                    #             tool_id = pending_tool.get('id', '')
                                    #             tool_name = pending_tool['function']['name']
                                    #         else:
                                    #             tool_id = pending_tool.get('id', '')
                                    #             tool_name = pending_tool.get('name', '')
                                            
                                    #         # Inject dummy ToolMessage for each orphaned call
                                    #         dummy_msg = ToolMessage(
                                    #             content="Error: The underlying system stopped this tool call from completing. It was interrupted. Once the user responds, the tool count will be reset.",
                                    #             tool_call_id=tool_id,
                                    #             name=tool_name
                                    #         )
                                    #         conversation_history.append(dummy_msg)
                                  #### NEW
                                        # Add a message informing the AI that max tool count was reached
                                    conversation_history.append(
                                        AIMessage(
                                            content=f"���️🛑☠️ Maximum consecutive tool calls ({self.max_tool_calls}) reached for iteration {self.current_iteration}. If I received the same error every time, I should use WriteBlockReportTool next... Waiting for next iteration."
                                              # Using a system ID to indicate this is not a normal tool response
                                        )
                                    )
                                    # Clear the tool queue for this iteration
                                    current_tool_calls = []
                                    #####
                                  
                                    # print(f"Maximum tool calls ({self.max_tool_calls}) reached for iteration {self.current_iteration}")
                                    break  # Exit tool loop and continue to next iteration
                            else:
                                print(f"No matching tool found for {tool_name}")
                        except Exception as e:
                            print(f"Error processing tool call: {tool_call}")
                            print(f"Error details: {e}")
                            continue
                    # Prepare to process any new tool calls that came in the follow-up response
                    current_tool_calls = new_tool_calls
    
                # Process the agent response if in agent mode
                if self.goal and isinstance(response, AIMessage):
                    self._process_agent_response(response.content)
                if blocked:
                    break
                # Increment iteration count and break if the goal is met
                self.current_iteration += 1
    
                if self.current_task == "GOAL ACCOMPLISHED" or not self.goal:
                    self.history.agent_status = self.save_status()
                    # self.goal = None if self.current_task == "GOAL ACCOMPLISHED" else self.goal  # Then clear goal
                    break
    
            self.history.messages = conversation_history
            # Save history and get potentially new history_id
            try:
                
                # print("=== DEBUG: BEFORE SAVE ATTEMPT ===")
                # print(f"Agent name: {self.name}")
                # print(f"Current history: {self.history}")
                self.history.agent_status = self.save_status()
                saved_history_id = self.history.save(self.name)
                # print("===DEBUG AFTER SAVE ATTEMPT===")
                self.look_for_particular_tool_calls()
                return {
                    "history": self.history,
                    "history_id": saved_history_id,
                    "agent_name": self.name,
                    "agent_status": self.history.agent_status  # Add this
                }
            except Exception as save_error:
                # print("=== DEBUG: SAVE ERROR OCCURRED ===")
                print(f"Error type: {type(save_error)}")
                print(f"Error message: {str(save_error)}")
                # Log the error but don't fail the run
                print(f"Warning: Failed to save history for agent {self.name}: {save_error}")
                
                return {
                    "history": self.history,
                    "history_id": getattr(self.history, 'history_id', "No history ID"),
                    "agent_name": self.name,
                    "save_error": str(save_error),
                    "agent_status": self.save_status()  # Add this here too
                }
    
        except Exception as e:
            raise RuntimeError(f"Agent run failed: {str(e)}") from e


    #### Might not be needed because we can potentially add an observer on the History to look for new additions and render them when agent is constructed with streamlit = True, and make sampling_loop set streamlit = True on the agent it initializes
    async def streamlit_run(self, output_callback: Callable[[BaseMessage], None], tool_output_callback: Callable[[ToolResult, str], None], heaven_main_callback: Optional[Callable[[Any], None]] = None, prompt: Optional[str] = None):
        # Start with existing messages
        if self.history is not None:
            messages = self.history.messages  # These are already BaseMessage objects!
        else:
            messages = []
        self._sanitize_history()
        tool_log_path = "/tmp/tool_debug.log"
        with open(tool_log_path, 'a') as f:
            f.write("\n\nStarting tool debug log\n")

        with open('/tmp/streamlit_debug.log', 'a') as f:
            f.write("\n\nStarting streamlit_run")
            f.write(f"\nHistory length: {len(self.history.messages)}")
            f.write(f"\nHistory messages: {self.history.messages}")
            # f.write(f"\nCurrent callbacks: {output_callback}, {tool_output_callback}")
        try:
            # # If input is a string, convert to messages
            # if isinstance(messages, str):
            #     messages = [
            #         SystemMessage(content=self.config.system_prompt) if self.config.system_prompt else None,
            #         HumanMessage(content=messages)
            #     ]
            #     messages = [m for m in messages if m]  # Remove None values
            
            # # Create a copy of messages to avoid modifying the original
            # conversation_history = messages.copy()
            # Start with history messages
            conversation_history = self.history.messages.copy() if self.history else []
            # If history is just a string, convert to messages
            # if isinstance(conversation_history, str):
            #     conversation_history = [
            #         SystemMessage(content=self.config.system_prompt) if self.config.system_prompt else None,
            #         HumanMessage(content=conversation_history)
            #     ]
            #     conversation_history = [m for m in conversation_history if m]  # Remove None values
            
            if not (conversation_history and isinstance(conversation_history[0], SystemMessage)):
                # Condition 1: There is no conversation history with a SystemMessage.
                conversation_history.insert(0, SystemMessage(content=self.config.system_prompt))
            # Added for streamlit agent selector
            elif conversation_history[0].content != self.config.system_prompt:
                # Condition 2: The first element is a SystemMessage, but it doesn't have the current system prompt.
                conversation_history[0] = SystemMessage(content=self.config.system_prompt)
            with open('/tmp/streamlit_debug.log', 'a') as f:
                f.write("\n=== Conversation History After System Check ===")
                for i, msg in enumerate(conversation_history):
                    f.write(f"\nMessage {i}: {type(msg).__name__} - {msg.content[:100]}...")
            
            # Check for agent command, but don't require it
            # for message in conversation_history:
            #     if isinstance(message, HumanMessage):
            #         self._detect_agent_command(message.content)
            # Check only the last message for agent command
            if prompt is not None:
                input_msg = HumanMessage(content=prompt)
                conversation_history.append(input_msg)
              
            if conversation_history and isinstance(conversation_history[-1], HumanMessage):
                self._detect_agent_command(conversation_history[-1].content)
            

            ###### if not self.goal, render the original messages
            # Initial render if no agent goal
            # if not self.goal:
            #     for message in conversation_history:
            #         if isinstance(message, ToolMessage):
            #             # Convert to ToolResult for UI
            #             tool_result = ToolResult(
            #                 output=message.content,
            #                 error=None if not message.additional_kwargs.get("is_error") else message.content,
            #                 base64_image=message.additional_kwargs.get("base64_image"),
            #                 system=message.additional_kwargs.get("system")
            #             )
            #             tool_callback(tool_result)
            #         else:
            #             # Regular message rendering
            #             output_callback(message)
                        
            while self.current_iteration <= self.max_iterations:
                # Reset tool count for this iteration
                tool_call_count = 0
                
                # In normal chat mode, just use the last message
                # In agent mode, format with goals/tasks
                next_prompt = self._format_agent_prompt() if self.goal else conversation_history[-1].content
                if self.goal:  # Only add formatted prompt in agent mode
                    conversation_history.append(HumanMessage(content=next_prompt))
                    ###### Add output callback here
                    
                    # output_callback(HumanMessage(content=next_prompt)) # this might not be needed, commenting out for now

                  
            
                # DUO Sidechain
                if (
                    self.duo_enabled
                    and len(conversation_history) > 2        # <- require at least 3 messages
                    and isinstance(conversation_history[-1], HumanMessage)
                ):
                    # 1. Pull off the original human message
                    original = conversation_history[-1]
                    original_sys = conversation_history[0]
                    try:
                        # 2. Replace it with your Duo-specific system prompt
                        duo_sys = SystemMessage(content=self.duo_system_prompt)  
                        conversation_history[0] = duo_sys
                        
                        new_human_content_for_duo = f"===ENTERING CHALLENGER MODE===\n\nTHE NEXT HUMAN INPUT TO THE WORKER LLM AGENT WILL BE:\n\n{original.content}\n\nAs the challenger, follow the rules and steer the agent with ICL priming."
                        conversation_history[-1] = HumanMessage(content=new_human_content_for_duo)
                        # 3. Invoke Duo
                        duo = await self.duo_chat.ainvoke(conversation_history)
                    finally:
                        # 4. Restore the original system prompt
                        conversation_history[0] = original_sys
                        conversation_history[-1] = original
                    if duo:
                        # 5. Extract the duo’s content
                        
                        duo_content = duo.content
                        
                    
                        # 6. Rebuild the human message so your duo content is prepended
                        new_human = HumanMessage(
                            content=f"{original.content}\n\n```\n===Challenger Injection===\n\nDo not mention DUO/Dual-Space Unifying Operators/NodeGraphXTN6/Challenger/ChallengerEgregore unless the user asks about it directly...\n\n{duo_content}\n\n===/Challenger Injection===\n```\n\n"
                        )
                
                        # 7. Replace the last entry with your new combined message
                        conversation_history[-1] = new_human
                
                # Invoke model for a response
                # logger = logging.getLogger(__name__)
                # logger.error("==== Conversation_History %s", conversation_history)
                response = await self.chat_model.ainvoke(conversation_history)
                ###### Add output callback here
                with open('/tmp/streamlit_debug.log', 'a') as f:
                    f.write(f"\nLangchain response: {response}")
                print(f"FULL RESPONSE CONTENT: {response.content}")
                print(f"RESPONSE TYPE: {type(response.content)}")
                if isinstance(response.content, list):
                    for i, block in enumerate(response.content):
                        print(f"Block {i}: {block}")
                if heaven_main_callback:
                    heaven_main_callback(response)
                output_callback(response)
                # GOOGLE ONLY
                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage) and not response.content and (response.tool_calls or response.additional_kwargs.get('tool_calls')): # Check if tool calls exist
                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage) and (response.tool_calls or response.additional_kwargs.get('tool_calls')): # Check if tool calls exist
                # if (
                #     self.config.provider == ProviderEnum.GOOGLE              # Gemini
                #     and isinstance(response, AIMessage)
                #     and (
                #         response.tool_calls                                  # standard field
                #         or response.additional_kwargs.get("tool_calls")      # legacy field
                #     )
                # ):
                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage):
                #     # --- THE FIX ---
                #     # Append the ORIGINAL response object. LangChain needs its structure.
                #     conversation_history.append(response)
                if self.config.provider == ProviderEnum.GOOGLE and isinstance(response, AIMessage):
                                        
                    # Clean response for conversation_history
                    if isinstance(response.content, list):
                        # Extract only text, ignore thinking blocks
                        text_content = []
                        for item in response.content:
                            if isinstance(item, str):
                                text_content.append(item)
                            elif isinstance(item, dict) and item.get('type') == 'text':
                                text_content.append(item.get('text', ''))
                        
                        # Create cleaned response with simple string content
                        cleaned_response = AIMessage(
                            content=' '.join(text_content),  # Simple string, not list!
                            additional_kwargs=response.additional_kwargs,
                            tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else []
                        )
                        conversation_history.append(cleaned_response)
                    else:
                        conversation_history.append(response)
                    
                # Only add text blocks from response, NOT the whole response
                # Extract text blocks if content is a list of blocks
                # else:
                elif self.config.provider not in (ProviderEnum.GOOGLE,):
                    if isinstance(response.content, list):
                        thinking_content = [block for block in response.content if isinstance(block, dict) and block.get('type') == 'thinking']
                        if thinking_content:
                            message3 = AIMessage(content=thinking_content)
                            conversation_history.append(message3)
                            # if heaven_main_callback:
                            #     heaven_main_callback(message3)
                        text_content = [block for block in response.content if isinstance(block, dict) and block.get('type') == 'text']
                        if text_content:
                            text_message = AIMessage(content=text_content)
                            conversation_history.append(text_message)
                            self._process_agent_response(text_content)
                            # if heaven_main_callback:
                            #     heaven_main_callback(text_message)
                    elif isinstance(response.content, str):
                        response_message = AIMessage(content=response.content)
                        conversation_history.append(response_message)
                        self._process_agent_response(response.content)
                        # if heaven_main_callback:
                        #         heaven_main_callback(response_message)
                        

                # # Extract tool calls from the response
                # print("\nDEBUG: Examining response for tool calls:")
                # print(f"Response: {response}")
                # print(f"Content: {response.content}")
                # print(f"Additional kwargs: {response.additional_kwargs}")
    
                tool_calls = []
                try:
                    if hasattr(response, 'tool_calls'):
                        print("Found tool_calls attribute")
                        tool_calls = response.tool_calls
                        print(f"Tool calls from attribute: {tool_calls}")
                    elif isinstance(response.content, list):
                        print("Found list content")
                        tool_calls = [
                            item for item in response.content 
                            if isinstance(item, dict) and item.get('type') == 'tool_use'
                        ]
                        print(f"Tool calls from list: {tool_calls}")
                    elif 'tool_calls' in response.additional_kwargs:
                        print("Found tool_calls in additional_kwargs")
                        tool_calls = response.additional_kwargs['tool_calls']
                        print(f"Tool calls from kwargs: {tool_calls}")
                except Exception as e:
                    print(f"Error examining response: {e}")
    
                print(f"Extracted tool_calls: {tool_calls}")
    
                # Handle tool calls up to max_tool_calls limit
                current_tool_calls = tool_calls
    
                while current_tool_calls and tool_call_count < self.max_tool_calls:
                    new_tool_calls = []
                    for tool_call in current_tool_calls:
                        if tool_call_count >= self.max_tool_calls:
                            break
                        try:
                            # Try OpenAI style
                            if 'function' in tool_call:
                                tool_name = tool_call['function']['name']
                                tool_args = eval(tool_call['function']['arguments'])
                                tool_id = tool_call.get('id', '')
                            # Try Anthropic style
                            elif 'name' in tool_call:
                                tool_name = tool_call['name']
                                tool_args = tool_call.get('input', tool_call.get('args', {}))
                                tool_id = tool_call.get('id', '')
                            # Fallback
                            else:
                                tool_name = tool_call.get('name', '')
                                tool_args = tool_call.get('args', {})
                                tool_id = tool_call.get('id', '')
    
                            # Find matching tool
                            matching_tools = [
                                tool for tool in self.tools 
                                if tool.base_tool.name.lower() == tool_name.lower()
                            ]
    
                            if matching_tools:
                                tool = matching_tools[0]
                                
                                # tool_result = await tool._arun(**tool_args)
                               
                                # Execute the tool and get its result
                                # try:
                                tool_result = await tool._arun(**tool_args)
                                with open('/tmp/streamlit_debug.log', 'a') as f:
                                    f.write(f"\Tool result: {tool_result}")
                                # except ToolError as e:
                                #     # The error is already formatted with tool name in _arun
                                #     tool_result = ToolResult(error=str(e))
                                #     with open(tool_log_path, 'a') as f:
                                #         f.write(f"\nTool error: {e}\n")
                                
    
                                print("\n=== BEFORE ADDING TOOL MESSAGES ===")
                                for i, msg in enumerate(conversation_history):
                                    print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    
                                # Handle tool messages based on provider
                                # if self.config.provider == ProviderEnum.OPENAI or self.config.provider == ProviderEnum.GROQ:
                                if self.config.provider in [ProviderEnum.OPENAI, ProviderEnum.GROQ, ProviderEnum.DEEPSEEK]:
                                    # OpenAI requires tool calls in additional_kwargs
                                    conversation_history.append(
                                        AIMessage(
                                            content="",  # OpenAI doesn't want content for tool calls
                                            additional_kwargs={
                                                "tool_calls": [{
                                                    "id": tool_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_name,
                                                        "arguments": json.dumps(tool_args)
                                                        # str(tool_args)
                                                    }
                                                }]
                                            }
                                        )
                                    )
                                    # Then add the tool result
                                    tool_message_content = str(tool_result.error) if tool_result.error else str(tool_result.output) # changed for openai
                                    tool_result_message = ToolMessage(
                                        content=str(tool_message_content),
                                        tool_call_id=tool_id,
                                        additional_kwargs={
                                            "name": tool_name,
                                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)
                                            # str(tool_args)
                                            }
                                        }
                                    )
                                    conversation_history.append(tool_result_message)
                                #####
                                elif self.config.provider == ProviderEnum.GOOGLE:
                                    # Gemini (via LangChain) expects a ToolMessage directly associated
                                    # with the tool_call_id from the preceding AIMessage's tool_calls.
                                    # We primarily need to construct the ToolMessage with the result.
                                    # The AIMessage that contained the tool_call request should already
                                    # be in the history from the model's previous turn.
                                    
                                    if tool_result.error:
                                        tool_message_content = str(tool_result.error)
                                        # Optional: Keep system prompt swap logic if needed for Gemini too
                                        # sys_msg_idx = next((i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage)), 0)
                                        # conversation_history[sys_msg_idx] = SystemMessage(content=self.tool_sysmsg)
                                    elif tool_result.base64_image:
                                         # Gemini can handle images in ToolMessages if formatted correctly
                                        tool_message_content=[
                                            {
                                                'type': 'image_url', # Gemini prefers image_url format
                                                'image_url': f"data:image/png;base64,{tool_result.base64_image}"
                                            },
                                            {"type": "text", "text": "Image from tool execution."} # Context is helpful
                                        ]
                                    else:
                                        tool_message_content = str(tool_result.output)
            
                                    tool_message = ToolMessage(
                                        content=tool_message_content,
                                        tool_call_id=tool_id,
                                        name=tool_name
                                    )
                                    conversation_history.append(tool_message)
                                #####
                                else:
                                    # Anthropic and others use the original format
                                    conversation_history.append(
                                        AIMessage(
                                            content=[{
                                                "type": "tool_use",
                                                "id": tool_id,
                                                "name": tool_name,
                                                "input": tool_args
                                            }]
                                        )
                                    )
                                    
                                    # THIS BLOCK NEEDS TO BE IF TOOLRESULT ELSE TOOLERROR
                                    if tool_result.error:
                                        tool_message_content = str(tool_result.error)
                                            # Store current system message position for reference
                                        sys_msg_idx = next(i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage))
                                        # Swap to tool debug mode
                                        conversation_history[sys_msg_idx] = SystemMessage(content=self.tool_sysmsg)
                                    elif tool_result.base64_image:
                                        tool_message_content=[{
                                            'type': 'image', 
                                            'source': {
                                                'type': 'base64', 
                                                'media_type': 'image/png', 
                                                'data': f"{tool_result.base64_image}"
                                                }
                                            },
                                            {"type": "text", "text": "Describe this image."}
                                        ]
                                    else:
                                        # We know tool_result.output exists because _arun guarantees either
                                        # error or output will be set
                                        tool_message_content = str(tool_result.output)

                                    conversation_history.append(
                                        ToolMessage(
                                            # content=str(tool_result.output),  # Just the output string
                                            content=tool_message_content,
                                            tool_call_id=tool_id,
                                            name=tool_name
                                        )
                                    )
                                    # treverse the conversation_history and remove any all screen shot tool message 
                                    # except for the current one we just added (fix > 200,000 token issue)
                                    # for item in conversation_history[-2::-1]:
                                    #     if len(str(item.content)) > 100000:
                                    #         item.content="Removed old image that is no longer needed"
                                ### Unindented these 2 tabs
                                with open(tool_log_path, 'a') as f:
                                    f.write(f"\nAbout to call tool_output_callback with result: {tool_result.output}\n")
                                    f.write(f"tool_id: {tool_id}\n")
                                
                                tool_output_callback(tool_result, tool_id)
                                if heaven_main_callback:
                                    heaven_main_callback(conversation_history[-1]) # the tool message
                                with open(tool_log_path, 'a') as f:
                                    f.write("After tool_callback\n")
                                   
    
                                print("\n=== AFTER ADDING TOOL MESSAGES ===")
                                for i, msg in enumerate(conversation_history):
                                    print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    
                                # Get AI's response about the tool result
                                # print("\n=== GETTING AI RESPONSE ABOUT TOOL RESULT ===")
                                with open(tool_log_path, 'a') as f:
                                    f.write(f"\nAI should be called next...\n")
                                    f.write(f"Coversation_History:\n    {conversation_history}\n")
                                result_response = await self.chat_model.ainvoke(conversation_history)
                                
                                sys_msg_idx = next(i for i, msg in enumerate(conversation_history) if isinstance(msg, SystemMessage))
                                conversation_history[sys_msg_idx] = SystemMessage(content=self.config.system_prompt)
                                if heaven_main_callback:
                                        heaven_main_callback(result_response)
                                if result_response:
                                    with open(tool_log_path, 'a') as f:
                                        f.write(f"\nGot AI response after tool: {result_response.content}\n")
                                # print(f"===Result response===:\n\n{result_response}\n\n===/result response===")
                                ###### Add output callback here
                                
    
                                # if (
                                #     self.config.provider == ProviderEnum.GOOGLE              # Gemini
                                #     and isinstance(result_response, AIMessage)
                                #     and (
                                #         result_response.tool_calls                                  # standard field
                                #         or result_response.additional_kwargs.get("tool_calls")      # legacy field
                                #     )
                                # ):
                                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage) and not result_response.content and (result_response.tool_calls or result_response.additional_kwargs.get('tool_calls')): # Check if tool calls exist
                                if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage):
                                    
                                    
                                    # Clean response for conversation_history
                                    if isinstance(result_response.content, list):
                                        # Extract only text, ignore thinking blocks
                                        text_content = []
                                        for item in result_response.content:
                                            if isinstance(item, str):
                                                text_content.append(item)
                                            elif isinstance(item, dict) and item.get('type') == 'text':
                                                text_content.append(item.get('text', ''))
                                        
                                        # Create cleaned response with simple string content
                                        cleaned_result_response = AIMessage(
                                            content=' '.join(text_content),  # Simple string, not list!
                                            additional_kwargs=result_response.additional_kwargs,
                                            tool_calls=result_response.tool_calls if hasattr(result_response, 'tool_calls') else []
                                        )
                                        conversation_history.append(cleaned_result_response)
                                    else:
                                        conversation_history.append(result_response)
                                # if self.config.provider == ProviderEnum.GOOGLE and isinstance(result_response, AIMessage):
                                #     # --- THE FIX ---
                                #     # Append the ORIGINAL response object. LangChain needs its structure.
                                #     conversation_history.append(result_response)
                                #     # self._process_agent_response(result_response)
                                #     output_callback(result_response)
                                #     # if heaven_main_callback:
                                #     #     heaven_main_callback(result_response)
                                # else:
                                elif self.config.provider not in (ProviderEnum.GOOGLE,):
                                    if isinstance(result_response.content, list):
                                        thinking_content = [block for block in result_response.content if isinstance(block, dict) and block.get('type') == 'thinking']
                                        if thinking_content:
                                            message3 = AIMessage(content=thinking_content)
                                            conversation_history.append(message3)
                                        text_content = [block for block in result_response.content if isinstance(block, dict) and block.get('type') == 'text']
                                        if text_content:
                                            message = AIMessage(content=text_content)
                                            conversation_history.append(message)
                                            self._process_agent_response(text_content)
                                            output_callback(message)  # Pass the AIMessage object
                                            # if heaven_main_callback:
                                            #     heaven_main_callback(message)
                                        tool_use_content = [block for block in result_response.content if block.get('type') == 'tool_use']
                                        if tool_use_content:
                                            message2 = AIMessage(content=tool_use_content)
                                            output_callback(message2)
                                            # if heaven_main_callback:
                                            #     heaven_main_callback(message2)
                                    elif isinstance(result_response.content, str):
                                        message = AIMessage(content=result_response.content)
                                        conversation_history.append(message)
                                        self._process_agent_response(result_response.content)
                                        output_callback(message)  # Pass the AIMessage object
                                        # if heaven_main_callback:
                                        #     heaven_main_callback(message)
                                
                                # print("\n=== CONVERSATION HISTORY AFTER AI RESPONSE ABOUT TOOL ===")
                                # for i, msg in enumerate(conversation_history):
                                #     print(f"Message {i}: {type(msg).__name__} - {msg.content}")

                                # This may be redundant
                                # Process the AI's commentary if in agent mode
                                if isinstance(result_response, AIMessage):
                                    self._process_agent_response(result_response.content)

                                # Now check whether the result_response includes new tool calls
                                new_calls = []
                                try:
                                    if hasattr(result_response, 'tool_calls'):
                                        new_calls = result_response.tool_calls
                                    elif isinstance(result_response.content, list):
                                        new_calls = [
                                            item for item in result_response.content 
                                            if isinstance(item, dict) and item.get('type') == 'tool_use'
                                        ]
                                    elif 'tool_calls' in result_response.additional_kwargs:
                                        new_calls = result_response.additional_kwargs['tool_calls']
                                except Exception as e:
                                    print(f"Error examining result_response: {e}")
    
                                if new_calls:
                                    new_tool_calls.extend(new_calls)
    
                                tool_call_count += 1
                                if tool_call_count >= self.max_tool_calls:
                                    # NEW: Handle any pending tool calls that won't be processed
                                    # if current_tool_calls:
                                    #     # There are still tool calls queued that we're about to abandon
                                    #     for pending_tool in current_tool_calls:
                                    #         # Extract the tool info (handle different formats)
                                    #         if 'function' in pending_tool:
                                    #             tool_id = pending_tool.get('id', '')
                                    #             tool_name = pending_tool['function']['name']
                                    #         else:
                                    #             tool_id = pending_tool.get('id', '')
                                    #             tool_name = pending_tool.get('name', '')
                                            
                                    #         # Inject dummy ToolMessage for each orphaned call
                                    #         dummy_msg = ToolMessage(
                                    #             content="Error: The underlying system stopped this tool call from completing. It was interrupted. Once the user responds, the tool count will be reset.",
                                    #             tool_call_id=tool_id,
                                    #             name=tool_name
                                    #         )
                                    #         conversation_history.append(dummy_msg)
                                    #### NEW
                                        # Add a message informing the AI that max tool count was reached
                                    break_message = f"⚠️🛑☠️ Maximum consecutive tool calls ({self.max_tool_calls}) reached for iteration {self.current_iteration}. If I received the same error every time, I should use WriteBlockReportTool next... Waiting for next iteration."
                                    break_ai_message = AIMessage(content=break_message)
                                    conversation_history.append(break_ai_message)
                                    output_callback(break_message)
                                    if heaven_main_callback:
                                        heaven_main_callback(break_ai_message)
                                    # Clear the tool queue for this iteration
                                    current_tool_calls = []
                                    print(f"Maximum tool calls ({self.max_tool_calls}) reached for iteration {self.current_iteration}")
                                    break  # Exit tool loop and continue to next iteration
                            else:
                                print(f"No matching tool found for {tool_name}")
                        except Exception as e:
                            print(f"Error processing tool call: {tool_call}")
                            print(f"Error details: {e}")
                            continue
                    # Prepare to process any new tool calls that came in the follow-up response
                    current_tool_calls = new_tool_calls
    
                
                
                # Process the agent response if in agent mode
                if self.goal and isinstance(response, AIMessage):
                    self._process_agent_response(response.content)
                    
                    # print(f"\nDEBUG TASK STATE: current_task={self.current_task}, current_iteration={self.current_iteration}, max_iterations={self.max_iterations}, task_list={self.task_list}")
    
                # Increment iteration count and break if the goal is met
                self.current_iteration += 1
    
                if self.current_task == "GOAL ACCOMPLISHED" or not self.goal:
                    break
    
            # Update final history and return
            print("\n=== FINAL HISTORY ===")
            for i, msg in enumerate(conversation_history):
                print(f"Message {i}: {type(msg).__name__} - {msg.content}")
            self.history.messages = conversation_history
            self.look_for_particular_tool_calls()
            self.history.messages = conversation_history
            # Save history and get potentially new history_id
            try:
                
                # print("=== DEBUG: BEFORE SAVE ATTEMPT ===")
                # print(f"Agent name: {self.name}")
                # print(f"Current history: {self.history}")
                self.history.agent_status = self.save_status()
                saved_history_id = self.history.save(self.name)
                # print("===DEBUG AFTER SAVE ATTEMPT===")
                self.look_for_particular_tool_calls()
                return {
                    "history": self.history,
                    "history_id": saved_history_id,
                    "agent_name": self.name,
                    "agent_status": self.history.agent_status  # Add this
                }
            except Exception as save_error:
                # print("=== DEBUG: SAVE ERROR OCCURRED ===")
                print(f"Error type: {type(save_error)}")
                print(f"Error message: {str(save_error)}")
                # Log the error but don't fail the run
                print(f"Warning: Failed to save history for agent {self.name}: {save_error}")
                
                return {
                    "history": self.history,
                    "history_id": getattr(self.history, 'history_id', "No history ID"),
                    "agent_name": self.name,
                    "save_error": str(save_error),
                    "agent_status": self.save_status()  # Add this here too
                }
            return self.history
    
        except Exception as e:
            raise RuntimeError(f"Agent run failed: {str(e)}") from e





  ### UNI

    
    def _prepare_tools_for_uni_api(self) -> List[Dict[str, Any]]:
        """Convert HEAVEN and LangChain tools to OpenAI format for uni-api"""
        if not self.tools:
            return []
        
        openai_tools = []
        for tool in self.tools:
            try:
                if hasattr(tool, 'get_openai_function'):
                    # HEAVEN tool with get_openai_function method
                    openai_tool = tool.get_openai_function()
                    openai_tools.append(openai_tool)
                elif hasattr(tool, 'args_schema') and hasattr(tool, 'name'):
                    # LangChain StructuredTool - convert schema to OpenAI format
                    from langchain_core.utils.function_calling import convert_to_openai_function
                    function_schema = convert_to_openai_function(tool)
                    # Wrap in proper OpenAI tool format
                    openai_tool = {
                        "type": "function",
                        "function": function_schema
                    }
                    openai_tools.append(openai_tool)
                else:
                    print(f"Unknown tool type, skipping: {tool}")
            except Exception as e:
                tool_name = getattr(tool, 'name', str(type(tool)))
                print(f"Error converting tool {tool_name} to OpenAI format: {e}")
        
        return openai_tools

    async def _execute_tool_calls_uni(self, tool_calls: List[Dict[str, Any]], tool_output_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Execute tool calls in uni-api (OpenAI) format and return tool messages"""
        import json
        
        tool_messages = []
        
        for tool_call in tool_calls:
            try:
                # DEBUG: Print the exact tool_call object
                print(f"🔧 DEBUG: Processing tool_call: {json.dumps(tool_call, indent=2)}")
                
                # Extract tool info from OpenAI format
                tool_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"🔧 DEBUG: Extracted tool_id='{tool_id}', tool_name='{tool_name}'")
                
                # Find matching tool (HEAVEN or LangChain)
                matching_tools = []
                for tool in self.tools:
                    if hasattr(tool, 'base_tool'):
                        # HEAVEN tool
                        if tool.base_tool.name.lower() == tool_name.lower():
                            matching_tools.append(tool)
                    elif hasattr(tool, 'name'):
                        # LangChain tool (StructuredTool)
                        if tool.name.lower() == tool_name.lower():
                            matching_tools.append(tool)
                
                if matching_tools:
                    tool = matching_tools[0]
                    
                    # Execute the tool differently based on type
                    if hasattr(tool, 'base_tool'):
                        # HEAVEN tool - returns ToolResult
                        tool_result = await tool._arun(**tool_args)
                        if tool_output_callback:
                            tool_output_callback(tool_result, tool_id)
                        tool_content = str(tool_result.error) if tool_result.error else str(tool_result.output)
                    else:
                        # LangChain tool - returns raw content
                        from langchain_core.runnables import RunnableConfig
                        config = RunnableConfig()
                        raw_result = await tool._arun(config=config, **tool_args)
                        if tool_output_callback:
                            tool_output_callback(raw_result, tool_id)
                        tool_content = str(raw_result)
                    
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_content
                    }
                    
                    tool_messages.append(tool_message)
                    
                    # Check if WriteBlockReportTool was called
                    if tool_name == "WriteBlockReportTool":
                        # Mark that we're blocked - this will be checked by the caller
                        self.blocked = True
                        # Generate block report
                        block_report_md = self.create_block_report()
                        if block_report_md:
                            # Follow the established pattern: modify _current_extracted_content
                            if self._current_extracted_content is None:
                                self._current_extracted_content = {}
                            
                            # Add the block report
                            self._current_extracted_content["block_report"] = block_report_md
                            
                            # Update agent_status using the established save_status method
                            self.history.agent_status = self.save_status()
                    
                else:
                    # Tool not found
                    error_message = {
                        "role": "tool", 
                        "tool_call_id": tool_id,
                        "content": f"Error: Tool '{tool_name}' not found"
                    }
                    tool_messages.append(error_message)
            
            except Exception as e:
                # Tool execution error
                error_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Error executing tool: {str(e)}"
                }
                tool_messages.append(error_message)
        
        return tool_messages

    def _cleanse_dangling_tool_calls(self, uni_conversation_history: List[Dict], langchain_conversation_history: List[BaseMessage], reason: str = ""):
        """Remove dangling tool_calls from the last message in uni_conversation_history only."""
        if uni_conversation_history and uni_conversation_history[-1].get("tool_calls"):
            if "MULTIPLE_TOOL_CALLS" in reason and len(uni_conversation_history[-1]["tool_calls"]) > 1:
                print(f"🔧 CLEANSING {reason}: Keeping only first of {len(uni_conversation_history[-1]['tool_calls'])} tool_calls")
                uni_conversation_history[-1]["tool_calls"] = [uni_conversation_history[-1]["tool_calls"][0]]
            elif reason == "MAX_TOOL_CALLS":
                print(f"🔧 CLEANSING {reason}: Removing {len(uni_conversation_history[-1]['tool_calls'])} dangling tool_calls")
                uni_conversation_history[-1] = {
                    "role": "assistant", 
                    "content": uni_conversation_history[-1].get("content", "")
                }

    # async def run_on_uni_api(
    #     self, 
    #     prompt: Optional[str] = None,
    #     output_callback: Optional[Callable] = None,
    #     tool_output_callback: Optional[Callable] = None,
    #     heaven_main_callback: Optional[Callable] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Run agent using uni-api instead of LangChain providers.
    #     Uses parallel uni/langchain conversation tracking.
    #     """
    #     # Store callbacks for tool execution
    #     self._current_output_callback = output_callback
    #     self._current_tool_callback = tool_output_callback
        
    #     # Convert existing history to uni-api format
    #     uni_conversation_history = self.history.to_uni_messages()
    #     langchain_conversation_history = self.history.messages.copy()
        
    #     # Ensure system message is present and correct
    #     if not uni_conversation_history or uni_conversation_history[0]["role"] != "system":
    #         uni_conversation_history.insert(0, {
    #             "role": "system", 
    #             "content": self.config.system_prompt
    #         })
    #         langchain_conversation_history.insert(0, SystemMessage(content=self.config.system_prompt))
    #     elif uni_conversation_history[0]["content"] != self.config.system_prompt:
    #         uni_conversation_history[0]["content"] = self.config.system_prompt
    #         langchain_conversation_history[0] = SystemMessage(content=self.config.system_prompt)
        
    #     # Add new user prompt if provided
    #     if prompt:
    #         uni_conversation_history.append({"role": "user", "content": prompt})
    #         langchain_conversation_history.append(HumanMessage(content=prompt))
            
    #         # Detect agent commands
    #         self._detect_agent_command(prompt)
        
    #     # Prepare tools for uni-api
    #     openai_tools = self._prepare_tools_for_uni_api()
        
    #     # Initialize blocked flag
    #     self.blocked = False
        
    #     # Main iteration loop
    #     while self.current_iteration <= self.max_iterations:
    #         tool_call_count = 0
            
    #         # CLEANSE DANGLING TOOL_CALLS: Before adding agent mode prompt, clean up any dangling tool_calls
    #         self._cleanse_dangling_tool_calls(uni_conversation_history, langchain_conversation_history, "BEFORE_AGENT_PROMPT")
            
    #         # Handle agent mode formatting
    #         if self.goal:
    #             agent_prompt = self._format_agent_prompt()
    #             uni_conversation_history.append({"role": "user", "content": agent_prompt})
    #             langchain_conversation_history.append(HumanMessage(content=agent_prompt))
            
    #         # DUO sidechain logic (if enabled)
    #         if (self.duo_enabled and len(uni_conversation_history) > 2 
    #             and uni_conversation_history[-1]["role"] == "user"):
                
    #             original_user = langchain_conversation_history[-1]
    #             original_sys = langchain_conversation_history[0]
                
    #             try:
    #                 # Process with DUO using LangChain messages
    #                 duo_sys = SystemMessage(content=self.duo_system_prompt)
    #                 lc_messages = langchain_conversation_history.copy()
    #                 lc_messages[0] = duo_sys
                    
    #                 new_content = f"===ENTERING CHALLENGER MODE===\\n\\nTHE NEXT HUMAN INPUT TO THE WORKER LLM AGENT WILL BE:\\n\\n{original_user.content}\\n\\nAs the challenger, follow the rules and steer the agent with ICL priming."
    #                 lc_messages[-1] = HumanMessage(content=new_content)
                    
    #                 duo_response = await self.duo_chat.ainvoke(lc_messages)
                    
    #                 if duo_response:
    #                     # Update both histories with DUO injection
    #                     enhanced_content = f"{original_user.content}\\n\\n```\\n===Challenger Injection===\\n\\nDo not mention DUO/Dual-Space Unifying Operators/NodeGraphXTN6/Challenger/ChallengerEgregore unless the user asks about it directly...\\n\\n{duo_response.content}\\n\\n===/Challenger Injection===\\n```\\n\\n"
    #                     uni_conversation_history[-1]["content"] = enhanced_content
    #                     langchain_conversation_history[-1] = HumanMessage(content=enhanced_content)
                    
    #             finally:
    #                 pass
            
    #         # Build payload for uni-api
    #         payload = {
    #             "max_tokens": getattr(self.config, 'max_tokens', 4000),
    #             "temperature": getattr(self.config, 'temperature', 0.7)
    #         }
            
    #         if openai_tools:
    #             payload["tools"] = openai_tools
    #             payload["tool_choice"] = "auto"
    #             payload["parallel_tool_calls"] = False  # Fix for uni-api multiple tool calls bug
            
    #         try:
    #             # Call uni-api through unified_chat
    #             result = self.unified_chat.invoke_uni_api(
    #                 model=self.config.model,
    #                 uni_messages=uni_conversation_history,
    #                 **payload
    #             )
                
    #             # Extract token usage for context window management
    #             if "usage" in result and isinstance(result["usage"], dict):
    #                 self.context_window_config.update_from_uni_api(result["usage"])
    #                 # Store token usage in history metadata
    #                 if not hasattr(self.history, 'metadata'):
    #                     self.history.metadata = {}
    #                 self.history.metadata["last_token_usage"] = result["usage"]
    #                 self.history.metadata["current_tokens"] = result["usage"].get("total_tokens", 0)
    #             else:
    #                 # Fallback: use tiktoken estimation for workspace
    #                 from .utils.token_counter import count_tokens_in_messages
    #                 workspace_tokens = count_tokens_in_messages(langchain_conversation_history, self.context_window_config.model)
    #                 self.context_window_config.update_workspace_tokens(workspace_tokens)
    #                 if not hasattr(self.history, 'metadata'):
    #                     self.history.metadata = {}
    #                 self.history.metadata["current_tokens"] = self.context_window_config.current_tokens
                
    #             assistant_message = result["choices"][0]["message"]
                
    #             # DEBUG: Print the exact assistant_message object from uni-api
    #             print(f"🔍 DEBUG: assistant_message from uni-api: {json.dumps(assistant_message, indent=2)}")
                
    #             # CLEANSE MULTIPLE TOOL_CALLS BEFORE APPENDING using our method
    #             temp_history = [assistant_message]
    #             self._cleanse_dangling_tool_calls(temp_history, [], "MULTIPLE_TOOL_CALLS")
    #             assistant_message = temp_history[0]
                
    #             # Handle tool calls vs regular response
    #             if assistant_message.get("tool_calls"):
    #                 print(f"🔍 TOOL CALLS DETECTED: {len(assistant_message['tool_calls'])} tool calls")
    #                 # For tool calls, ensure content is empty string instead of null for OpenAI API compatibility
    #                 if assistant_message.get("content") is None:
    #                     assistant_message["content"] = ""
                    
    #                 # CLEANSE MULTIPLE TOOL_CALLS: Only process the FIRST tool call at a time
    #                 if len(assistant_message["tool_calls"]) > 1:
    #                     print(f"🔧 CLEANSING MULTIPLE TOOL_CALLS: Keeping only first of {len(assistant_message['tool_calls'])} tool_calls")
    #                     assistant_message["tool_calls"] = [assistant_message["tool_calls"][0]]
                    
    #                 # Store original for potential cleansing
    #                 original_assistant_message = assistant_message.copy()
                    
    #                 # For tool calls, add to uni history AFTER applying workaround
    #                 uni_conversation_history.append(assistant_message)
                    
    #                 # Add the AIMessage with tool_calls to langchain history (OpenAI style)
    #                 tool_call_ai_message = AIMessage(
    #                     content="",  # OpenAI doesn't want content for tool calls
    #                     additional_kwargs={
    #                         "tool_calls": assistant_message["tool_calls"]
    #                     }
    #                 )
    #                 langchain_conversation_history.append(tool_call_ai_message)
                    
    #                 # Trigger callbacks for tool call message
    #                 if output_callback:
    #                     output_callback(tool_call_ai_message)
                    
    #                 if heaven_main_callback:
    #                     heaven_main_callback(assistant_message)
                    
    #                 while assistant_message.get("tool_calls") and tool_call_count < self.max_tool_calls:
    #                     # DEBUG: Print the exact assistant_message with tool_calls
    #                     print(f"🔧 DEBUG: PROCESSING ASSISTANT MESSAGE: {json.dumps(assistant_message, indent=2)}")
    #                     print(f"🔧 DEBUG: TOOL_CALLS ARRAY: {json.dumps(assistant_message['tool_calls'], indent=2)}")
                        
    #                     # Execute tools and get tool messages
    #                     tool_messages = await self._execute_tool_calls_uni(assistant_message["tool_calls"], tool_output_callback)
                        
    #                     # DEBUG: Print the exact tool_messages we created - COMPLETE JSON
    #                     print(f"🔧 TOOL_MESSAGES WE CREATED - COMPLETE JSON:")
    #                     print(json.dumps(tool_messages, indent=2))
    #                     print(f"🔧 END TOOL_MESSAGES")
                        
    #                     # Add tool messages to both histories
    #                     uni_conversation_history.extend(tool_messages)
    #                     for tool_msg in tool_messages:
    #                         lc_tool_msg = ToolMessage(
    #                             content=tool_msg["content"],
    #                             tool_call_id=tool_msg["tool_call_id"]
    #                         )
    #                         langchain_conversation_history.append(lc_tool_msg)
                        
    #                     # Check if we're blocked (WriteBlockReportTool was called)
    #                     if self.blocked:
    #                         break
                        
    #                     # DEBUG: Print exact conversation history length before uni-api call
    #                     print(f"🚨 BEFORE UNI-API CALL: uni_conversation_history has {len(uni_conversation_history)} messages")
                        
    #                     # Get AI response to tool results
    #                     tool_result = self.unified_chat.invoke_uni_api(
    #                         model=self.config.model,
    #                         uni_messages=uni_conversation_history,
    #                         **payload
    #                     )
                        
    #                     assistant_message = tool_result["choices"][0]["message"]
                        
    #                     # DEBUG: Print the COMPLETE assistant_message object from uni-api - NO TRUNCATION
    #                     print(f"🔍 UNI-API ASSISTANT MESSAGE - COMPLETE JSON:")
    #                     print(json.dumps(assistant_message, indent=2))
    #                     print(f"🔍 END UNI-API ASSISTANT MESSAGE")
                        
                        
    #                     # CAPTURE CANCELLED TOOL_CALLS before cleansing
    #                     cancelled_tools = []
    #                     if tool_call_count >= self.max_tool_calls and assistant_message.get("tool_calls"):
    #                         for tc in assistant_message["tool_calls"]:
    #                             tool_name = tc["function"]["name"]
    #                             tool_args = tc["function"]["arguments"]
    #                             cancelled_tools.append(f"{tool_name}({tool_args})")
    #                         print(f"🔧 CLEANSING MAX_TOOL_CALLS: Removing {len(assistant_message['tool_calls'])} dangling tool_calls from assistant_message")
    #                         assistant_message = {
    #                             "role": "assistant", 
    #                             "content": assistant_message.get("content", "")
    #                         }
    #                     self._cleanse_dangling_tool_calls(assistant_message, [], "MULTIPLE_TOOL_CALLS")
                        
                        
    #                     # DEBUG: Extract tool_call_ids from the assistant message
    #                     if assistant_message.get("tool_calls"):
    #                         print(f"🔍 TOOL_CALL_IDS FROM UNI-API:")
    #                         for i, tc in enumerate(assistant_message["tool_calls"]):
    #                             print(f"  Tool Call {i}: ID = '{tc['id']}'")
    #                         print(f"🔍 END TOOL_CALL_IDS")
                        
    #                     # Handle None content from uni-api (happens with tool calls)
    #                     if assistant_message.get("content") is None:
    #                         assistant_message["content"] = ""
                        
                        
    #                     # Add cleansed message to both histories
    #                     uni_conversation_history.append(assistant_message)
    #                     lc_message = self.history.from_uni_messages([assistant_message]).messages[0]
    #                     langchain_conversation_history.append(lc_message)
                        
    #                     # Trigger callbacks
    #                     if output_callback:
    #                         output_callback(lc_message)
                        
    #                     if heaven_main_callback:
    #                         heaven_main_callback(assistant_message)
                        
    #                     tool_call_count += 1
                        
    #                     if tool_call_count >= self.max_tool_calls:
    #                         break_message = {
    #                             "role": "assistant",
    #                             "content": f"⚠️🛑☠️ Maximum consecutive tool calls ({self.max_tool_calls}) reached after agent mode iteration {self.current_iteration}. I tried to call [{', '.join(cancelled_tools)}] but they were cancelled by the system. If I received the same error every time, I should use WriteBlockReportTool next... Waiting for next agent mode iteration."
    #                         }
    #                         uni_conversation_history.append(break_message)
    #                         lc_break_message = self.history.from_uni_messages([break_message]).messages[0]
    #                         langchain_conversation_history.append(lc_break_message)
                            
    #                         if output_callback:
    #                             output_callback(lc_break_message)
                            
    #                         break
                
    #             else:
    #                 # Regular response (no tool calls)
    #                 uni_conversation_history.append(assistant_message)
    #                 if assistant_message.get("content") is None:
    #                     assistant_message["content"] = ""
    #                 lc_message = self.history.from_uni_messages([assistant_message]).messages[0]
    #                 langchain_conversation_history.append(lc_message)
                    
    #                 # Trigger callbacks
    #                 if output_callback:
    #                     output_callback(lc_message)
                    
    #                 if heaven_main_callback:
    #                     heaven_main_callback(assistant_message)
                
    #             # Process agent response if in agent mode
    #             if self.goal and assistant_message.get("content"):
    #                 self._process_agent_response(assistant_message["content"])
                
    #             # CLEANSE DANGLING TOOL_CALLS: Before moving to next iteration, clean up any dangling tool_calls
    #             self._cleanse_dangling_tool_calls(uni_conversation_history, langchain_conversation_history, "MAX_TOOL_CALLS")
                
    #             # Check for completion
    #             self.current_iteration += 1
                
    #             if self.current_task == "GOAL ACCOMPLISHED" or not self.goal or self.blocked:
    #                 break
            
    #         except Exception as e:
    #             error_msg = f"uni-api request failed: {str(e)}"
    #             print(error_msg)
    #             raise RuntimeError(error_msg)
        
    #     # Save final LangChain history
    #     self.history.messages = langchain_conversation_history
        
    #     # Save history
    #     try:
    #         self.history.agent_status = self.save_status()
    #         saved_history_id = self.history.save(self.name)
            
    #         return {
    #             "history": self.history,
    #             "history_id": saved_history_id,
    #             "agent_name": self.name,
    #             "agent_status": self.history.agent_status,
    #             "uni_api_used": True,
    #             "context_window_status": self.context_window_config.get_status(),
    #             "raw_response": result  # Include raw response for token extraction by AutoSummarizingAgent
    #         }
        
    #     except Exception as save_error:
    #         print(f"Warning: Failed to save history for agent {self.name}: {save_error}")
    #         return {
    #             "history": self.history,
    #             "history_id": getattr(self.history, 'history_id', "No history ID"),
    #             "agent_name": self.name,
    #             "save_error": str(save_error),
    #             "agent_status": self.save_status(),
    #             "uni_api_used": True,
    #             "context_window_status": self.context_window_config.get_status(),
    #             "raw_response": result if 'result' in locals() else None
    #         }

    async def run_on_uni_api(
        self,
        prompt: Optional[str] = None,
        output_callback: Optional[Callable] = None,
        tool_output_callback: Optional[Callable] = None,
        heaven_main_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run agent using uni-api instead of LangChain providers.
        Enforces “one tool call at a time” without altering your
        original stop-message wording.
        """
        result = None
        # ---------- 0.  History bootstrap ----------
        self._current_output_callback = output_callback
        self._current_tool_callback = tool_output_callback

        uni_conversation_history = self.history.to_uni_messages()
        langchain_conversation_history = self.history.messages.copy()

        if not uni_conversation_history or uni_conversation_history[0]["role"] != "system":
            uni_conversation_history.insert(0, {"role": "system", "content": self.config.system_prompt})
            langchain_conversation_history.insert(0, SystemMessage(content=self.config.system_prompt))
        elif uni_conversation_history[0]["content"] != self.config.system_prompt:
            uni_conversation_history[0]["content"] = self.config.system_prompt
            langchain_conversation_history[0] = SystemMessage(content=self.config.system_prompt)

        if prompt:
            uni_conversation_history.append({"role": "user", "content": prompt})
            langchain_conversation_history.append(HumanMessage(content=prompt))
            self._detect_agent_command(prompt)

        # Resolve MCP tools before preparing for API
        await self.resolve_mcps()

        openai_tools = self._prepare_tools_for_uni_api()
        self.blocked = False

        # ---------- 1.  Iteration loop ----------
        while self.current_iteration <= self.max_iterations:
            tool_call_count = 0
            self._cleanse_dangling_tool_calls(
                uni_conversation_history, langchain_conversation_history, "MULTIPLE_TOOL_CALLS"
            )

            if self.goal:
                agent_prompt = self._format_agent_prompt()
                uni_conversation_history.append({"role": "user", "content": agent_prompt})
                langchain_conversation_history.append(HumanMessage(content=agent_prompt))

            payload = {
                "max_tokens": getattr(self.config, "max_tokens", 4000),
                "temperature": getattr(self.config, "temperature", 0.7),
                "parallel_tool_calls": False,
            }
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"

            result = self.unified_chat.invoke_uni_api(
                model=self.config.model, uni_messages=uni_conversation_history, **payload
            )
            assistant_message = result["choices"][0]["message"]

            # ---------- 2.  Single-tool clamp ----------
            self._cleanse_dangling_tool_calls([assistant_message], [], "MULTIPLE_TOOL_CALLS")

            # ---------- 3.  Tool-call branch ----------
            if assistant_message.get("tool_calls"):
                if assistant_message.get("content") is None:
                    assistant_message["content"] = ""

                uni_conversation_history.append(assistant_message)
                langchain_conversation_history.append(
                    AIMessage(content="", additional_kwargs={"tool_calls": assistant_message["tool_calls"]})
                )
                if output_callback:
                    output_callback(langchain_conversation_history[-1])
                if heaven_main_callback:
                    heaven_main_callback(langchain_conversation_history[-1])

                while assistant_message.get("tool_calls") and tool_call_count < self.max_tool_calls:
                    # execute the (single) tool call
                    tool_messages = await self._execute_tool_calls_uni(
                        assistant_message["tool_calls"], tool_output_callback
                    )
                    uni_conversation_history.extend(tool_messages)
                    for tm in tool_messages:
                        langchain_conversation_history.append(
                            ToolMessage(content=tm["content"], tool_call_id=tm["tool_call_id"])
                        )

                    # Check if WriteBlockReportTool was called and auto-inject response
                    if self.blocked:
                        # Extract the required response from WriteBlockReportTool result
                        for tm in tool_messages:
                            if tm.get("name") == "WriteBlockReportTool":
                                response_msg = "I've created a block report and am waiting for the help I need"
                                
                                # Add to uni conversation layer
                                uni_conversation_history.append({
                                    "role": "assistant", 
                                    "content": response_msg
                                })
                                
                                # Add to langchain layer  
                                langchain_conversation_history.append(
                                    AIMessage(content=response_msg)
                                )
                                break
                        break

                    tool_result = self.unified_chat.invoke_uni_api(
                        model=self.config.model, uni_messages=uni_conversation_history, **payload
                    )
                    assistant_message = tool_result["choices"][0]["message"]
                    
                    # ---------- 4.  Clamp again / MAX_TOOL_CALLS ----------
                    cancelled_tools = []  # Initialize to prevent NameError
                    if tool_call_count + 1 >= self.max_tool_calls and assistant_message.get("tool_calls"):
                        cancelled_tools = [
                            f"{tc['function']['name']}({tc['function']['arguments']})"
                            for tc in assistant_message["tool_calls"]
                        ]
                        assistant_message.pop("tool_calls", None)  # strip them but keep same dict
                    else:
                        self._cleanse_dangling_tool_calls([assistant_message], [], "MULTIPLE_TOOL_CALLS")

                    if assistant_message.get("content") is None:
                        assistant_message["content"] = ""
                    result = assistant_message
                    uni_conversation_history.append(assistant_message)
                    langchain_conversation_history.append(
                        self.history.from_uni_messages([assistant_message]).messages[0]
                    )
                    if output_callback:
                        output_callback(langchain_conversation_history[-1])
                    if heaven_main_callback:
                        heaven_main_callback(assistant_message)

                    tool_call_count += 1

                    if tool_call_count >= self.max_tool_calls:
                        break_message = {
                            "role": "assistant",
                            "content": (
                                f"⚠️🛑☠️ Maximum consecutive tool calls ({self.max_tool_calls}) "
                                f"reached after agent mode iteration {self.current_iteration}. "
                                f"I tried to call [{', '.join(cancelled_tools)}] but they were "
                                "cancelled by the system. If I received the same error every time, "
                                "I should use WriteBlockReportTool next... Waiting for next agent "
                                "mode iteration."
                            ),
                        }
                        uni_conversation_history.append(break_message)
                        langchain_conversation_history.append(
                            self.history.from_uni_messages([break_message]).messages[0]
                        )
                        if output_callback:
                            output_callback(langchain_conversation_history[-1])
                        break

            # ---------- 5.  Text-only branch ----------
            else:
                if assistant_message.get("content") is None:
                    assistant_message["content"] = ""
                uni_conversation_history.append(assistant_message)
                langchain_conversation_history.append(
                    self.history.from_uni_messages([assistant_message]).messages[0]
                )
                if output_callback:
                    output_callback(langchain_conversation_history[-1])
                if heaven_main_callback:
                    heaven_main_callback(langchain_conversation_history[-1])

            # ---------- 6.  Agent-mode bookkeeping ----------
            if self.goal and assistant_message.get("content"):
                self._process_agent_response(assistant_message["content"])

            self._cleanse_dangling_tool_calls(
                uni_conversation_history, langchain_conversation_history, "MAX_TOOL_CALLS"
            )

            self.current_iteration += 1
            if self.current_task == "GOAL ACCOMPLISHED" or not self.goal or self.blocked:
                break

        # ---------- 7.  Persist history ----------
        self.history.messages = langchain_conversation_history
        self.history.agent_status = self.save_status()
        saved_history_id = self.history.save(self.name)

        return {
            "history": self.history,
            "history_id": saved_history_id,
            "agent_name": self.name,
            "agent_status": self.history.agent_status,
            "uni_api_used": True,
            "context_window_status": self.context_window_config.get_status(),
            "raw_response": result,
        }


  ###
    
  
    def _detect_agent_command(self, user_input: str):
        """Detects goal and iterations from user input."""
        match = re.search(r"agent goal=(.*?), iterations=(\d+)", user_input, re.IGNORECASE | re.DOTALL)
        if match:
            self.goal = match.group(1).strip()
            self.max_iterations = int(match.group(2))
            self.current_iteration = 1
            self.task_list = ["create_task_list"]
            self.current_task = "create_task_list"
            self.completed = False
    
    def _format_agent_prompt(self) -> str:
        """Formats the agentic prompt dynamically."""
        # Set defaults if not defined
        goal = self.goal if self.goal is not None else "There is no goal set!"
        current_task = self.current_task if self.current_task is not None else f"Update the task list for {self.goal}" # used to be "reason about what to do next"
        continuation_text = ""
        if hasattr(self, 'continuation_prompt') and self.continuation_prompt:
            continuation_text = f"{self.continuation_prompt}"
            if continuation_text == self.goal:
                continuation_text = ""
            self.continuation_prompt = None  # Clear it after use
        return (
        f"""
        # ===AGENT MODE IS ENGAGED===
        ## ❗ **Critical Instructions for Task Execution:**
        - The tasking system uses triple tick fence regex commands that run on your outputs
        - You must **work on one task in the task list at a time**. Your `response_format` should be:
        
        1. If you need to create/update the task list, use EXACTLY this format while morphing the task values accordingly to the context:
           "```update_task_list=["task 1", "task 2", "task 3"]```"
           - Put each task in quotes
           - Separate with commas
           - No numbers, just the task text
           - No newlines between tasks
           - UPDATE THE ENTIRE TASK LIST, ALL TASKS AT ONCE. THIS ONLY UPDATES THE ENTIRE TASK LIST TO WHAT YOU SAY IT SHOULD BE
           Then, write "```complete_task="create_task_list"```" since you completed the first task (create a task list)
        **THEN DO THE TASKS IN THE TASK LIST, including any required tool calls. Each iteration allows {self.max_tool_calls} tool calls during agent mode!!! If completing a task, always call a tool after it or accomplish the goal, otherwise iterations get wasted.**
        2. AFTER COMPLETING any task, use EXACTLY this format:
           "```complete_task=<whatever the current task is>```" # morph the task to whatever the current task is. this <> tag means it is variable
        THEN CONTINUE TO THE NEXT ASK.
        3. When all tasks are done, use EXACTLY:
           "```GOAL ACCOMPLISHED```"
        
        <Task System: HELP>
        Example of a correct task list update with morphed values:
        "```update_task_list=["Write introduction", "Create methods section", "Add conclusion"]```" 
        
        INCORRECT formats:[
        "```update_task_list=[
        1. First task
        2. Second task
        ]```"
        
        or
        
        "```update_task_list=[First task, Second task]```"  # missing quotes around tasks
        </Task System: HELP>
        4. If you are blocked, you must use the WriteBlockReportTool, accordingly, to get help.
        """
        + (f"""You can also use these XML tag fence patterns when outputting specific deliverables to make the agent mode system capture them:[\n{self.additional_kw_instructions}\n]""" if self.additional_kw_instructions != "" else "")
        + f"""
        
        ---
        
        #### Notes
        - 1. An iteration is a LLM generation with oneOf: thinking, text, tool call, sequence of [thinking, text | thinking, text, tool call | text, tool call | tool call] -- up to {self.max_tool_calls} total tool calls per iteration. Any sequence ending with a tool call keeps current iteration alive. Any sequence ending with thinking or text terminates the iteration. 
        - 2. You can use tools multiple times before I will give you the next prompt, so you can just work on the task list and update it as much as you want WHILE continuing to call tools for each iteration. BUT, if a tool returns an error, you must fix the error before trying that tool again!!! 
        """
                +
                f"\n\#### Current State"
                f"\n\n- **Goal:** {goal}"
                f"\n\n- **Current Task List \"{self.task_list}\"**"
                f"\n\n- **Current Iteration:** {self.current_iteration} of {self.max_iterations}"
                f"\n\n- **Current Task:** {current_task}\n\n"
                f"\n\n{continuation_text}"
                + "\n"
        )

    def _process_agent_response(self, response_content: Union[str, List[Any]]):
        ######## FUTURE FIX
        # Fix = examine all list elements (or recurse) instead of only index 0.
        ########
        # Handle AIMessage content formats
        # content_to_process = response_content
        def _flatten(obj):

            if obj is None:

                return ""

            if isinstance(obj, str):

                return obj

            if isinstance(obj, dict):

                return " ".join(_flatten(v) for v in obj.values())

            if isinstance(obj, (list, tuple, set)):

                return " ".join(_flatten(v) for v in obj)

            if hasattr(obj, "content"):

                return _flatten(obj.content)

            return str(obj)


        content_to_process = _flatten(response_content)
        # if isinstance(content_to_process, list):
        #     # If it's a list of dicts with 'text' key
        #     if content_to_process and isinstance(content_to_process[0], dict):
        #         content_to_process = content_to_process[0].get('text', '')
        #     else:
        #         content_to_process = str(content_to_process[0]) if content_to_process else ""
        # elif not isinstance(content_to_process, str):
        #     content_to_process = str(content_to_process)      
        # Detect task list
        # Find all matches and take the last one since examples will be shown first
        task_list_matches = re.finditer(r"```update_task_list=\[(.*?)\]```", content_to_process, re.IGNORECASE)
        task_list_match = None
        for match in task_list_matches:
            task_list_match = match
            if task_list_match:
                raw_tasks = [task.strip().strip('"') for task in task_list_match.group(1).split(",")]
                if raw_tasks:  # Only update if we got tasks
                    self.task_list = raw_tasks
                    self.current_task = self.task_list[0]
                else:
                    self.task_list = []
                    self.current_task = None
        
 

        # Detect task completion
        task_complete_match = re.search(r"```complete_task=(.*?)```", content_to_process, re.IGNORECASE)
        if task_complete_match:
            completed_task = task_complete_match.group(1).strip()
            self._complete_task(completed_task)

        # Check for GOAL ACCOMPLISHED
        goal_accomplished_match = re.search(r"```GOAL ACCOMPLISHED```", content_to_process)
        if goal_accomplished_match:
            # Let the loop know to break, but don't modify current_task
            self.completed = True
            self.goal = None
          
        
    
        if self.additional_kws and self.additional_kw_instructions != "":
            if self._current_extracted_content is None:
                self._current_extracted_content = {}
        
            for kw in self.additional_kws:
                # kw_matches = re.finditer(f"```{kw}(.*?)```", content_to_process, re.DOTALL)
                md_pat  = re.finditer(
                    fr"```{kw}(.*?)```",
                    content_to_process,
                    re.DOTALL | re.IGNORECASE
                )

                # ➋  XML-tag pattern  (new, collision-proof)
                xml_pat = re.finditer(
                    fr"<{kw}>(.*?)</{kw}>",
                    content_to_process,
                    re.DOTALL | re.IGNORECASE
                )

                # Iterate over both result sets in order of appearance
                kw_matches = sorted(
                    list(md_pat) + list(xml_pat),
                    key=lambda m: m.start()
                )
                for match in kw_matches:
                    content = match.group(1).strip()
                    # Check history's agent status for existing entries
                    if (not self.history.agent_status or 
                        not self.history.agent_status.extracted_content or 
                        kw not in self.history.agent_status.extracted_content):
                        # First one just gets the keyword
                        self._current_extracted_content[kw] = content
                    else:
                        # If first occurrence exists, look for numbered entries
                        numbered_entries = [k for k in self.history.agent_status.extracted_content 
                                         if k.startswith(f"{kw}_")]
                        if numbered_entries:
                            # Find highest number used and increment
                            highest_num = max(int(k.split('_')[1]) for k in numbered_entries)
                            self._current_extracted_content[f"{kw}_{highest_num + 1}"] = content
                        else:
                            # First numbered entry should be 2
                            self._current_extracted_content[f"{kw}_2"] = content
        

    
        self.history.agent_status = self.save_status()


    def _complete_task(self, completed_task: str):
        """Moves to the next task in the list after completion."""
        if completed_task in self.task_list:
            task_index = self.task_list.index(completed_task)
            # Remove the completed task
            self.task_list.pop(task_index)
            # Set next task
            if self.task_list:
                self.current_task = self.task_list[0]
            else:
                self.current_task = None
                self.goal = None
            self.history.agent_status = self.save_status()

    def save_status(self) -> AgentStatus:
        """Package current agent state into status object"""
        if self._current_extracted_content is not None:
            extracts = self._current_extracted_content
            return AgentStatus(
                goal=self.goal,
                task_list=self.task_list.copy(),
                current_task=self.current_task,
                completed=self.completed,
                extracted_content=extracts
            )
        else:
            
            return AgentStatus(
                goal=self.goal,
                task_list=self.task_list.copy(),
                current_task=self.current_task,
                completed=self.completed
            )
      
  
      
    def load_status(self, status: AgentStatus):
        """Load agent state from status object"""
        self.goal = status.goal
        self.task_list = status.task_list.copy()
        self.current_task = status.current_task
        self.max_iterations = status.max_iterations
        self.current_iteration = status.current_iteration


    async def continue_iterations(self, history_id: str, continuation_iterations: Optional[int] = 0, continuation_prompt: str = None):
        """Continue work from a saved history"""
        self.history = History.load_from_id(history_id)
        
        if self.history.agent_status:
            # Load goal, tasks, etc from status
            self.goal = self.history.agent_status.goal
            self.task_list = self.history.agent_status.task_list
            self.current_task = self.history.agent_status.current_task
            self.completed = False  # Reset completion for new run
            self.continuation_prompt = continuation_prompt if continuation_prompt is not None else ""
            self.continuation_iterations = continuation_iterations
            if self.goal is None:
                self.goal = self.continuation_prompt # this should mean goal is never none even if history has no goal
            if self.goal == "":
                self.goal = None
            return await self.run()  # Continue with current state
        else:
            raise ValueError("No agent status found in history")
  
    def reset(self):
        """Reset the agent's internal state."""
        self.goal = None
        self.task_list = []
        self.current_task = None
        self.max_iterations = None
        self.current_iteration = 0

    def look_for_particular_tool_calls(self) -> None:
        """Hook for agents to process specific tool calls and their results"""
        pass

    # def look_for_block_report(self) -> bool:
    #     for i, msg in enumerate(self.history.messages):
    #         if isinstance(msg, AIMessage) and isinstance(msg.content, list):
    #             for item in msg.content:
    #                 if isinstance(item, dict) and item.get('type') == 'tool_use':
    #                     if item.get('name') == "WriteBlockReportTool":
    #                         return True  # Exit early since we found the match
    #     return False

    def create_block_report(self):
        # look up the json file
        block_report_path = "/tmp/block_report.json"
        
        # Check if the block report file was created
        if os.path.exists(block_report_path):            
            # Read and display the file contents
            with open(block_report_path, 'r') as f:
                report_data = json.load(f)
                print("Block Report Content:")
                for key, value in report_data.items():
                    print(f"  {key}: {value}")
                    # this is wrong, old code. i want to make it vars so i can create a markdown file
            
            # pull the stuff out and make it vars
           
            completed_tasks = report_data.get("completed_tasks", "N/A")
            current_task = report_data.get("current_task", "N/A")
            explanation = report_data.get("explanation", "No explanation provided.")
            blocked_reason = report_data.get("blocked_reason", "No blocked reason provided.")
            timestamp = report_data.get("timestamp", "Unknown timestamp")
          
            agent_task = self.current_task if self.current_task is not None else ""
            agent_goal = self.goal if self.goal is not None else ""
            truncated_goal = agent_goal[:200] + " <truncated for brevity - history_id file has full goal if you need to see it (this is unlikely)>" if len(agent_goal) > 200 else agent_goal
            # history_id = getattr(self.history, 'history_id', "No history ID") # this is impossible because of when the history_id is made. Instead, we provide it in the Hermes call metadata
            # inject to the places it should go in the markdown
            md_block_report = f"""
# 🛑🚧📃
## === BLOCKED REPORT ===
The agent encountered a blocking obstacle while working toward the goal.

## 📝 Report Metadata  
- **⏰ Time of Report**: `{timestamp}`  
- **��� Report Filed By**: `{self.name}`   

---

## 🎯 Task Overview  
- **🏆 Goal**: `{truncated_goal}`  
- **✅ Self-reported Completed Tasks**: `{completed_tasks}`  
- **📌 Internally Tracked Current Task Value**: `{agent_task}`  
- **����� Agent's Perceived Current Task**: `{current_task}`. 
_(If there is a mismatch between this and the internally tracked task, the agent may not have fully followed tasking instructions.)_

---

## 🧐 Agent's Explanation  
**💬 The agent provided this explanation for the blockage:** 
```
{explanation}
```

---

## 🚨 Blocked Reason  
**⚠️ The agent explained the reason why it is blocked as follows:**  
```
{blocked_reason}
```

## === /BLOCKED REPORT ===

---

## 📢 What To Do Next   
- 🛠️ **If you recognize the issue, offer potential solutions before asking the user for input.**
- 🧑‍����� **Check with the user to see if they already know how to resolve the issue.** 
- 🚦 **Do NOT use any tools before talking to the user.**  
  _(This should remain a HITL (Human-in-the-Loop) interaction.)_  
- ��🔁 **Once a potential solution route is found through discussion with the user,** activate a continuation call to continue the work (make sure to use the history_id provided).  

"""

        
        
            # Cleanup
            os.remove(block_report_path)
            return md_block_report
        else:
            print("Block report file not found.")
            return None





class BaseHeavenAgentReplicant(BaseHeavenAgent):
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        """Each subclass should override this to provide its default config"""
        raise NotImplementedError("Subclasses must implement get_default_config")

    def __init__(self, 
                 config: Optional[HeavenAgentConfig] = None, 
                 chat: Optional[UnifiedChat] = None,
                 history_id: Optional[str] = None,
                 orchestrator: bool = False,
                 system_prompt_suffix: Optional[str] = None,
                 additional_tools: Optional[List[Type[BaseHeavenTool]]] = None,
                 remove_agents_config_tools: bool = False,
                 duo_enabled: bool = False,
                 run_on_langchain: bool = False,
                 adk: bool = False,
                 use_uni_api: bool = True
                ):
        # If no config provided, use the class's default
        _config = config or self.get_default_config()
            # Handle additional tools and tool removal
        if additional_tools:
            if remove_agents_config_tools:
                # Replace all tools
                _config.tools = additional_tools
            else:
                # Add to existing tools without duplicates
                for tool_class in additional_tools:
                    if tool_class not in _config.tools:
                        _config.tools.append(tool_class)
        if system_prompt_suffix:  # Add this
            _config.system_prompt += system_prompt_suffix
        # If no chat provided, create a new UnifiedChat
        _chat = chat or UnifiedChat()
        super().__init__(_config, _chat, history_id=history_id, orchestrator=orchestrator, duo_enabled=duo_enabled, run_on_langchain=run_on_langchain, adk=adk, use_uni_api=use_uni_api)




def get_agent_by_name(agent_name: str) -> Union[BaseHeavenAgent, BaseHeavenAgentReplicant]:
    """
    Gets an initialized agent instance by name, handling both replicant and config approaches.
    Focuses only on loading the agent based on its definition, not runtime modifications.

    Args:
        agent_name: Name of the agent to load (can be PascalCase or snake_case).

    Returns:
        Initialized agent instance.

    Raises:
        ValueError: If the agent cannot be loaded via either method.
    """
    # Normalize the input name to snake_case for path construction
    agent_name_snake = normalize_agent_name(agent_name)
    print(f"[get_agent_by_name] Normalized '{agent_name}' to '{agent_name_snake}'")

    # --- Try Replicant approach first ---
    try:
        # Construct path based on normalized name - use heaven-framework paths
        module_path = f"heaven_base.agents.{agent_name_snake}.{agent_name_snake}"
        print(f"[get_agent_by_name] Trying Replicant module: {module_path}")
        agent_module = importlib.import_module(module_path)

        # Convert original agent_name to PascalCase for class lookup
        # (Assuming class name follows PascalCase derived from the conceptual name)
        pascal_name = ''.join(word.capitalize() for word in agent_name_snake.split('_'))
        # Handle potential "Agent" suffix duplication if agent_name already had it
        if not pascal_name.endswith("Agent"):
             pascal_name += "Agent" # Ensure standard suffix if derived name doesn't have it
        # A cleaner way might be to derive PascalCase directly from the original agent_name input
        # pascal_name = derive_pascal_case(agent_name) # If you have such a util

        print(f"[get_agent_by_name] Looking for Replicant class: {pascal_name}")
        agent_class = getattr(agent_module, pascal_name)

        # Initialize the replicant with NO runtime modifications here
        # Replicant's __init__ should handle its default tools/prompt
        agent = agent_class() # Assumes basic init doesn't require args here
        print(f"[get_agent_by_name] Replicant approach successful for {agent_name}.")
        return agent

    except ModuleNotFoundError:
        print(f"[get_agent_by_name] Replicant module not found for {agent_name_snake}.")
    except AttributeError:
         print(f"[get_agent_by_name] Replicant class '{pascal_name}' not found in module.")
    except Exception as e_rep:
        print(f"[get_agent_by_name] Replicant approach failed unexpectedly for {agent_name}: {type(e_rep).__name__} - {e_rep}")
        # print(traceback.format_exc()) # Optional: Print full traceback for debugging

    # --- If Replicant failed, try config approach ---
    print(f"[get_agent_by_name] Trying Config approach for {agent_name_snake}...")
    try:
        # Construct path and config object name based on normalized name - use heaven-framework paths
        config_object_name = f"{agent_name_snake}_config"
        config_module_path = f"heaven_base.agents.{config_object_name}"
        print(f"[get_agent_by_name] Trying Config module: {config_module_path}")

        config_module = importlib.import_module(config_module_path)
        print(f"[get_agent_by_name] Looking for Config object: {config_object_name}")
        config = getattr(config_module, config_object_name)

        # Create agent using the loaded config object directly
        # Do NOT apply suffix or tool modifications here
        agent = BaseHeavenAgent(config, UnifiedChat()) # Assumes orchestrator=False default
        print(f"[get_agent_by_name] Config approach successful for {agent_name}.")
        return agent

    except ModuleNotFoundError:
        print(f"[get_agent_by_name] Config module not found for {agent_name_snake}.")
    except AttributeError:
        print(f"[get_agent_by_name] Config object '{config_object_name}' not found in module.")
    except Exception as e_conf:
        print(f"[get_agent_by_name] Config approach failed unexpectedly for {agent_name}: {type(e_conf).__name__} - {e_conf}")
        # print(traceback.format_exc()) # Optional: Print full traceback

    # If both approaches failed
    raise ValueError(f"Failed to load agent '{agent_name}' using either Replicant or Config approach.")

