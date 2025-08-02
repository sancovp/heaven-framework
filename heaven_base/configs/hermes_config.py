# hermes_config.py
import json
from dataclasses import dataclass, field
from pydantic import Field, BaseModel
from typing import Optional, List, Dict, Any, Union, TypedDict, Type
from collections.abc import Callable
from .base_config import BaseFunctionConfig


class HermesConfig(BaseFunctionConfig):
    """
    A configuration class for defining a Hermes execution.
    It must define func_name and args_template as required by BaseFunctionConfig
    """
    func_name: str = "use_hermes"  # The function this config is FOR, for metadata purposes, not Callable
    args_template: Dict[str, Any] = Field(
        default_factory=lambda: {
            "goal": "",
            "iterations": 1,
            "agent": None,
            "history_id": None,
            "return_summary": False,
            "ai_messages_only": True,
            "continuation": None,
            "additional_tools": None,
            "remove_agents_config_tools": False,
            "orchestration_preprocess": False,
            "variable_inputs": {},
            "system_prompt_suffix": None
        }
    )
  
    def to_command_data(self, variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None) -> Dict[str, Any]:
        # print("\n=== START TO_COMMAND_DATA ===")
        # print(f"Initial args_template: {json.dumps(self.args_template, indent=2)}")
        # print(f"Received variable_inputs: {json.dumps(variable_inputs, indent=2)}")
    
        command_data = self.args_template.copy()
        #### TESTING THIS
        # Apply any custom args_template values that differ from defaults
        # Apply any custom args_template values that differ from defaults
        for key, value in self.args_template.items():
            if value != command_data.get(key):
                command_data[key] = value
        
        # print(f"After applying custom args: {json.dumps(command_data, indent=2)}")
        ####
        if not variable_inputs:
            print("No variable_inputs provided, returning default command_data")
            return command_data
        
        # Get the template configuration
        template_config = command_data["variable_inputs"]
        # print(f"\nTemplate config from args_template: {json.dumps(template_config, indent=2)}")
    
        # Apply the provided values to their respective parameters
        for param_name, param_config in template_config.items():
            # print(f"\nProcessing parameter: {param_name}")
            # print(f"Parameter config: {json.dumps(param_config, indent=2)}")
    
            if not param_config.get("template"):
                print(f"Skipping {param_name} - not templated")
                continue
    
            if param_name not in variable_inputs:
                print(f"Skipping {param_name} - not in variable_inputs")
                continue
    
            # print(f"Value from variable_inputs: {json.dumps(variable_inputs[param_name], indent=2)}")
    
            # Special handling for goal (or any parameter with variables)
            if "variables" in param_config:
                # print(f"Processing as variable template")
                # print(f"Before: {command_data[param_name]}")
                command_data[param_name] = command_data[param_name].format(**variable_inputs[param_name])
                # print(f"After: {command_data[param_name]}")
            else:
                # print(f"Processing as direct value")
                # print(f"Before: {command_data[param_name]}")
                command_data[param_name] = variable_inputs[param_name]
                # print(f"After: {command_data[param_name]}")
    
        # print(f"\nFinal command_data: {json.dumps(command_data, indent=2)}")
        # print("=== END TO_COMMAND_DATA ===\n")
        return command_data






class HermesConfigInput(BaseModel):
    """Represents input mapping for a Hermes configuration"""
    source_key: str  # Key to extract from previous result
    transform: Optional[Callable] = None  # Optional transformation function
    required: bool = True  # Whether this input is mandatory

class DovetailModel(BaseModel):
    """Defines how to chain Hermes configurations"""
    expected_outputs: List[str]
    input_map: Dict[str, HermesConfigInput]
    
    def prepare_next_config(self, previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for the next Hermes configuration"""
        next_config_inputs = {}
        
        for config_key, input_spec in self.input_map.items():
            # Extract source value
            source_value = previous_result.get(input_spec.source_key)
            
            # Apply optional transformation
            if input_spec.transform:
                source_value = input_spec.transform(source_value)
            
            # Check required inputs
            if input_spec.required and source_value is None:
                raise ValueError(f"Required input {config_key} not found")
            
            next_config_inputs[config_key] = source_value
        
        return next_config_inputs





# Here’s a sketch of two approaches:

# A plain Python “runner” class


# class HermesChainRunner:

#     def __init__(self, base_config: HermesConfig, dovetail: DovetailModel, max_steps: int = 5):

#         self.config = base_config

#         self.dovetail = dovetail

#         self.max_steps = max_steps

#         self.last_result = None

#         self.last_extracts = {}


#     def run(self):

#         from computer_use_demo.tools.base.tool_utils.hermes_utils import hermes_step


#         for i in range(self.max_steps):

#             result = hermes_step(hermes_config=self.config)

#             status = result.get("agent_status")

#             extracts = getattr(status, "extracted_content", {}) or {}

#             self.last_result, self.last_extracts = result, extracts


#             # stop when all expected keys are present

#             if all(k in extracts for k in self.dovetail.expected_outputs):

#                 break


#             next_args = self.dovetail.prepare_next_config(extracts)

#             self.config.args_template.update(next_args)


#         return self.last_result, self.last_extracts
# A Pydantic-driven orchestrator model


# from pydantic import BaseModel, Field


# class HermesChainConfig(BaseModel):

#     base_config: HermesConfig

#     dovetail: DovetailModel

#     max_steps: int = Field(5, gt=0)


# class HermesChain(BaseModel):

#     cfg: HermesChainConfig


#     def run(self):

#         from computer_use_demo.tools.base.tool_utils.hermes_utils import hermes_step


#         config = self.cfg.base_config

#         for _ in range(self.cfg.max_steps):

#             result = hermes_step(hermes_config=config)

#             status = result.get("agent_status")

#             extracts = getattr(status, "extracted_content", {}) or {}


#             if all(k in extracts for k in self.cfg.dovetail.expected_outputs):

#                 return result, extracts


#             next_args = self.cfg.dovetail.prepare_next_config(extracts)

#             config.args_template.update(next_args)


#         return result, extracts
# Why this helps:

# • Encapsulation—you can instantiate once with your JSON or Python HermesConfig + DovetailModel and just call .run().

# • Validation—Pydantic will catch missing or mis-typed fields before you ever start the loop.

# • Reusability—you can register this as a tool or import it wherever you need a multi-step Hermes chain.

# Which pattern makes more sense depends on how strictly you want schema-validation (go Pydantic) versus simplicity (plain class).







# OLD

# BASIC IDEA

# from langchain_core.tools import tool

# @tool
# def hermes_sequential_chain_tool(initial_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Execute a sequential Hermes chain as a tool"""
#     hermes_sequential_chain = SequentialChain(
#         input_variables=['initial_data'],
#         chains=[
#             RunnableLambda(
#                 lambda x: use_hermes_dict(
#                     hermes_config=config1, 
#                     variable_inputs=x
#                 )
#             ),
#             RunnableLambda(
#                 lambda prev_result: use_hermes_dict(
#                     hermes_config=config2,
#                     variable_inputs=prev_result
#                 )
#             )
#         ]
#     )
    
#     # Synchronous invocation for tools
#     result = hermes_sequential_chain.invoke(initial_data)
#     return result

# # If you need async support
# @tool
# async def async_hermes_sequential_chain_tool(initial_data: Dict[str, Any]) -> Dict[str, Any]:
#     hermes_sequential_chain = SequentialChain(...)
#     result = await hermes_sequential_chain.ainvoke(initial_data)
#     return result

# every chain function will look like that (basically). it will have a chain type that gets set up then executed and return result, and this function will be a tool, so then agents can call it


    # def to_command_data(self, variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None) -> Dict[str, Any]:
    #     """
    #     Prepare the final command data, handling any templated parameters according to their configuration
    #     Args:
    #         variable_inputs: Dictionary of values to use for templated parameters
    #     """
    #     try:
    #         with open('/tmp/hermes_debug.log', 'a') as f:
    #             f.write("\nDEBUG to_command_data:")
    #             f.write(f"\nargs_template: {json.dumps(self.args_template, indent=2)}")
    #             f.write(f"\nvariable_inputs: {json.dumps(variable_inputs, indent=2)}")
    #     except Exception as e:
    #         print(f"Debug logging failed: {e}")
        
    #     command_data = self.args_template.copy()
        
    #     if not variable_inputs:
    #         return command_data
        
    #     # Get the template configuration
    #     template_config = command_data["variable_inputs"]
        
    #     # Apply the provided values to their respective parameters
    #     for param_name, param_config in template_config.items():
    #         if not param_config.get("template"):
    #             continue
            
    #         if param_name not in variable_inputs:
    #             continue
            
    #         # Special handling for goal (or any parameter with variables)
    #         if "variables" in param_config:
    #             command_data[param_name] = command_data[param_name].format(**variable_inputs[param_name])
    #         else:
    #             # Direct copy for all other templated parameters
    #             command_data[param_name] = variable_inputs[param_name]
        
    #     try:
    #         with open('/tmp/hermes_debug.log', 'a') as f:
    #             f.write(f"\nFinal command_data: {json.dumps(command_data, indent=2)}\n")
    #     except Exception as e:
    #         print(f"Debug logging failed: {e}")
        
    #     return command_data

    # def to_command_data(self, variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None) -> Dict[str, Any]:
    #     """
    #     Prepare the final command data, handling any templated parameters according to their configuration
    #     Args:
    #         variable_inputs: Dictionary of values to use for templated parameters
    #     """
    #     try:
    #         with open('/tmp/hermes_debug.log', 'a') as f:
    #             f.write("\nDEBUG to_command_data:")
    #             f.write(f"\nargs_template: {json.dumps(self.args_template, indent=2)}")
    #             f.write(f"\nvariable_inputs: {json.dumps(variable_inputs, indent=2)}")
    #     except Exception as e:
    #         print(f"Debug logging failed: {e}")
    
    #     command_data = self.args_template.copy()
    
    #     if not variable_inputs:
    #         return command_data
    
    #     # Get the template configuration - this is our schema for what can be templated
    #     template_config = command_data["variable_inputs"]
    
    #     # Now apply the provided values to their respective parameters
    #     for param_name, param_config in template_config.items():
    #         if not param_config.get("template"):
    #             continue
    
    #         # Handle string templates (like goal)
    #         if "variables" in param_config:
    #             # Format the parameter using the provided values
    #             command_data[param_name] = command_data[param_name].format(**variable_inputs[param_name])
    
    #         # Handle list templates
    #         elif param_config.get("type") == "list":
    #             command_data[param_name] = variable_inputs[param_name]
    
    #         # Handle direct value templates
    #         elif "value" in param_config:
    #             command_data[param_name] = variable_inputs[param_name]
    
    #     try:
    #         with open('/tmp/hermes_debug.log', 'a') as f:
    #             f.write(f"\nFinal command_data: {json.dumps(command_data, indent=2)}\n")
    #     except Exception as e:
    #         print(f"Debug logging failed: {e}")
    
    #     return command_data


