# hermes_utils.py



import json
import base64
import os
import docker
import importlib
import traceback
from pathlib import Path

from typing import Dict, Any, Optional, List, Type, Union, Callable
from docker.errors import DockerException
from computer_use_demo.tools.base.baseheaventool import BaseHeavenTool, ToolArgsSchema, ToolResult
from computer_use_demo.tools.base.baseheavenagent import BaseHeavenAgent, BaseHeavenAgentReplicant
from computer_use_demo.tools.base.unified_chat import UnifiedChat
from computer_use_demo.tools.base.configs.hermes_config import HermesConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
# from computer_use_demo.tools.base.tools import *  
from computer_use_demo.tools.base.agents import *

def message_to_dict(msg):
    if isinstance(msg, (SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage)):
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        }
    return msg
  
def exec_agent_run_via_docker(
    target_container: str,
    goal: str,
    iterations: int,
    agent: Optional[str] = None,
    source_container: str = "mind_of_god",
    history_id: str = None,
    return_summary: bool = False,
    ai_messages_only: bool = False,
    remove_agents_config_tools: bool = False,
    orchestration_preprocess: bool = False,
    continuation: bool = None,
    additional_tools: Optional[List[str]] = None,
    system_prompt_suffix: Optional[str] = None
    
) -> str:
    """
    Execute an agent command in a Docker container via the Docker SDK.
    The command data is safely serialized and embedded in the inline script.
    """
    client = docker.from_env()
    # username = "computeruse"
    # if target_container in ["mind_of_god", "mind_of_god_dind"]:
    #     
    username = "GOD"

    # Build command data and encode
    command_data = {
        "goal": goal,
        "iterations": iterations,
        "agent": agent,
        "history_id": history_id,
        "continuation": continuation,
        "ai_messages_only": ai_messages_only,
        "return_summary": return_summary,
        "additional_tools": additional_tools,
        "remove_agents_config_tools": remove_agents_config_tools,
        "orchestration_preprocess": orchestration_preprocess,
        "system_prompt_suffix": system_prompt_suffix
    }
    b64_data = base64.b64encode(json.dumps(command_data).encode('utf-8')).decode("utf-8")

    # Build the inline Python script
    python_script = """
import traceback
import asyncio
import base64
import json
import importlib
from pathlib import Path
from typing import Type, Union
from computer_use_demo.tools.base.baseheavenagent import BaseHeavenAgent, BaseHeavenAgentReplicant, HeavenAgentConfig
from computer_use_demo.tools.base.unified_chat import UnifiedChat, ProviderEnum
from computer_use_demo.tools.base.baseheaventool import BaseHeavenTool, ToolArgsSchema, ToolResult
# from computer_use_demo.tools.base.tools import *  
from computer_use_demo.tools.base.agents import *
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage, BaseMessage

    """ + """
def message_to_dict(msg):
    if isinstance(msg, (SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage)):
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        }
    return msg
    """ + f"""
async def send_message():
  
    # Decode the command data
    b64_data = {json.dumps(b64_data)}
    decoded_data = base64.b64decode(b64_data.encode('utf-8')).decode('utf-8')
    
    command_data = json.loads(decoded_data)
    try:
        
        
        # Convert additional_tools names to classes if specified
        additional_tool_classes = []
        # changed from truthy check
        if command_data['additional_tools'] is not None: 
            additional_tool_classes = [globals()[tool_name] for tool_name in command_data['additional_tools']]

        if command_data['remove_agents_config_tools']:
            # This doesnt seem like it works
            # Only use additional tools
            tools = additional_tool_classes

        if command_data['agent'] is None:
            # Create default agent with default config + additional tools
            base_tools = [] 
            # if command_data['remove_agents_config_tools'] else [BashTool, EditTool]
            config = HeavenAgentConfig(
                name="UnnamedAgent",
                system_prompt="You are a helpful assistant who can solve tasks using agent mode, which is a prompt format that gives you a tasking system." + (command_data.get('system_prompt_suffix', '') or ''),
                tools=base_tools + additional_tool_classes,
                provider=ProviderEnum.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
                temperature=0.0
            )
            agent = BaseHeavenAgent(config, UnifiedChat(), orchestrator=command_data.get('orchestration_preprocess', False))
        
        elif command_data['agent'] is not None:
            agent_name = command_data['agent']
            # print(f"Processing agent: {{agent_name}}")
            
            # Try Replicant approach
            replicant_success = False
            
            try:
                # Basic module path
                module_path = f"computer_use_demo.tools.base.agents.{{agent_name.lower()}}.{{agent_name.lower()}}"
                # print(f"Importing from: {{module_path}}")
                
                agent_module = importlib.import_module(module_path)
                
                # Convert agent_name to PascalCase
                pascal_name = ''.join(word.capitalize() for word in agent_name.split('_'))
                # print(f"Looking for class: {{pascal_name}}")
                
                # Try to get the class
                agent_class = getattr(agent_module, pascal_name)
                
                # Initialize the replicant
                
                
                if command_data['history_id']:
                    agent = agent_class(
                        history_id=command_data['history_id'],
                        system_prompt_suffix=command_data.get('system_prompt_suffix', ''),
                        additional_tools=[globals()[tool_name] for tool_name in command_data['additional_tools']] if command_data['additional_tools'] else None,
                        remove_agents_config_tools=command_data['remove_agents_config_tools']
                    )
                else:
                    agent = agent_class(system_prompt_suffix=command_data.get('system_prompt_suffix', ''),
                    additional_tools=[globals()[tool_name] for tool_name in command_data['additional_tools']] if command_data['additional_tools'] else None,
                    remove_agents_config_tools=command_data['remove_agents_config_tools']
                    )
                    
                replicant_success = True
                # print(f"Replicant created successfully")
                
            except Exception as e:
                print(f"Replicant approach failed: {{str(e)}}")
            
            # If Replicant failed, try config
            if not replicant_success:
                try:
                    config_name = f"{{agent_name.lower()}}_config"
                    config_path = f"computer_use_demo.tools.base.agents.{{agent_name.lower()}}.{{config_name}}"
                    # print(f"Looking for config at: {{config_path}}")
                    
                    config_module = importlib.import_module(config_path)
                    config = getattr(config_module, config_name)
                    config.system_prompt += (command_data.get('system_prompt_suffix', '') or '')
                    
                    
                    # Handle tool removal if specified
                    if command_data['remove_agents_config_tools']:
                        config.tools = []  # Clear existing tools
                    
                    # Add additional tools if we have any
                    if additional_tool_classes:
                        config.tools.extend(additional_tool_classes)
                        
                    # Create agent
                    agent = BaseHeavenAgent(config, UnifiedChat(), orchestrator=command_data.get('orchestration_preprocess', False))
                    # print(f"BaseHeavenAgent created with config")
                    
                except Exception as e:
                    print(f"Config approach failed: {{str(e)}}")
                    raise ValueError(f"Failed to load agent: {{agent_name}}")
                    
        if command_data['system_prompt_suffix'] is not None:
            agent.config.system_prompt += command_data['system_prompt_suffix']
            
        agent_command = f"agent goal={{command_data['goal']}}, iterations={{command_data['iterations']}}"

        if command_data['history_id'] and (command_data['continuation'] or 
            (command_data['continuation'] is None and agent.status.task_list)):
            result = await agent.continue_iterations(
                history_id=command_data['history_id'],
                continuation_iterations=command_data['iterations'],
                continuation_prompt=command_data['goal']
            )
        else:
            result = await agent.run(agent_command)

        messages = result["history"].messages

        response_data = {{
            "history_id": result["history_id"],
            "agent_name": result["agent_name"],
            "agent_status": result["agent_status"].dict() if result["agent_status"] else None,  
            "messages": [message_to_dict(msg) for msg in messages]
        }}

        if command_data['return_summary']:
            from computer_use_demo.tools.base.agents.summary_agent.summary_util import call_summary_agent
            summary_result = await call_summary_agent(result["history_id"])
            response_data["summary"] = summary_result

        # Encode response
        response_json = json.dumps(response_data)
        b64_response = base64.b64encode(response_json.encode("utf-8")).decode("utf-8")
        print(b64_response)
    
    except Exception as e:
        error_data = {{
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        error_json = json.dumps(error_data)
        b64_error = base64.b64encode(error_json.encode("utf-8")).decode("utf-8")
        print(b64_error)
        

asyncio.run(send_message())
"""

    try:
        # Create the Docker exec instance with the inline script
        exec_instance = client.api.exec_create(
            container=target_container,
            cmd=["python3", "-c", python_script]
        )
        exec_output = client.api.exec_start(exec_instance).decode("utf-8")
        
        # Get last line and decode base64
        lines = [line.strip() for line in exec_output.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("No output received from Docker exec")
        b64_result = lines[-1]

        try:
              decoded_json = base64.b64decode(b64_result.encode("utf-8")).decode("utf-8")
              data = json.loads(decoded_json)
              _remove_base64_images(data)
              if "error" in data:
                  # print(f"===> Error in Docker exec: {data['error']}\n\n")
                  # print(f"===> Traceback: {data['traceback']}")
                  raise ValueError(f"Error in docker exec: {data['error']}\n\nTraceback: {data['traceback']}")
              return data
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise RuntimeError(f"===> Base64 decoding failed. Raw output:\n{lines}\nError: {e}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        else:
            error_trace = traceback.format_exc()
            raise RuntimeError(f"Hermes error: Docker error: {str(e)}\n\nTraceback:\n{error_trace}")
    finally:
        client.close()
        

def _remove_base64_images(obj):
    """
    Recursively remove or blank out large base64 image data in the JSON object.
    """
    if isinstance(obj, dict):
        if obj.get('type') == 'base64' and 'image' in obj.get('media_type', '') and 'data' in obj:
            obj['data'] = "[REMOVED]"
        for v in obj.values():
            _remove_base64_images(v)
    elif isinstance(obj, list):
        for item in obj:
            _remove_base64_images(item)

def format_message(msg_dict, ai_messages_only=True):
    msg_type = msg_dict["type"]
    content = msg_dict["content"]

    if msg_type == "SystemMessage":
        return f"�� **System:** SystemMessage has been truncated. It can be viewed in the history."
    elif msg_type == "HumanMessage":
        if ai_messages_only is True:  # default True
            return f"💉 AGENT MODE's Task System injected a prompt (it can be viewed in the target container's copy of the history)...\n\n"
        else:
            return f"👤 **Human:**\n\n{content}\n\n"
    
    elif msg_type == "AIMessage":
        if isinstance(content, list):
            # Handle tool use blocks
            if any(block.get('type') == 'tool_use' for block in content):
                tool_blocks = [block for block in content if block.get('type') == 'tool_use']
                return "🤖 **Assistant (Tool Use):**\n" + "\n".join(
                    f"🛠️ Using {block['name']}\n\nInput: {block['input']}\n\n" 
                    for block in tool_blocks
                )
            # Handle text blocks
            text_blocks = [block.get('text', '') for block in content if block.get('type') == 'text']
            return "🤖 **Assistant:**\n\n" + "\n\n".join(text_blocks)
        return f"🤖 **Assistant:**\n\n{content}\n\n"
    elif msg_type == "ToolMessage":
        return f"🛠️ **Tool Result:**\n\n{content}\n\n"

    return f"📝 {msg_type}:\n{content}\n"

# ORIGINAL
async def use_hermes(
    target_container: str,
    source_container: str,
    goal: Optional[str] = None,
    agent: Optional[str] = None,
    iterations: int = 1,
    history_id: str = None,
    return_summary: bool = False,
    ai_messages_only: bool = True,
    remove_agents_config_tools: bool = False,
    continuation: bool = None,
    orchestration_preprocess: bool = False,
    additional_tools: Optional[List[str]] = None,
    hermes_config: Optional[Union[str, HermesConfig]] = None,  # Accepts a config name (str) or HermesConfig instance
    variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None,
    
    system_prompt_suffix: Optional[str] = None,
    return_last_response_only: bool = False
) -> str:
    """Execute an agent command via Hermes. If a config is provided,
    variable_inputs should match the config's templating structure."""
    print("\n=== START use_hermes ===")
    # print(f"Initial params - config: {hermes_config}, goal: {goal}")
    if additional_tools is not None:
        raise ValueError(f"DEPRECATED! additional_tools param is deprecated. Use None or use an agent that has the tool in its agent config.")

    if hermes_config is not None:
        if isinstance(hermes_config, str):
            home_dir = Path.home()
            config_path = home_dir / ".heaven" / "configs" / "hermes_configs" / f"{hermes_config}.json"
            with config_path.open("r") as f:
                config_data = json.load(f)
            config = HermesConfig.load_from_json(config_data)
            # print(f"\nAfter loading config - config type: {type(config)}")
        # Get the config's templating structure
        config_template = config.args_template.get("variable_inputs", {})

        # Validate variable inputs against config template
        if not variable_inputs and any(param["template"] for param in config_template.values()):
            raise ValueError(f"Config '{hermes_config}' requires variable inputs for templated parameters")

        if variable_inputs:
            for param_name, param_config in config_template.items():
                if not param_config.get("template"):
                    continue

                if param_name not in variable_inputs:
                    raise ValueError(f"Missing variable inputs for templated parameter: {param_name}")

                # Handle string templates (like goal)
                if "variables" in param_config:
                    required_vars = set(param_config["variables"])
                    provided_vars = set(variable_inputs[param_name].keys())
                    missing_vars = required_vars - provided_vars
                    if missing_vars:
                        raise ValueError(f"Missing variables for {param_name}: {missing_vars}")

                    # Check for empty values
                    empty_vars = [var for var, val in variable_inputs[param_name].items() 
                                if var in required_vars and not val]
                    if empty_vars:
                        raise ValueError(f"Empty values provided for variables: {empty_vars}")
                



        # Get command data with variables applied
        command_data = config.to_command_data(variable_inputs)
        # print(f"\nIn use_hermes: command_data: {command_data}")
       
        final_params = {
            "target_container": target_container,
            "source_container": source_container,
            "goal": command_data["goal"],
            "iterations": command_data["iterations"],
            "agent": command_data["agent"],
            "history_id": command_data["history_id"],
            "return_summary": command_data["return_summary"],
            "ai_messages_only": command_data["ai_messages_only"],
            "continuation": command_data["continuation"],
            "additional_tools": command_data["additional_tools"],
            "orchestration_preprocess": command_data["orchestration_preprocess"],
            "system_prompt_suffix": command_data["system_prompt_suffix"],
            "remove_agents_config_tools": command_data["remove_agents_config_tools"]
        }
        
        # print(f"\nHermes Debug: New Execution\n")
        # print(f"Final params: {final_params}\n")
        
    else:
        # If no hermes_config is provided, use the function parameters as given
        final_params = {
            "target_container": target_container,
            "source_container": source_container,
            "goal": goal,
            "iterations": iterations,
            "agent": agent,
            "history_id": history_id,
            "return_summary": return_summary,
            "ai_messages_only": ai_messages_only,
            "continuation": continuation,
            "additional_tools": additional_tools,
            "orchestration_preprocess": orchestration_preprocess,
            "system_prompt_suffix": system_prompt_suffix,
            "remove_agents_config_tools": remove_agents_config_tools
        }
        # print(f"\nHermes Debug: New Execution\n")
        # print(f"Final params before entering exec_agent try block: {final_params}\n")
   


    try:
        
        result = exec_agent_run_via_docker(
            target_container=final_params["target_container"],
            goal=final_params["goal"],
            iterations=final_params["iterations"],
            agent=final_params["agent"],
            source_container=final_params["source_container"],
            history_id=final_params["history_id"],
            return_summary=final_params["return_summary"],
            ai_messages_only=final_params["ai_messages_only"],
            continuation=final_params["continuation"],
            additional_tools=final_params["additional_tools"],
            orchestration_preprocess=final_params["orchestration_preprocess"],
            system_prompt_suffix=final_params["system_prompt_suffix"],
            remove_agents_config_tools=final_params["remove_agents_config_tools"]
        )
        
        # Check if there's a block report in the agent status
        if (result.get('agent_status') and 
            'extracted_content' in result['agent_status'] and 
            result['agent_status']['extracted_content'] and 
            'block_report' in result['agent_status']['extracted_content']):
            
            # Get the block report
            report_to_show = result['agent_status']['extracted_content']['block_report']
            if return_summary:
                # Include both block report and summary
                formatted_output = f"""===HERMES🦶🪽===\n\n**History ID:** {result['history_id']}\n\n
                📝 **Summary:** {result.get('summary', 'Summary was not completed for some reason...')}\n\n{report_to_show}
                \n\n===/HERMES🦶🪽==="""
            else:
                # Just show the block report
                formatted_output = f"""===HERMES🦶🪽===\n\n**History ID:** {result['history_id']}\n\n{report_to_show}\n\n===/HERMES🦶🪽==="""

        elif return_summary:
            formatted_output = f"""===HERMES🦶🪽===\n\n
            ```
            **History ID:** {result['history_id']}
            **Agent:** {result['agent_name']}
            **Status:** {result['agent_status']}
            ```
            📝 **Summary:** {result.get('summary', 'Summary was not completed for some reason...')}
            """
          
        else:
            # Filter messages if ai_messages_only
            messages = result["messages"]
            if return_last_response_only is True:
                # Just use the last message
                ai_messages = [msg for msg in messages if msg.get("type") == "AIMessage"]
                if ai_messages:
                    formatted_messages = format_message(ai_messages[-1], ai_messages_only=ai_messages_only)
                else:
                    messages = []
            else:
                formatted_messages = "\n\n".join(format_message(msg, ai_messages_only=ai_messages_only) for msg in result["messages"])
    
            formatted_output = f"""===HERMES🦶🪽===\n\n
    ```
    **History ID:** {result['history_id']}
    **Agent:** {result['agent_name']}
    **Status:** {result['agent_status']}
    ```
\n\n        
            {formatted_messages}
            
            \n\n===/HERMES🦶🪽===
            """
        return ToolResult(output=formatted_output)
        
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return ToolResult(error=f"Hermes error: {str(e)}\n\nTraceback:\n{error_trace}\nNote: Errors may be printed twice when the error is in exec_agent_run_via_docker")
        
        

async def use_hermes_dict(
    target_container: str,
    source_container: str,
    goal: Optional[str] = None,
    agent: Optional[str] = None,
    iterations: int = 1,
    history_id: str = None,
    return_summary: bool = False,
    ai_messages_only: bool = True,
    remove_agents_config_tools: bool = False,
    continuation: bool = None,
    orchestration_preprocess: bool = False,
    additional_tools: Optional[List[str]] = None,
    hermes_config: Optional[Union[str, HermesConfig]] = None,
    variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None,
    system_prompt_suffix: Optional[str] = None
) -> Union[Dict[str, Any], str]:  # Can return dict or error string
    """Execute an agent command via Hermes. If a config is provided,
    variable_inputs should match the config's templating structure."""
    print("\n=== START use_hermes_dict ===")
    print(f"Initial params - config: {hermes_config}, goal: {goal}")
    if additional_tools is not None:
        raise ValueError(f"DEPRECATED! additional_tools param is deprecated. Use None or use an agent that has the tool in its agent config.")

    if hermes_config is not None:
        if isinstance(hermes_config, str):
            home_dir = Path.home()
            config_path = home_dir / ".heaven" / "configs" / "hermes_configs" / f"{hermes_config}.json"
            with config_path.open("r") as f:
                config_data = json.load(f)
            config = HermesConfig.load_from_json(config_data)
            print(f"\nAfter loading config - config type: {type(config)}")
        # Get the config's templating structure
        config_template = config.args_template.get("variable_inputs", {})

        # Validate variable inputs against config template
        if not variable_inputs and any(param["template"] for param in config_template.values()):
            raise ValueError(f"Config '{hermes_config}' requires variable inputs for templated parameters")

        if variable_inputs:
            for param_name, param_config in config_template.items():
                if not param_config.get("template"):
                    continue

                if param_name not in variable_inputs:
                    raise ValueError(f"Missing variable inputs for templated parameter: {param_name}")

                # Handle string templates (like goal)
                if "variables" in param_config:
                    required_vars = set(param_config["variables"])
                    provided_vars = set(variable_inputs[param_name].keys())
                    missing_vars = required_vars - provided_vars
                    if missing_vars:
                        raise ValueError(f"Missing variables for {param_name}: {missing_vars}")

                    # Check for empty values
                    empty_vars = [var for var, val in variable_inputs[param_name].items() 
                                if var in required_vars and not val]
                    if empty_vars:
                        raise ValueError(f"Empty values provided for variables: {empty_vars}")
  



        # Get command data with variables applied
        command_data = config.to_command_data(variable_inputs)
        print(f"\nIn use_hermes: command_data: {command_data}")
   
        final_params = {
            "target_container": target_container,
            "source_container": source_container,
            "goal": command_data["goal"],
            "iterations": command_data["iterations"],
            "agent": command_data["agent"],
            "history_id": command_data["history_id"],
            "return_summary": command_data["return_summary"],
            "ai_messages_only": command_data["ai_messages_only"],
            "continuation": command_data["continuation"],
            "additional_tools": command_data["additional_tools"],
            "orchestration_preprocess": command_data["orchestration_preprocess"],
            "system_prompt_suffix": command_data["system_prompt_suffix"],
            "remove_agents_config_tools": command_data["remove_agents_config_tools"]
        }
        
        print(f"\nHermes Debug: New Execution\n")
        # print(f"Final params: {final_params}\n")
        
    else:
        # If no hermes_config is provided, use the function parameters as given
        final_params = {
            "target_container": target_container,
            "source_container": source_container,
            "goal": goal,
            "iterations": iterations,
            "agent": agent,
            "history_id": history_id,
            "return_summary": return_summary,
            "ai_messages_only": ai_messages_only,
            "continuation": continuation,
            "additional_tools": additional_tools,
            "orchestration_preprocess": orchestration_preprocess,
            "system_prompt_suffix": system_prompt_suffix,
            "remove_agents_config_tools": remove_agents_config_tools
        }
        print(f"\nHermes Debug: New Execution\n")
        print(f"Final params before entering exec_agent try block: {final_params}\n")
   


    try:
        # print(f"Final params in exec_agent try block: {final_params}\n")
        print("use_hermes_dict calling exec_agent_run_via_docker")
        result = exec_agent_run_via_docker(
            target_container=final_params["target_container"],
            goal=final_params["goal"],
            iterations=final_params["iterations"],
            agent=final_params["agent"],
            source_container=final_params["source_container"],
            history_id=final_params["history_id"],
            return_summary=final_params["return_summary"],
            ai_messages_only=final_params["ai_messages_only"],
            continuation=final_params["continuation"],
            additional_tools=final_params["additional_tools"],
            orchestration_preprocess=final_params["orchestration_preprocess"],
            system_prompt_suffix=final_params["system_prompt_suffix"],
            remove_agents_config_tools=final_params["remove_agents_config_tools"]
        )
        
        # Initialize variables
        has_block_report = False
        last_error = None
        
        # Check if there's a block report in the agent status
        if (result.get('agent_status') and 
            'extracted_content' in result['agent_status'] and 
            result['agent_status']['extracted_content'] and 
            'block_report' in result['agent_status']['extracted_content']):
            
            has_block_report = True
            report_to_show = result['agent_status']['extracted_content']['block_report']
            
            # Find the second-to-last ToolMessage in the history
            tool_messages = [msg for msg in result.get("messages", []) 
                            if msg.get("type") == "ToolMessage"]
            
            if len(tool_messages) >= 2:
                # Get the second-to-last ToolMessage
                error_message = tool_messages[-2]
                last_error = error_message.get("content", "Unknown tool error or no tool error found")
            else:
                last_error = "Error occurred but details not found in message history"
            
            if return_summary:
                # Include both block report and summary
                formatted_output = f"""===HERMES🦶🪽===\n\n**History ID:** {result['history_id']}\n\n
                📝 **Summary:** {result.get('summary', 'Summary was not completed for some reason...')}\n\n{report_to_show}
                \n\n===/HERMES🦶🪽==="""
            else:
                # Just show the block report
                formatted_output = f"""===HERMES🦶🪽===\n\n**History ID:** {result['history_id']}\n\n{report_to_show}\n\n===/HERMES🦶🪽==="""

        elif return_summary:
            formatted_output = f"""===HERMES🦶🪽===\n\n
            ```
            **History ID:** {result['history_id']}
            **Agent:** {result['agent_name']}
            **Status:** {result['agent_status']}
            ```
            📝 **Summary:** {result.get('summary', 'Summary was not completed for some reason...')}
            """
          
        else:
            # Filter messages if ai_messages_only
            messages = result["messages"]
      
            formatted_messages = "\n\n".join(format_message(msg, ai_messages_only=ai_messages_only) for msg in result["messages"])
    
            formatted_output = f"""===HERMES🦶🪽===\n\n
    ```
    **History ID:** {result['history_id']}
    **Agent:** {result['agent_name']}
    **Status:** {result['agent_status']}
    ```
            
            {formatted_messages}
            
            \n\n===/HERMES🦶🪽===
            """
        
        # Set has_error based on whether last_error has a value
        has_error = bool(last_error)
        
        # Check if agent_status indicates completion
        goal_accomplished = result.get('agent_status', {}).get('completed', False)
        
        # Get extracted_content_keys if available
        extracted_content_keys = []
        if (result.get('agent_status') and 
            result['agent_status'].get('extracted_content')):
            extracted_content_keys = list(result['agent_status']['extracted_content'].keys())
        
        # Return a dictionary with all relevant information
        return {
            "formatted_output": formatted_output,
            "history_id": result['history_id'],
            "agent_name": result.get('agent_name'),
            "agent_status": result['agent_status'],
            "has_block_report": has_block_report,
            "last_error": last_error,     # Error from second-to-last ToolMessage
            "has_error": has_error,       # True if last_error has a value
            "goal_accomplished": goal_accomplished,
            "extracted_content_keys": extracted_content_keys,
            "raw_result": result
        }
    
    except Exception as e:
        error_trace = traceback.format_exc()
        # Just return a string error message
        return f"Error in use_hermes_dict: {str(e)}\n\nTraceback:\n{error_trace}\n\nNote: Errors may be printed twice when the error is in exec_agent_run_via_docker"


#### Chaining

def handle_hermes_response(
    result: Dict[str, Any],
    handle_block_report_callable: Optional[Callable] = None
) -> Dict[str, Any]:
    """Handle hermes response, adding prepared_message to result dict."""
    
    formatted_output = result.get("formatted_output", "")
    
    if result.get("goal_accomplished", False):
        message = f"{formatted_output}\n\nThe result was processed by the handler system.\nHANDLER: ✅ Agent believes the goal was accomplished. If the summary indicates that might be a hallucination, check the work."
    elif result.get("has_block_report", False):
        message = formatted_output
    else:  # Neither complete nor blocked
        message = f"{formatted_output}\n\nThe result was processed by the handler system.\nHANDLER: 🤔 Agent has not indicated goal completion or blockage. Consider using continuation via history_id."
    
    result["prepared_message"] = message

    if result.get("has_block_report", False) and handle_block_report_callable:
        return handle_block_report_callable(result)
    
    return result


async def hermes_step(
    target_container: str,
    source_container: str,
    goal: Optional[str] = None,
    agent: Optional[str] = None,
    iterations: int = 1,
    history_id: Optional[str] = None,
    return_summary: bool = False,
    ai_messages_only: bool = True,
    remove_agents_config_tools: bool = False,
    continuation: Optional[bool] = None,
    orchestration_preprocess: bool = False,
    additional_tools: Optional[List[str]] = None,
    hermes_config: Optional[Union[str, HermesConfig]] = None,
    variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None,
    system_prompt_suffix: Optional[str] = None,
    handle_block_report_callable: Optional[Callable] = None,
    as_tool: bool = False
) -> Union[Dict[str, Any], str]:
    """Execute hermes and handle response. Returns dict unless as_tool=True."""
    result = await use_hermes_dict(
        target_container=target_container,
        source_container=source_container,
        goal=goal,
        agent=agent,
        iterations=iterations,
        history_id=history_id,
        return_summary=return_summary,
        ai_messages_only=ai_messages_only,
        remove_agents_config_tools=remove_agents_config_tools,
        continuation=continuation,
        orchestration_preprocess=orchestration_preprocess,
        additional_tools=additional_tools,
        hermes_config=hermes_config,
        variable_inputs=variable_inputs,
        system_prompt_suffix=system_prompt_suffix
    )
    
    # If result is an error string, return it directly
    if isinstance(result, str):
        return result
        
    result = handle_hermes_response(result, handle_block_report_callable)
    return result["prepared_message"] if as_tool else result


async def hermes_step_as_tool(
    target_container: str,
    source_container: str,
    goal: Optional[str] = None,
    agent: Optional[str] = None,
    iterations: int = 1,
    history_id: Optional[str] = None,
    return_summary: bool = False,
    ai_messages_only: bool = True,
    remove_agents_config_tools: bool = False,
    continuation: Optional[bool] = None,
    orchestration_preprocess: bool = False,
    additional_tools: Optional[List[str]] = None,
    hermes_config: Optional[Union[str, HermesConfig]] = None,
    variable_inputs: Optional[Dict[str, Union[Dict[str, Any], List[Any]]]] = None,
    system_prompt_suffix: Optional[str] = None,
    handle_block_report_callable: Optional[Callable] = None
) -> str:
    """Wrapper for hermes_step that always returns just the prepared message string for tools."""
    return await hermes_step(
        target_container=target_container,
        source_container=source_container,
        goal=goal,
        agent=agent,
        iterations=iterations,
        history_id=history_id,
        return_summary=return_summary,
        ai_messages_only=ai_messages_only,
        remove_agents_config_tools=remove_agents_config_tools,
        continuation=continuation,
        orchestration_preprocess=orchestration_preprocess,
        additional_tools=additional_tools,
        hermes_config=hermes_config,
        variable_inputs=variable_inputs,
        system_prompt_suffix=system_prompt_suffix,
        handle_block_report_callable=handle_block_report_callable,
        as_tool=True
    )