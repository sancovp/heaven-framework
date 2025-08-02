
import traceback
import asyncio
from typing import Optional
from computer_use_demo.tools.base.tool_utils.hermes_utils import use_hermes
from computer_use_demo.tools.base.baseheaventool import ToolResult

async def produce_prompt_block_util(prompt_type: str, prompt_purpose: str, iterations: int, history_id: Optional[str] = None, continuation: Optional[bool] = None, target_container: str = "mind_of_god", source_container: str = "mind_of_god"):
    variable_inputs = {}
    variable_inputs['goal'] = {
        'prompt_type': prompt_type,
        'prompt_purpose': prompt_purpose
    }
    variable_inputs['iterations'] = iterations
    variable_inputs['history_id'] = history_id
    variable_inputs['continuation'] = continuation
    hermes_config_identifier = "prompt_block"
    print(f"Executing util function (via use_hermes) for config: {hermes_config_identifier}")
    print(f"Reconstructed variable_inputs: {variable_inputs}")
    try:
        return await use_hermes(
            hermes_config=hermes_config_identifier, variable_inputs=variable_inputs,
            target_container=target_container, source_container=source_container,
            iterations=iterations, return_summary=False, ai_messages_only=True,
            goal=None, agent=None, history_id=None, continuation=None, orchestration_preprocess=False,
            system_prompt_suffix=None, return_last_response_only=False
        )
    except Exception as e:
        return ToolResult(error=f"Error in produce_prompt_block_util calling use_hermes: {e}\n{traceback.format_exc()}")
