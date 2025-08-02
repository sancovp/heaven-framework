from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..baseheavenagent import HeavenAgentConfig

# Version X1 - Recreated

class BlockTypeVX1(Enum):
    FREESTYLE = "freestyle"
    REFERENCE = "reference"

class PromptBlockDefinitionVX1(BaseModel):
    type: BlockTypeVX1
    content: str # For FREESTYLE, this is the raw string. For REFERENCE, this is the reference_string.

    class Config:
        use_enum_values = True # Ensures type is stored as string value

class PromptStepDefinitionVX1(BaseModel):
    name: Optional[str] = None
    blocks: List[PromptBlockDefinitionVX1]

class PromptInjectionSystemConfigVX1(BaseModel):
    steps: List[PromptStepDefinitionVX1]
    template_vars: Dict[str, Any]
    agent_config: HeavenAgentConfig # Expect a fully configured HeavenAgentConfig for PIS's own reference resolution

class PromptInjectionSystemVX1:
    """
    Processes a sequence of PromptStepDefinitions. Each step consists of multiple blocks
    (freestyle or reference) that are processed and concatenated to form a single prompt string for that step.
    FREESTYLE blocks are templated using PIS config's template_vars.
    REFERENCE blocks are resolved by the provided HeavenAgentConfig and used as-is.
    """
    def __init__(self, config: PromptInjectionSystemConfigVX1):
        self.config = config
        self.current_step_index = 0
        # print(f"DEBUG PIS_VX1 (Recreated): Initialized with {len(config.steps)} steps.")

    def _render_single_block(self, block: PromptBlockDefinitionVX1) -> str:
        """Renders a single PromptBlockDefinition."""
        if block.type == BlockTypeVX1.FREESTYLE.value: # Compare with string value of enum
            raw_freestyle_content = block.content
            if self.config.template_vars and raw_freestyle_content:
                try:
                    formatted_content = raw_freestyle_content.format(**self.config.template_vars)
                    return formatted_content
                except KeyError as e:
                    # print(f"DEBUG PIS_VX1: KeyError in FREESTYLE: {e}. Using raw: '{raw_freestyle_content}'")
                    return raw_freestyle_content 
                except Exception as ex:
                    # print(f"DEBUG PIS_VX1: Exception in FREESTYLE: {ex}. Using raw: '{raw_freestyle_content}'")
                    return raw_freestyle_content 
            else:
                return raw_freestyle_content
        
        elif block.type == BlockTypeVX1.REFERENCE.value: # Compare with string value of enum
            reference_string = block.content
            if reference_string:
                hac = self.config.agent_config # PIS uses its own HAC for its reference blocks
                hac.system_prompt = ""  
                hac.prompt_suffix_blocks = [reference_string] 
                
                resolved_content_raw = hac.get_system_prompt()

                if resolved_content_raw.startswith("\n\n") and len(resolved_content_raw) > 2:
                    return resolved_content_raw[2:]
                elif resolved_content_raw == "\n\n":
                    return ""
                else:
                    return resolved_content_raw
            else:
                return "" 
        
        return "" # Should not happen

    def get_next_prompt(self) -> Optional[str]:
        """
        Processes the next step, concatenates its blocks, returns the single prompt string.
        """
        if not self.has_next_prompt():
            return None
        
        current_step_def = self.config.steps[self.current_step_index]
        prompt_pieces_for_this_step: List[str] = []
        for block_def in current_step_def.blocks:
            rendered_block_piece = self._render_single_block(block_def)
            prompt_pieces_for_this_step.append(rendered_block_piece)
        
        final_prompt_for_step = "".join(prompt_pieces_for_this_step)
        self.current_step_index += 1
        return final_prompt_for_step

    def has_next_prompt(self) -> bool:
        return self.current_step_index < len(self.config.steps)

    def reset_sequence(self) -> None:
        self.current_step_index = 0
