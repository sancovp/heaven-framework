#!/usr/bin/env python3
"""
ContextManager - Universal Context Engineering for AI Agents

Provides CRUD operations for synthetic history engineering with cross-provider compatibility.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableSequence
from pydantic import BaseModel, Field

# Import your existing classes
import sys
sys.path.append('/home/GOD/core')
from ..memory.history import History

# Import PIS vX1 system
from computer_use_demo.tools.base.tool_utils.prompt_injection_system_vX1 import (
    PromptInjectionSystemVX1,
    PromptInjectionSystemConfigVX1,
    PromptStepDefinitionVX1,
    PromptBlockDefinitionVX1,
    BlockTypeVX1
)
from computer_use_demo.tools.base.baseheavenagent import HeavenAgentConfig


# Pydantic v2 Models for Weave Operations
class WeaveOperationType(str, Enum):
    MESSAGE_RANGE = "message_range"
    ITERATION = "iteration"
    ITERATION_RANGE = "iteration_range"
    FULL_HISTORY = "full_history"

class MessageRangeOperation(BaseModel):
    type: Literal[WeaveOperationType.MESSAGE_RANGE] = WeaveOperationType.MESSAGE_RANGE
    source_history_id: str
    start_index: int
    end_index: int
    target_position: int = -1

class IterationOperation(BaseModel):
    type: Literal[WeaveOperationType.ITERATION] = WeaveOperationType.ITERATION
    source_history_id: str
    iteration_number: int
    target_position: int = -1
    include_system: bool = False

class IterationRangeOperation(BaseModel):
    type: Literal[WeaveOperationType.ITERATION_RANGE] = WeaveOperationType.ITERATION_RANGE
    source_history_id: str
    start_iteration: int
    end_iteration: int
    target_position: int = -1
    include_system: bool = False

class FullHistoryOperation(BaseModel):
    type: Literal[WeaveOperationType.FULL_HISTORY] = WeaveOperationType.FULL_HISTORY
    source_history_id: str
    target_position: int = -1
    strategy: Literal["append", "interleave", "merge"] = "append"

# Union type for all operations
WeaveOperation = Union[
    MessageRangeOperation,
    IterationOperation, 
    IterationRangeOperation,
    FullHistoryOperation
]

# Pydantic v2 Models for Universal Chain Pattern System
class ChainOperationType(str, Enum):
    # Context Engineering Operations
    CONTEXT_WEAVE = "context_weave"
    CONTEXT_INJECT = "context_inject" 
    CONTEXT_MOVE = "context_move"
    
    # PIS Operations
    PIS_INJECTION = "pis_injection"
    PIS_SEQUENCE = "pis_sequence"
    
    # Tool Operations
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    
    # Agent Operations
    AGENT_SPAWN = "agent_spawn"
    AGENT_MEASURE = "agent_measure"
    
    # History Data Operations
    HISTORY_EXTRACT = "history_extract"
    
    # Meta Operations
    PATTERN_REF = "pattern_ref"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"

class BaseChainOperation(BaseModel):
    type: ChainOperationType
    description: Optional[str] = None
    result_var: Optional[str] = None  # Variable to store operation result
    depends_on: List[str] = Field(default_factory=list)  # Dependencies on other result_vars
    
class ContextWeaveOperation(BaseChainOperation):
    type: Literal[ChainOperationType.CONTEXT_WEAVE] = ChainOperationType.CONTEXT_WEAVE
    weave_operations: List[Dict[str, Any]]  # WeaveOperation objects as dicts
    base_history_id: Optional[str] = None
    max_tokens: Optional[int] = None

class ContextInjectOperation(BaseChainOperation):
    type: Literal[ChainOperationType.CONTEXT_INJECT] = ChainOperationType.CONTEXT_INJECT
    target_history_id: str
    content: str  # Can use template variables
    message_type: Literal["human", "ai", "system"] = "human"
    position: int = -1
    use_pis: bool = False
    pis_blocks: Optional[List[str]] = None

class PisInjectionOperation(BaseChainOperation):
    type: Literal[ChainOperationType.PIS_INJECTION] = ChainOperationType.PIS_INJECTION
    target_history_id: str
    pis_blocks: List[str]  # Block names to load and render
    template_vars: Dict[str, Any] = Field(default_factory=dict)
    agent_config_ref: Optional[str] = None  # Registry reference to agent config

class AgentSpawnOperation(BaseChainOperation):
    type: Literal[ChainOperationType.AGENT_SPAWN] = ChainOperationType.AGENT_SPAWN
    agent_config_ref: str  # Registry reference to agent configuration
    context_history_id: str  # History to provide as context
    agent_prompt: Optional[str] = None
    max_iterations: int = 10
    success_criteria: Optional[Dict[str, Any]] = None

class ToolUseOperation(BaseChainOperation):
    type: Literal[ChainOperationType.TOOL_USE] = ChainOperationType.TOOL_USE
    target_history_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: Optional[str] = None

class PatternRefOperation(BaseChainOperation):
    type: Literal[ChainOperationType.PATTERN_REF] = ChainOperationType.PATTERN_REF
    pattern_registry_ref: str  # registry_object_ref=pattern_registry:pattern_name
    template_vars_override: Dict[str, Any] = Field(default_factory=dict)

class HistoryExtractOperation(BaseChainOperation):
    type: Literal[ChainOperationType.HISTORY_EXTRACT] = ChainOperationType.HISTORY_EXTRACT
    source_history_id: str
    kw: str  # Extract keyword (auto-resolves kw_1 to kw)
    get_all: bool = False  # Get all variants
    return_last_instance_only: bool = False  # Get only the latest one

class ConditionalOperation(BaseChainOperation):
    type: Literal[ChainOperationType.CONDITIONAL] = ChainOperationType.CONDITIONAL
    condition: str  # Python expression using result_vars
    true_operations: List[Dict[str, Any]]
    false_operations: List[Dict[str, Any]] = Field(default_factory=list)

# Union type for all chain operations
ChainOperation = Union[
    ContextWeaveOperation,
    ContextInjectOperation,
    PisInjectionOperation,
    AgentSpawnOperation,
    ToolUseOperation,
    HistoryExtractOperation,
    PatternRefOperation,
    ConditionalOperation
]

class ChainPatternContextEngineeringChainPattern(BaseModel):
    pattern_name: str
    description: Optional[str] = None
    chain_operations: List[Dict[str, Any]]  # ChainOperation objects with variable placeholders
    template_vars: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution state (not persisted to registry)
    execution_state: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    result_vars: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    model_config = {
        "extra": "allow"
    }

# Legacy alias for backward compatibility
WeavePattern = ChainPatternContextEngineeringChainPattern


@dataclass
class ContextMetadata:
    """Metadata for synthetic histories"""
    name: str
    domain: str
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    token_count: int = 0
    provider_compatibility: List[str] = None
    synthetic: bool = True
    pattern: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.provider_compatibility is None:
            self.provider_compatibility = ["openai", "anthropic", "google", "xai"]


class ContextManager:
    """Universal context engineering manager for synthetic history creation and manipulation"""
    
    def __init__(self, storage_path: str = "/tmp/synthetic_histories"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
    # CRUD Operations
    async def create_synthetic_history(self, 
                                     name: str, 
                                     domain: str, 
                                     system_prompt: Optional[str] = None,
                                     **metadata_kwargs) -> History:
        """Create new synthetic history with metadata"""
        
        # Create metadata
        metadata = ContextMetadata(
            name=name,
            domain=domain,
            **metadata_kwargs
        )
        
        # Create empty history
        history = History(messages=[])
        history.metadata = asdict(metadata)
        
        # Add system message if provided
        if system_prompt:
            history.messages.append(SystemMessage(content=system_prompt))
            
        # Generate unique ID and save
        history_id = f"synthetic_{name}_{uuid.uuid4().hex[:8]}"
        history.history_id = history_id
        
        await self._save_history(history_id, history)
        return history
    
    async def read_history(self, history_id: str) -> History:
        """Load existing history"""
        history_file = self.storage_path / f"{history_id}.json"
        
        if not history_file.exists():
            # Try loading from regular History system
            try:
                return History.load_from_id(history_id)
            except:
                raise FileNotFoundError(f"History {history_id} not found")
                
        with open(history_file, 'r') as f:
            data = json.load(f)
            
        # Reconstruct History object
        history = History(messages=[])
        history.history_id = data['history_id']
        history.metadata = data['metadata']
        
        # Reconstruct messages
        for msg_data in data['messages']:
            msg_type = msg_data['type']
            content = msg_data['content']
            
            if msg_type == 'human':
                history.messages.append(HumanMessage(content=content))
            elif msg_type == 'ai':
                history.messages.append(AIMessage(content=content))
            elif msg_type == 'system':
                history.messages.append(SystemMessage(content=content))
            elif msg_type == 'tool':
                history.messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg_data.get('tool_call_id', '')
                ))
                
        return history
    
    async def update_history_metadata(self, history_id: str, **metadata_updates):
        """Update metadata fields"""
        history = await self.read_history(history_id)
        
        # Update metadata
        history.metadata.update(metadata_updates)
        history.metadata['updated_at'] = datetime.now().isoformat()
        
        await self._save_history(history_id, history)
        return history
    
    async def delete_history(self, history_id: str):
        """Remove history"""
        history_file = self.storage_path / f"{history_id}.json"
        if history_file.exists():
            history_file.unlink()
    
    # Message Construction & Injection
    async def inject(self, 
                    history_id: str,
                    content: Union[str, Path, bytes, Any],
                    message_type: Literal["human", "ai", "system"] = "human",
                    position: int = -1,
                    image_url: Optional[str] = None,
                    **kwargs) -> History:
        """Universal message constructor and injector"""
        
        history = await self.read_history(history_id)
        
        # Process content based on type
        processed_content = await self._process_content(content, image_url)
        
        # Create appropriate message
        if message_type == "human":
            message = HumanMessage(content=processed_content, **kwargs)
        elif message_type == "ai":
            message = AIMessage(content=processed_content, **kwargs)
        elif message_type == "system":
            message = SystemMessage(content=processed_content, **kwargs)
        else:
            raise ValueError(f"Unsupported message type: {message_type}")
        
        # Insert at position
        if position == -1:
            history.messages.append(message)
        else:
            history.messages.insert(position, message)
            
        # Update token count estimate
        await self._update_token_count(history)
        
        await self._save_history(history_id, history)
        return history
    
    async def inject_tool_use(self, 
                             history_id: str,
                             tool_name: str,
                             tool_args: Dict[str, Any],
                             tool_call_id: Optional[str] = None,
                             position: int = -1) -> History:
        """Inject tool use message (OpenAI format)"""
        
        history = await self.read_history(history_id)
        
        if tool_call_id is None:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        # Create AIMessage with tool call
        tool_call = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args)
            }
        }
        
        ai_message = AIMessage(
            content="",  # Empty content for tool calls
            additional_kwargs={"tool_calls": [tool_call]}
        )
        
        # Insert message
        if position == -1:
            history.messages.append(ai_message)
        else:
            history.messages.insert(position, ai_message)
            
        await self._update_token_count(history)
        await self._save_history(history_id, history)
        return history
    
    async def inject_tool_result(self, 
                                history_id: str,
                                tool_call_id: str,
                                result: Any,
                                error: bool = False) -> History:
        """Inject tool result message"""
        
        history = await self.read_history(history_id)
        
        # Convert result to string
        if isinstance(result, str):
            result_content = result
        else:
            result_content = json.dumps(result, default=str)
            
        if error:
            result_content = f"Error: {result_content}"
        
        tool_message = ToolMessage(
            content=result_content,
            tool_call_id=tool_call_id
        )
        
        history.messages.append(tool_message)
        
        await self._update_token_count(history)
        await self._save_history(history_id, history)
        return history
    
    # PIS Preprocessor Methods
    def pis_preprocessor(self,
                        blocks: List[PromptBlockDefinitionVX1],
                        template_vars: Dict[str, Any],
                        agent_config: HeavenAgentConfig) -> str:
        """Render PIS blocks into single prompt string for inject"""
        
        # Create minimal PIS config for single step
        step = PromptStepDefinitionVX1(blocks=blocks)
        config = PromptInjectionSystemConfigVX1(
            steps=[step],
            template_vars=template_vars,
            agent_config=agent_config
        )
        
        # Render and return
        pis = PromptInjectionSystemVX1(config)
        return pis.get_next_prompt()
    
    def loading_pis_preprocessor(self,
                                block_names: List[str],
                                template_vars: Dict[str, Any],
                                agent_config: HeavenAgentConfig,
                                block_registry: Optional[Dict[str, PromptBlockDefinitionVX1]] = None) -> str:
        """Load prompt blocks by name and render into single prompt string"""
        
        # Load blocks from registry/filesystem/database
        blocks = []
        for block_name in block_names:
            block = self._load_prompt_block(block_name, block_registry)
            blocks.append(block)
        
        # Use existing preprocessor
        return self.pis_preprocessor(blocks, template_vars, agent_config)
    
    def _load_prompt_block(self, 
                          block_name: str, 
                          registry: Optional[Dict[str, PromptBlockDefinitionVX1]] = None) -> PromptBlockDefinitionVX1:
        """Load block definition by name from registry or storage"""
        
        # Check registry first
        if registry and block_name in registry:
            return registry[block_name]
        
        # Check filesystem - look for blocks in storage directory
        block_file = self.storage_path / "blocks" / f"{block_name}.json"
        if block_file.exists():
            with open(block_file, 'r') as f:
                block_data = json.load(f)
                return PromptBlockDefinitionVX1(**block_data)
        
        # Default blocks library
        default_blocks = self._get_default_blocks()
        if block_name in default_blocks:
            return default_blocks[block_name]
        
        # If not found, create a simple FREESTYLE block with the name as content
        return PromptBlockDefinitionVX1(
            type=BlockTypeVX1.FREESTYLE,
            content=f"[Block '{block_name}' not found - using name as content]"
        )
    
    def _get_default_blocks(self) -> Dict[str, PromptBlockDefinitionVX1]:
        """Get default prompt block library"""
        return {
            "system_setup": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.FREESTYLE,
                content="You are an AI assistant. You help users with {domain} tasks."
            ),
            "user_intro": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.FREESTYLE,
                content="I'm working on {task_type} and need help with {specific_need}."
            ),
            "task_spec": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.FREESTYLE,
                content="Please {action} while keeping in mind {constraints}."
            ),
            "context_ref": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.REFERENCE,
                content="@current_codebase_context"
            ),
            "expert_intro": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.FREESTYLE,
                content="You are an expert {expert_domain} consultant with {experience_level} experience."
            ),
            "debugging_setup": PromptBlockDefinitionVX1(
                type=BlockTypeVX1.FREESTYLE,
                content="Help debug this {language} code. Focus on {error_type} issues."
            )
        }
    
    async def save_block(self, 
                        block_name: str, 
                        block: PromptBlockDefinitionVX1):
        """Save a prompt block to filesystem for reuse"""
        blocks_dir = self.storage_path / "blocks"
        blocks_dir.mkdir(exist_ok=True)
        
        block_file = blocks_dir / f"{block_name}.json"
        with open(block_file, 'w') as f:
            json.dump(block.dict(), f, indent=2)
    
    # Chain Pattern System - Registry Integration
    async def save_chain_pattern(self, 
                                pattern: ChainPatternContextEngineeringChainPattern):
        """Save chain pattern to registry system"""
        from computer_use_demo.tools.utils.omnitool import omnitool
        
        # Ensure registry exists
        try:
            omnitool('RegistryTool', parameters={
                'operation': 'create_registry',
                'registry_name': 'context_engineering_patterns'
            })
        except:
            pass  # Registry may already exist
        
        # Save pattern to registry
        result = omnitool('RegistryTool', parameters={
            'operation': 'add',
            'registry_name': 'context_engineering_patterns',
            'key': pattern.pattern_name,
            'value': pattern.model_dump(exclude={'execution_state', 'result_vars'})
        })
        
        return result
    
    async def load_chain_pattern(self, pattern_name: str) -> ChainPatternContextEngineeringChainPattern:
        """Load chain pattern from registry system"""
        from computer_use_demo.tools.utils.omnitool import omnitool
        
        try:
            # Try to load from registry
            result = omnitool('RegistryTool', parameters={
                'operation': 'get',
                'registry_name': 'context_engineering_patterns',
                'key': pattern_name
            })
            
            if result and hasattr(result, 'output'):
                pattern_data = json.loads(result.output)
                return ChainPatternContextEngineeringChainPattern(**pattern_data)
                
        except Exception as e:
            # Fallback to default patterns
            default_pattern = self._get_default_chain_pattern(pattern_name)
            if default_pattern:
                return default_pattern
            raise FileNotFoundError(f"Chain pattern '{pattern_name}' not found: {e}")
    
    # Legacy alias
    async def load_weave_pattern(self, pattern_name: str) -> WeavePattern:
        """Legacy alias for load_chain_pattern"""
        return await self.load_chain_pattern(pattern_name)
    
    async def list_weave_patterns(self) -> List[str]:
        """List all available weave patterns"""
        patterns_dir = self.storage_path / "patterns"
        patterns = []
        
        # Add saved patterns
        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob("*.json"):
                patterns.append(pattern_file.stem)
        
        # Add default patterns
        patterns.extend(self._get_default_pattern_names())
        
        return sorted(list(set(patterns)))
    
    async def delete_weave_pattern(self, pattern_name: str):
        """Delete weave pattern from filesystem"""
        patterns_dir = self.storage_path / "patterns"
        pattern_file = patterns_dir / f"{pattern_name}.json"
        
        if pattern_file.exists():
            pattern_file.unlink()
        else:
            raise FileNotFoundError(f"Pattern '{pattern_name}' not found")
    
    async def execute_pattern(self, 
                            pattern_name: str, 
                            template_vars: Dict[str, Any],
                            base_history_id: Optional[str] = None,
                            max_tokens: Optional[int] = None) -> str:
        """Execute weave pattern with variable substitution"""
        
        # Load pattern
        pattern = await self.load_weave_pattern(pattern_name)
        
        # Merge template vars (execution vars override pattern defaults)
        final_vars = {**pattern.template_vars, **template_vars}
        
        # Substitute variables using safe string replacement
        operations_copy = json.loads(json.dumps(pattern.operations))  # Deep copy
        
        def substitute_variables(obj, variables):
            """Recursively substitute variables in nested dict/list structure"""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    result[key] = substitute_variables(value, variables)
                return result
            elif isinstance(obj, list):
                return [substitute_variables(item, variables) for item in obj]
            elif isinstance(obj, str):
                # Only substitute if the entire string is a variable placeholder
                if obj.startswith("{") and obj.endswith("}") and obj.count("{") == 1:
                    var_name = obj[1:-1]
                    if var_name in variables:
                        return variables[var_name]
                    else:
                        raise KeyError(var_name)
                return obj
            else:
                return obj
        
        try:
            substituted_operations = substitute_variables(operations_copy, final_vars)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
        except Exception as e:
            raise ValueError(f"Error in variable substitution: {e}")
        
        # Convert to operation objects
        operation_objects = []
        for op_data in substituted_operations:
            op_type = op_data.get('type')
            
            if op_type == WeaveOperationType.MESSAGE_RANGE:
                operation_objects.append(MessageRangeOperation(**op_data))
            elif op_type == WeaveOperationType.ITERATION:
                operation_objects.append(IterationOperation(**op_data))
            elif op_type == WeaveOperationType.ITERATION_RANGE:
                operation_objects.append(IterationRangeOperation(**op_data))
            elif op_type == WeaveOperationType.FULL_HISTORY:
                operation_objects.append(FullHistoryOperation(**op_data))
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
        
        # Execute weave with substituted operations
        return await self.weave(operation_objects, base_history_id, max_tokens)
    
    def _get_default_pattern_names(self) -> List[str]:
        """Get names of default patterns"""
        return [
            "extract_problem_solution",
            "debugging_flow",
            "conversation_summary",
            "expert_consultation"
        ]
    
    def _get_default_pattern(self, pattern_name: str) -> Optional[WeavePattern]:
        """Get default pattern by name"""
        defaults = {
            "extract_problem_solution": WeavePattern(
                pattern_name="extract_problem_solution",
                description="Extract problem statement and solution from debugging conversation",
                operations=[
                    {
                        "type": "iteration",
                        "source_history_id": "{source_session}",
                        "iteration_number": "{problem_iteration}",
                        "include_system": False
                    },
                    {
                        "type": "iteration_range",
                        "source_history_id": "{source_session}",
                        "start_iteration": "{solution_start}",
                        "end_iteration": "{solution_end}",
                        "include_system": False
                    }
                ],
                template_vars={
                    "source_session": "",
                    "problem_iteration": 0,
                    "solution_start": 3,
                    "solution_end": 5
                },
                metadata={"domain": "debugging", "use_case": "knowledge_extraction"}
            ),
            
            "debugging_flow": WeavePattern(
                pattern_name="debugging_flow",
                description="Create linear debugging narrative from multi-turn conversation",
                operations=[
                    {
                        "type": "message_range",
                        "source_history_id": "{debug_session}",
                        "start_index": "{start_msg}",
                        "end_index": "{end_msg}"
                    }
                ],
                template_vars={
                    "debug_session": "",
                    "start_msg": 1,
                    "end_msg": 10
                },
                metadata={"domain": "debugging", "use_case": "narrative_creation"}
            ),
            
            "conversation_summary": WeavePattern(
                pattern_name="conversation_summary",
                description="Extract key messages from long conversation",
                operations=[
                    {
                        "type": "iteration",
                        "source_history_id": "{source_session}",
                        "iteration_number": 0,
                        "include_system": True
                    },
                    {
                        "type": "iteration",
                        "source_history_id": "{source_session}",
                        "iteration_number": "{final_iteration}",
                        "include_system": False
                    }
                ],
                template_vars={
                    "source_session": "",
                    "final_iteration": 5
                },
                metadata={"domain": "general", "use_case": "summarization"}
            ),
            
            "expert_consultation": WeavePattern(
                pattern_name="expert_consultation",
                description="Combine expert setup with user questions",
                operations=[
                    {
                        "type": "full_history",
                        "source_history_id": "{expert_setup}",
                        "strategy": "append"
                    },
                    {
                        "type": "message_range",
                        "source_history_id": "{user_session}",
                        "start_index": "{question_start}",
                        "end_index": "{question_end}"
                    }
                ],
                template_vars={
                    "expert_setup": "",
                    "user_session": "",
                    "question_start": 0,
                    "question_end": 5
                },
                metadata={"domain": "consultation", "use_case": "expert_pairing"}
            )
        }
        
        return defaults.get(pattern_name)
    
    def _get_default_chain_pattern(self, pattern_name: str) -> Optional[ChainPatternContextEngineeringChainPattern]:
        """Get default chain pattern by name"""
        # Convert legacy patterns to chain patterns for backward compatibility
        legacy_pattern = self._get_default_pattern(pattern_name)
        if legacy_pattern:
            # Convert WeavePattern operations to ContextWeaveOperation
            chain_operations = [{
                "type": "context_weave",
                "weave_operations": legacy_pattern.operations,
                "result_var": "woven_history"
            }]
            
            return ChainPatternContextEngineeringChainPattern(
                pattern_name=legacy_pattern.pattern_name,
                description=legacy_pattern.description,
                chain_operations=chain_operations,
                template_vars=legacy_pattern.template_vars,
                metadata=legacy_pattern.metadata
            )
        
        # Add new chain-specific default patterns
        chain_defaults = {
            "complete_debugging_workflow": ChainPatternContextEngineeringChainPattern(
                pattern_name="complete_debugging_workflow",
                description="Complete debugging analysis with PIS prompts and agent collaboration",
                chain_operations=[
                    {
                        "type": "context_weave",
                        "description": "Extract problem context",
                        "weave_operations": [
                            {
                                "type": "iteration",
                                "source_history_id": "{debug_session}",
                                "iteration_number": 0,
                                "include_system": False
                            }
                        ],
                        "result_var": "problem_context"
                    },
                    {
                        "type": "pis_injection",
                        "description": "Add expert analysis prompt",
                        "target_history_id": "{problem_context}",
                        "pis_blocks": ["debugging_expert_intro", "analysis_prompt"],
                        "template_vars": {"domain": "software_debugging"},
                        "result_var": "analysis_ready_context"
                    },
                    {
                        "type": "agent_spawn",
                        "description": "Spawn debugging expert agent",
                        "agent_config_ref": "registry_object_ref=agent_configs:debugging_expert",
                        "context_history_id": "{analysis_ready_context}",
                        "result_var": "expert_analysis"
                    }
                ],
                template_vars={
                    "debug_session": "",
                    "domain": "general_debugging"
                },
                metadata={"domain": "debugging", "use_case": "complete_workflow"}
            )
        }
        
        return chain_defaults.get(pattern_name)
    
    async def execute_chain_pattern(self,
                                   pattern_name: str,
                                   template_vars: Dict[str, Any],
                                   base_history_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute chain pattern with comprehensive operation support"""
        from computer_use_demo.tools.utils.omnitool import omnitool
        
        # Load pattern
        pattern = await self.load_chain_pattern(pattern_name)
        
        # Initialize execution state
        pattern.execution_state = {"current_step": 0, "completed_steps": []}
        pattern.result_vars = {}
        
        # Merge template vars
        final_vars = {**pattern.template_vars, **template_vars}
        
        # Execute chain operations sequentially
        for i, op_data in enumerate(pattern.chain_operations):
            try:
                # Substitute variables in operation
                substituted_op = self._substitute_chain_variables(op_data, final_vars, pattern.result_vars)
                
                # Execute operation based on type
                operation_type = substituted_op.get('type')
                result = None
                
                if operation_type == ChainOperationType.CONTEXT_WEAVE:
                    result = await self._execute_context_weave(substituted_op)
                elif operation_type == ChainOperationType.CONTEXT_INJECT:
                    result = await self._execute_context_inject(substituted_op)
                elif operation_type == ChainOperationType.PIS_INJECTION:
                    result = await self._execute_pis_injection(substituted_op)
                elif operation_type == ChainOperationType.TOOL_USE:
                    result = await self._execute_tool_use(substituted_op)
                elif operation_type == ChainOperationType.AGENT_SPAWN:
                    result = await self._execute_agent_spawn(substituted_op)
                elif operation_type == ChainOperationType.HISTORY_EXTRACT:
                    result = await self._execute_history_extract(substituted_op)
                elif operation_type == ChainOperationType.PATTERN_REF:
                    result = await self._execute_pattern_ref(substituted_op, final_vars)
                elif operation_type == ChainOperationType.CONDITIONAL:
                    result = await self._execute_conditional(substituted_op, pattern.result_vars)
                else:
                    raise ValueError(f"Unknown operation type: {operation_type}")
                
                # Store result if result_var specified
                if substituted_op.get('result_var') and result:
                    pattern.result_vars[substituted_op['result_var']] = result
                
                # Update execution state
                pattern.execution_state["current_step"] = i + 1
                pattern.execution_state["completed_steps"].append({
                    "step": i,
                    "operation": operation_type,
                    "result_var": substituted_op.get('result_var'),
                    "success": True
                })
                
            except Exception as e:
                # Record failure and optionally continue or halt
                pattern.execution_state["completed_steps"].append({
                    "step": i,
                    "operation": substituted_op.get('type', 'unknown'),
                    "error": str(e),
                    "success": False
                })
                
                # For now, halt on error (could be configurable)
                raise ValueError(f"Chain execution failed at step {i}: {e}")
        
        return {
            "execution_state": pattern.execution_state,
            "result_vars": pattern.result_vars,
            "success": True
        }
    
    # Movement & Weaving Operations
    async def weave(self,
                   operations: List[WeaveOperation],
                   base_history_id: Optional[str] = None,
                   max_tokens: Optional[int] = None) -> str:
        """Create new history by weaving operations"""
        
        # 1. Create new empty history or copy base
        if base_history_id:
            base_history = await self.read_history(base_history_id)
            new_history = History(messages=base_history.messages.copy())
            new_history.metadata = base_history.metadata.copy() if base_history.metadata else {}
        else:
            new_history = History(messages=[])
            new_history.metadata = {}
        
        # 2. For each operation, dispatch to appropriate mover
        for operation in operations:
            if operation.type == WeaveOperationType.MESSAGE_RANGE:
                new_history = await self._move_message_range(new_history, operation)
            elif operation.type == WeaveOperationType.ITERATION:
                new_history = await self._move_iteration(new_history, operation)
            elif operation.type == WeaveOperationType.ITERATION_RANGE:
                new_history = await self._move_iteration_range(new_history, operation)
            elif operation.type == WeaveOperationType.FULL_HISTORY:
                new_history = await self._move_full_history(new_history, operation)
        
        # 3. Apply token limit if specified
        if max_tokens:
            new_history = await self._apply_token_limit(new_history, max_tokens)
        
        # 4. Save and return new history ID
        new_history_id = f"woven_{uuid.uuid4().hex[:8]}"
        new_history.history_id = new_history_id
        new_history.metadata['synthetic'] = True
        new_history.metadata['weave_operations'] = len(operations)
        
        await self._update_token_count(new_history)
        await self._save_history(new_history_id, new_history)
        
        return new_history_id
    
    # Internal Mover Methods (operate on history objects)
    async def _move_message_range(self, target_history: History, operation: MessageRangeOperation) -> History:
        """Internal mover - move message range from source to target history object"""
        source_history = await self.read_history(operation.source_history_id)
        
        # Validate range
        if operation.end_index > len(source_history.messages):
            operation.end_index = len(source_history.messages)
        if operation.start_index < 0:
            operation.start_index = 0
            
        # Extract message range
        messages_to_move = source_history.messages[operation.start_index:operation.end_index]
        
        # Insert into target
        if operation.target_position == -1:
            target_history.messages.extend(messages_to_move)
        else:
            for i, msg in enumerate(messages_to_move):
                target_history.messages.insert(operation.target_position + i, msg)
                
        return target_history
    
    async def _move_iteration(self, target_history: History, operation: IterationOperation) -> History:
        """Internal mover - move single iteration from source to target history object"""
        source_history = await self.read_history(operation.source_history_id)
        
        # Get iteration messages using ViewHistoryTool approach
        if hasattr(source_history, 'iterations') and source_history.iterations:
            iterations = source_history.iterations
            iteration_key = f"iteration_{operation.iteration_number}"
            
            if iteration_key in iterations:
                iteration_messages = iterations[iteration_key]
                
                # Filter system messages if needed
                if not operation.include_system:
                    iteration_messages = [m for m in iteration_messages if not isinstance(m, SystemMessage)]
                
                # Insert into target
                if operation.target_position == -1:
                    target_history.messages.extend(iteration_messages)
                else:
                    for i, msg in enumerate(iteration_messages):
                        target_history.messages.insert(operation.target_position + i, msg)
        
        return target_history
    
    async def _move_iteration_range(self, target_history: History, operation: IterationRangeOperation) -> History:
        """Internal mover - move iteration range from source to target history object"""
        source_history = await self.read_history(operation.source_history_id)
        
        # Get iterations in range
        if hasattr(source_history, 'iterations') and source_history.iterations:
            iterations = source_history.iterations
            
            all_messages = []
            for iter_num in range(operation.start_iteration, operation.end_iteration + 1):
                iteration_key = f"iteration_{iter_num}"
                if iteration_key in iterations:
                    iteration_messages = iterations[iteration_key]
                    
                    # Filter system messages if needed
                    if not operation.include_system:
                        iteration_messages = [m for m in iteration_messages if not isinstance(m, SystemMessage)]
                    
                    all_messages.extend(iteration_messages)
            
            # Insert into target
            if operation.target_position == -1:
                target_history.messages.extend(all_messages)
            else:
                for i, msg in enumerate(all_messages):
                    target_history.messages.insert(operation.target_position + i, msg)
        
        return target_history
    
    async def _move_full_history(self, target_history: History, operation: FullHistoryOperation) -> History:
        """Internal mover - move entire history using specified strategy"""
        source_history = await self.read_history(operation.source_history_id)
        
        if operation.strategy == "append":
            if operation.target_position == -1:
                target_history.messages.extend(source_history.messages)
            else:
                for i, msg in enumerate(source_history.messages):
                    target_history.messages.insert(operation.target_position + i, msg)
                    
        elif operation.strategy == "interleave":
            # Interleave with existing messages
            max_len = max(len(target_history.messages), len(source_history.messages))
            new_messages = []
            
            for i in range(max_len):
                if i < len(target_history.messages):
                    new_messages.append(target_history.messages[i])
                if i < len(source_history.messages):
                    new_messages.append(source_history.messages[i])
                    
            target_history.messages = new_messages
            
        elif operation.strategy == "merge":
            # Merge by message type
            all_messages = target_history.messages + source_history.messages
            
            systems = [m for m in all_messages if isinstance(m, SystemMessage)]
            humans = [m for m in all_messages if isinstance(m, HumanMessage)]
            ais = [m for m in all_messages if isinstance(m, AIMessage)]
            tools = [m for m in all_messages if isinstance(m, ToolMessage)]
            
            target_history.messages = systems + humans + ais + tools
        
        return target_history
    
    # Chain Operation Execution Methods
    async def _execute_history_extract(self, operation: Dict[str, Any]) -> Union[str, List[str]]:
        """Execute history extract operation with complete logic"""
        source_history_id = operation['source_history_id']
        kw = operation['kw']
        get_all = operation.get('get_all', False)
        return_last_instance_only = operation.get('return_last_instance_only', False)
        
        # Load source history
        source_history = await self.read_history(source_history_id)
        
        # Check if history has agent status and extracted content
        if (not source_history.agent_status or 
            not source_history.agent_status.extracted_content):
            raise ValueError(f"No extracted content found in history {source_history_id}")
        
        extracted_content = source_history.agent_status.extracted_content
        
        # 1. Auto-resolve kw_1 to kw
        if kw.endswith("_1"):
            kw = kw[:-2]
        
        # 2. Find all variants of the keyword
        variants = [k for k in extracted_content.keys() if k == kw or k.startswith(f"{kw}_")]
        
        # 3. Apply resolution logic
        if len(variants) == 0:
            raise ValueError(f"No extract found for keyword '{kw}' in {source_history_id}")
        
        elif len(variants) == 1:
            # Single occurrence - return it
            return extracted_content[variants[0]]
        
        elif get_all:
            # Return all variants in order
            def sort_key(variant):
                if variant == kw:
                    return 0  # Base keyword comes first
                elif '_' in variant:
                    return int(variant.split('_')[1])
                else:
                    return 999  # Fallback
            
            sorted_variants = sorted(variants, key=sort_key)
            return [extracted_content[v] for v in sorted_variants]
        
        elif return_last_instance_only:
            # Get highest numbered variant
            def variant_number(variant):
                if variant == kw:
                    return 1  # Base keyword is like _1
                elif '_' in variant:
                    return int(variant.split('_')[1])
                else:
                    return 1
            
            last_variant = max(variants, key=variant_number)
            return extracted_content[last_variant]
        
        else:
            # Multiple occurrences - raise helpful error
            variant_examples = [v for v in variants if v != kw][:3]  # Show up to 3 examples
            example_text = ", ".join(f"`{v}`" for v in variant_examples)
            raise ValueError(
                f"There are {len(variants)} occurrences of '{kw}' in {source_history_id}. "
                f"Specify which one (like {example_text}, for example.)"
            )
    
    # Operation Chaining (LCEL Integration)
    def chain(self, *operations) -> Runnable:
        """Chain context operations using LangChain LCEL"""
        return RunnableSequence(*operations)
    
    # Helper Methods
    async def _process_content(self, content: Any, image_url: Optional[str] = None) -> str:
        """Process different content types into string format"""
        
        if isinstance(content, str):
            processed = content
        elif isinstance(content, Path):
            # Read file content
            with open(content, 'r') as f:
                processed = f.read()
        elif isinstance(content, bytes):
            # Decode bytes
            processed = content.decode('utf-8')
        else:
            # Convert to JSON string
            processed = json.dumps(content, default=str)
        
        # Add image reference if provided
        if image_url:
            processed = f"{processed}\n\n[Image: {image_url}]"
            
        return processed
    
    async def _update_token_count(self, history: History):
        """Estimate token count for history"""
        total_tokens = 0
        for message in history.messages:
            # Rough estimation: 1 token per 4 characters
            total_tokens += len(str(message.content)) // 4
            
        if history.metadata is None:
            history.metadata = {}
        history.metadata['token_count'] = total_tokens
        history.metadata['updated_at'] = datetime.now().isoformat()
    
    async def _apply_token_limit(self, history: History, max_tokens: int) -> History:
        """Truncate history to fit within token limit"""
        current_tokens = 0
        kept_messages = []
        
        # Keep system messages first
        system_messages = [m for m in history.messages if isinstance(m, SystemMessage)]
        for msg in system_messages:
            current_tokens += len(str(msg.content)) // 4
            if current_tokens < max_tokens:
                kept_messages.append(msg)
        
        # Add other messages from the end (most recent first)
        other_messages = [m for m in history.messages if not isinstance(m, SystemMessage)]
        for msg in reversed(other_messages):
            msg_tokens = len(str(msg.content)) // 4
            if current_tokens + msg_tokens < max_tokens:
                current_tokens += msg_tokens
                kept_messages.insert(-len(system_messages), msg)
            else:
                break
        
        history.messages = kept_messages
        return history
    
    async def _save_history(self, history_id: str, history: History):
        """Save history to storage"""
        history_file = self.storage_path / f"{history_id}.json"
        
        # Convert to serializable format
        data = {
            'history_id': history_id,
            'metadata': history.metadata or {},
            'messages': []
        }
        
        for msg in history.messages:
            msg_data = {
                'type': msg.__class__.__name__.lower().replace('message', ''),
                'content': msg.content
            }
            
            if isinstance(msg, ToolMessage):
                msg_data['tool_call_id'] = msg.tool_call_id
                
            data['messages'].append(msg_data)
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# Context Engineering Patterns Base Class
class ContextEngineeringPattern:
    """Base class for reusable context engineering templates"""
    
    def __init__(self, name: str, domain: str, context_manager: ContextManager, **config):
        self.name = name
        self.domain = domain
        self.context_manager = context_manager
        self.config = config
    
    async def generate(self, **kwargs) -> History:
        """Generate synthetic history using this pattern"""
        # Separate metadata kwargs from pattern-specific kwargs
        metadata_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['tags', 'provider_compatibility']}
        
        history = await self.context_manager.create_synthetic_history(
            name=f"{self.name}_{uuid.uuid4().hex[:8]}",
            domain=self.domain,
            pattern=self.name,
            **metadata_kwargs
        )
        
        return await self.build_context(history, **kwargs)
    
    async def build_context(self, history: History, **kwargs) -> History:
        """Override this method to implement the pattern"""
        raise NotImplementedError("Subclasses must implement build_context")


# Example Pattern Implementations
class MentorshipPattern(ContextEngineeringPattern):
    """Creates expert-mentoring-novice conversation patterns"""
    
    async def build_context(self, history: History, **kwargs) -> History:
        expert_domain = kwargs.get('expert_domain', self.domain)
        difficulty = kwargs.get('difficulty', 'intermediate')
        
        # Add expert system prompt
        await self.context_manager.inject(
            history.history_id,
            f"You are an expert {expert_domain} mentor. Guide the student with patience and clear explanations.",
            message_type="system"
        )
        
        # Add student introduction
        await self.context_manager.inject(
            history.history_id,
            f"I'm a {difficulty} level student looking to improve my {expert_domain} skills. Can you help me?",
            message_type="human"
        )
        
        # Add mentor response
        await self.context_manager.inject(
            history.history_id,
            f"Absolutely! I'd be happy to help you develop your {expert_domain} skills. What specific area would you like to focus on first?",
            message_type="ai"
        )
        
        return await self.context_manager.read_history(history.history_id)


class DebuggingPattern(ContextEngineeringPattern):
    """Creates systematic debugging conversation flows"""
    
    async def build_context(self, history: History, **kwargs) -> History:
        language = kwargs.get('language', 'Python')
        error_type = kwargs.get('error_type', 'general')
        
        # Add debugging expert system prompt
        await self.context_manager.inject(
            history.history_id,
            f"You are a {language} debugging expert. Help identify and fix code issues systematically.",
            message_type="system"
        )
        
        # Add problem statement
        await self.context_manager.inject(
            history.history_id,
            f"I'm having a {error_type} issue with my {language} code. Can you help me debug it?",
            message_type="human"
        )
        
        # Add structured debugging response
        await self.context_manager.inject(
            history.history_id,
            "I'd be happy to help you debug your code! Please share:\n1. The code that's causing issues\n2. The error message (if any)\n3. What you expected to happen\n4. What actually happened\n\nThis will help me give you the most targeted assistance.",
            message_type="ai"
        )
        
        return await self.context_manager.read_history(history.history_id)


# Usage Example
async def main():
    """Example usage of ContextManager and patterns"""
    
    # Initialize context manager
    cm = ContextManager()
    
    # Create a synthetic history
    history = await cm.create_synthetic_history(
        name="coding_tutorial",
        domain="programming",
        tags=["python", "tutorial", "beginner"]
    )
    
    print(f"Created history: {history.history_id}")
    
    # Use mentorship pattern
    mentorship = MentorshipPattern("coding_mentor", "programming", cm)
    mentor_history = await mentorship.generate(
        expert_domain="Python programming",
        difficulty="beginner"
    )
    
    print(f"Generated mentorship context: {mentor_history.history_id}")
    print(f"Messages: {len(mentor_history.messages)}")
    print(f"Token estimate: {mentor_history.metadata.get('token_count', 0)}")


if __name__ == "__main__":
    asyncio.run(main())