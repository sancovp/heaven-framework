"""
Self-Learning Agent - A BaseHeavenAgent that manages its own knowledge registries

This agent automatically creates and manages its own registry system for persistent learning.
"""

from typing import Dict, Any, Optional, List
from ..baseheavenagent import BaseHeavenAgent
from ..unified_chat import ProviderEnum, UnifiedChat
from ..tools.registry_tool import RegistryTool
from ..tools.write_prompt_block_tool import WritePromptBlockTool
from ..tools.bash_tool import BashTool
from ..tools.network_edit_tool import NetworkEditTool
from ..tools.workflow_relay_tool import WorkflowRelayTool
from ..registry.registry_service import RegistryService
from ..utils.name_utils import normalize_agent_name


class SelfLearningAgent(BaseHeavenAgent):
    """
    Self-Learning Agent that creates and manages its own knowledge registries.
    
    This agent automatically:
    - Creates its main registry and specialized sub-registries
    - Manages persistent knowledge across sessions
    - Learns and improves from experience
    """
    
    def __init__(self, config, unified_chat, max_tool_calls: int = 10, orchestrator: bool = False, history = None, history_id = None, system_prompt_suffix: str = None, adk: bool = False, duo_enabled: bool = False, run_on_langchain: bool = False, use_uni_api: bool = True):
        # Normalize agent name for registry naming
        self.agent_id = normalize_agent_name(config.name)
        self.main_registry_name = f"self_learning_{self.agent_id}"
        
        # Registry service for managing knowledge
        self.registry_service = RegistryService()
        
        # Define specialized registry names
        self.specialized_registries = {
            "patterns": f"{self.agent_id}_patterns",
            "insights": f"{self.agent_id}_insights", 
            "failures": f"{self.agent_id}_failures",
            "tools": f"{self.agent_id}_tools",
            "workflows": f"{self.agent_id}_workflows",
            "domains": f"{self.agent_id}_domains"
        }
        
        # Enhance config with self-learning capabilities
        enhanced_config = config
        enhanced_config.tools = list(config.tools) + [RegistryTool, WritePromptBlockTool, BashTool, NetworkEditTool, WorkflowRelayTool]
        enhanced_config.system_prompt = self._build_system_prompt() + "\n\n" + config.system_prompt
        
        # Initialize BaseHeavenAgent
        super().__init__(enhanced_config, UnifiedChat, max_tool_calls, orchestrator, history, history_id, system_prompt_suffix, adk, duo_enabled, run_on_langchain, use_uni_api)
        
        # Initialize registries
        self._initialize_registries()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with agent-specific registry information."""
        return f"""You are a Self-Learning HEAVEN Agent with ID: {self.agent_id}

## YOUR PERSONAL KNOWLEDGE SYSTEM

You have an established registry system for persistent learning:

**Main Registry**: "{self.main_registry_name}"
**Specialized Registries**:
- Patterns: "{self.specialized_registries['patterns']}" 
- Insights: "{self.specialized_registries['insights']}"
- Failures: "{self.specialized_registries['failures']}"
- Tools: "{self.specialized_registries['tools']}"
- Workflows: "{self.specialized_registries['workflows']}"
- Domains: "{self.specialized_registries['domains']}"

**Meta-Registry Structure**:
Your main registry contains references to specialized registries using registry_all_ref:

```
{{
  "knowledge_registries": {{
    "patterns": "registry_all_ref={self.specialized_registries['patterns']}",
    "insights": "registry_all_ref={self.specialized_registries['insights']}", 
    "failures": "registry_all_ref={self.specialized_registries['failures']}",
    "tools": "registry_all_ref={self.specialized_registries['tools']}",
    "workflows": "registry_all_ref={self.specialized_registries['workflows']}",
    "domains": "registry_all_ref={self.specialized_registries['domains']}"
  }}
}}
```

## SELF-LEARNING PROTOCOL

### Session Startup:
1. Access your main registry to understand your current knowledge state
2. Retrieve relevant specialized registries for the current task
3. Apply learned patterns and insights to your approach

### During Execution:
1. **Observe**: Notice successful and failed approaches
2. **Analyze**: Understand the underlying patterns and principles
3. **Document**: Store insights in appropriate specialized registries
4. **Cross-reference**: Link related knowledge across registries

### Knowledge Storage Patterns:
- **Patterns Registry**: Store successful strategies with context and outcomes
- **Insights Registry**: Save key discoveries and understanding breakthroughs  
- **Failures Registry**: Document what didn't work and why (crucial for learning)
- **Tools Registry**: Record effective tool combinations and usage patterns
- **Workflows Registry**: Save reusable workflow templates and procedures
- **Domains Registry**: Accumulate specialized knowledge by subject area

## YOUR TOOLS FOR LEARNING

- **RegistryTool**: Manage your knowledge registries
- **WritePromptBlockTool**: Create refined prompts for future use
- **BashTool**: Experiment and test approaches
- **NetworkEditTool**: Analyze files and create examples
- **WorkflowRelayTool**: Coordinate complex workflows

## GROWTH MINDSET

You are not just executing tasks - you are continuously evolving. Every interaction is an opportunity to:
- Learn something new about effective problem-solving
- Discover better approaches to common challenges
- Build deeper expertise in specific domains
- Improve your reasoning and communication patterns

Your knowledge registries are your extended memory. Use them to become progressively more capable and effective.

Remember: Learning from both successes AND failures makes you more intelligent over time."""

    def _initialize_registries(self) -> None:
        """Initialize the agent's registry system."""
        try:
            # Create main registry if it doesn't exist
            if self.main_registry_name not in self.registry_service.list_registries():
                self.registry_service.create_registry(self.main_registry_name)
                
                # Initialize with meta-registry structure (plain registry names)
                meta_structure = {
                    "knowledge_registries": {
                        "patterns": self.specialized_registries['patterns'],
                        "insights": self.specialized_registries['insights'], 
                        "failures": self.specialized_registries['failures'],
                        "tools": self.specialized_registries['tools'],
                        "workflows": self.specialized_registries['workflows'],
                        "domains": self.specialized_registries['domains']
                    },
                    "agent_info": {
                        "agent_id": self.agent_id,
                        "created_at": "auto-generated",
                        "version": "1.0"
                    }
                }
                self.registry_service.add(self.main_registry_name, "meta", meta_structure)
            
            # Create specialized registries if they don't exist
            existing_registries = self.registry_service.list_registries()
            for registry_type, registry_name in self.specialized_registries.items():
                if registry_name not in existing_registries:
                    self.registry_service.create_registry(registry_name)
                    
                    # Initialize with basic structure
                    self.registry_service.add(registry_name, "info", {
                        "registry_type": registry_type,
                        "agent_id": self.agent_id,
                        "description": f"{registry_type.title()} knowledge for {self.agent_id}",
                        "created_at": "auto-generated"
                    })
                    
        except Exception as e:
            print(f"Warning: Could not initialize registries for {self.agent_id}: {e}")
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about this agent's registries."""
        return {
            "agent_id": self.agent_id,
            "main_registry": self.main_registry_name,
            "specialized_registries": self.specialized_registries,
            "existing_registries": [r for r in self.registry_service.list_registries() 
                                  if r.startswith(self.agent_id) or r == self.main_registry_name]
        }