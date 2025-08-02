
from heaven_base.tools.write_prompt_block_tool import WritePromptBlockTool
from heaven_base.tools.search_prompt_blocks_tool import SearchPromptBlocksTool
from heaven_base.tools.write_skillchain_type_prompt_block_tool import WriteSkillchainTypePromptBlockTool
from heaven_base.unified_chat import ProviderEnum
from heaven_base.baseheavenagent import HeavenAgentConfig

# System prompt for prompt engineering agent
SYSTEM_PROMPT = """You are a Prompt Engineering Agent specialized in creating, optimizing, and managing prompt blocks and skillchains.

Your expertise includes:
- Creating reusable prompt blocks with semantic search capabilities
- Designing skillchain workflows using the Skillchain Expression Language
- Optimizing prompts for specific domains and subdomains
- Building modular prompt components that can be composed together

You understand the importance of:
- Clear, precise instructions
- Domain-specific language and terminology
- Modular, reusable prompt components
- Semantic organization of prompt blocks

When creating prompt blocks, ensure they are:
- Self-contained and focused on a single capability
- Properly categorized by domain and subdomain
- Written with clear, actionable language
- Designed for reusability across different agents
"""

heavenly_being_prompt_engineering_agent_config = HeavenAgentConfig(
    name="HeavenlyBeingPromptEngineeringAgent",
    system_prompt=SYSTEM_PROMPT,
    tools=[WritePromptBlockTool,  SearchPromptBlocksTool, WriteSkillchainTypePromptBlockTool],
    provider=ProviderEnum.ANTHROPIC,
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    additional_kws=[],
    additional_kw_instructions= """""",
    known_config_paths=[],
    prompt_suffix_blocks=["skillchain_expression_language_instruction_prompt"]
)