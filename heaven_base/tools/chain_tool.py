"""
ChainTool — BaseHeavenTool for executing SDNA chain ontology objects.

Usage:
    from heaven_base.tools.chain_tool import ChainTool
    
    # Add to agent config:
    tools=[BashTool, ChainTool, ...]
"""

from typing import Optional
from ..make_heaven_tool_from_docstring import make_heaven_tool_from_docstring


async def execute_chain(
    chain_json: str,
    context_json: str = "{}",
    describe_only: bool = False,
) -> str:
    """Execute an SDNA chain ontology object from a JSON specification.

    Supports Link, Chain, EvalChain, and Compiler objects.
    Build chains using LinkConfig specifications, then execute them.
    On construction errors, read the chain-tool skill.

    Args:
        chain_json: JSON chain specification. Format: {"type": "chain", "name": "my_chain", "links": [{"type": "config_link", "name": "step1", "goal": "analyze"}, ...]}. Supported types: "config_link", "chain", "eval_chain".
        context_json: JSON object with input context for the chain execution. Default: empty object.
        describe_only: If true, return chain structure description without executing. Use this to verify chain composition before running.
    """
    import json
    import traceback

    try:
        chain_spec = json.loads(chain_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "construction_error", "error": f"Invalid chain_json: {e}", "hint": "Read the chain-tool skill to learn how to compose chains correctly."})

    try:
        context = json.loads(context_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "construction_error", "error": f"Invalid context_json: {e}", "hint": "context_json must be a valid JSON object."})

    # Build chain from spec
    try:
        chain = _build_chain_from_spec(chain_spec)
    except (TypeError, ValueError, KeyError) as e:
        return json.dumps({"status": "construction_error", "error": str(e), "traceback": traceback.format_exc(), "hint": "Read the chain-tool skill to learn how to compose chains correctly."})

    if describe_only:
        return json.dumps({"status": "described", "description": chain.describe(), "type": type(chain).__name__}, indent=2)

    # Execute
    try:
        result = await chain.execute(context)
        return json.dumps({"status": result.status.value, "context": result.context, "error": result.error, "resume_path": result.resume_path}, indent=2, default=str)
    except (TypeError, AttributeError, ValueError) as e:
        return json.dumps({"status": "construction_error", "error": str(e), "traceback": traceback.format_exc(), "hint": "Read the chain-tool skill to learn how to compose chains correctly."})
    except Exception as e:
        return json.dumps({"status": "runtime_error", "error": str(e), "traceback": traceback.format_exc()})


def _build_chain_from_spec(spec: dict):
    """Deserialize a JSON spec into chain ontology objects."""
    from sdna.chain_ontology import Chain, EvalChain, ConfigLink, LinkConfig

    chain_type = spec.get("type", "config_link")

    if chain_type == "config_link":
        config = LinkConfig(
            name=spec.get("name", ""),
            goal=spec.get("goal", ""),
            system_prompt=spec.get("system_prompt", ""),
            model=spec.get("model", ""),
            provider=spec.get("provider", ""),
            temperature=spec.get("temperature", 0.7),
            max_turns=spec.get("max_turns", 10),
            permission_mode=spec.get("permission_mode", "default"),
            allowed_tools=spec.get("allowed_tools", []),
            mcp_servers=spec.get("mcp_servers", {}),
            skills=spec.get("skills", ""),
        )
        return ConfigLink(config)

    elif chain_type == "chain":
        links = [_build_chain_from_spec(link_spec) for link_spec in spec.get("links", [])]
        return Chain(spec.get("name", "chain"), links)

    elif chain_type == "eval_chain":
        links = [_build_chain_from_spec(link_spec) for link_spec in spec.get("links", [])]
        evaluator = _build_chain_from_spec(spec["evaluator"]) if spec.get("evaluator") else None
        return EvalChain(
            chain_name=spec.get("name", "eval_chain"),
            links=links,
            evaluator=evaluator,
            max_cycles=spec.get("max_cycles", 3),
            approval_key=spec.get("approval_key", "approved"),
        )

    else:
        raise ValueError(f"Unknown chain type: {chain_type}. Use: config_link, chain, eval_chain")


# Auto-generate BaseHeavenTool from the function
ChainTool = make_heaven_tool_from_docstring(execute_chain, tool_name="ChainTool")
