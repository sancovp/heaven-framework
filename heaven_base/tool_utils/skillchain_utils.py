skillchain_instructions = """
Skillchain Expression Language: Rules and Syntax
-------------------------------------------------

The Skillchain Expression Language provides a concise way to describe complex, multi-step workflows using agents, tools, and emoji variables. It is designed for clarity, modularity, and easy parsing by both humans and LLMs.

Rules:
- Emoji Assignment: Assign any agent or tool to an emoji variable using `emoji = entity_name` at the start of the skillchain, with no curly braces.
- Tool Calls: Use `>>` to represent an agent or tool invoking another tool/entity or performing an action.
- Return Values: Use `=` to bind the output of a call to a variable or to indicate a result.
- Final Output: Use `==>` to indicate the final output of the workflow.
- Loops: Use `🔁` before a block to indicate a loop, with variables inside the loop. The variable in the parentheses is available in the loop's body.
- Unique Tools: Differentiate multiple tool instances with emoji suffixes like `🛠️_1`, `🛠️_2`, etc.
- Variable Usage: Variables can be referenced in calls, including within loops. Any variable created as a result in a previous step (e.g., `result_var`) may be used as an argument in subsequent steps, including when calling a different agent or chain. This allows agents and chains to pass outputs to each other explicitly.
- Hierarchy/Swarm: Use distinct emojis for orchestrators, managers, workers, etc. Assign these as variables with the same syntax.
- All assignments and calls must use valid emoji variables previously defined in the chain.
- No undefined variables or duplicate emoji variable assignments allowed.
- All tool/entity names must map to a known set of agents/tools.
- The expression must end with a final return (`==>`).
"""
import re
from pydantic import BaseModel, validator, root_validator
from typing import List, Dict, Any, Optional
from ..prompts.prompt_blocks.prompt_block_utils import write_prompt_block 


def section_acronym(section):
    # Split PascalCase or snake_case into words and get first letter of each
    words = re.findall(r'[A-Z][a-z]*|[a-z]+', section)
    return ''.join(word[0].upper() for word in words)

def get_acronym(phrase):
    sections = phrase.split('.')  # Split on periods
    acronyms = [section_acronym(section) for section in sections]
    return '.'.join(acronyms)


class SkillchainStep(BaseModel):
    caller: str  # emoji variable
    callee: str  # emoji variable or tool name
    action: str  # e.g., 'call', 'loop'
    args: Optional[Dict[str, Any]] = None
    result: Optional[str] = None  # assigned variable (emoji or string)
    loop_var: Optional[str] = None  # for loops

class Skillchain(BaseModel):
    name: str
    domain: str
    subdomain: str
    assignments: Dict[str, str]  # emoji variable -> entity/tool name
    steps: List[SkillchainStep]
    final_return: str  # final output variable
    write_prompt_block: bool
  
    @validator("assignments")
    def unique_emojis(cls, v):
        if len(set(v.keys())) != len(v):
            raise ValueError("Duplicate emoji variable assignments found.")
        return v

def check_defined_variables(cls, values):
        assignments = values.get("assignments", {})
        steps = values.get("steps", [])
        defined = set(assignments.keys())
        for step in steps:
            if step.caller not in defined:
                raise ValueError(f"Undefined caller variable: {step.caller}")
            if step.callee not in defined and not step.callee.startswith("🛠️"):
                raise ValueError(f"Undefined callee variable: {step.callee}")
            if step.result:
                defined.add(step.result)
        if values.get("final_return") not in defined:
            raise ValueError("Final return variable is not defined in the skillchain.")
        return values

import re

EMOJI_PATTERN = r"^[\U0001F300-\U0001F6FF][a-zA-Z0-9_]*$"

def is_emoji_var(s):
    return bool(re.match(EMOJI_PATTERN, s))

# Patch Skillchain BaseModel to enforce emoji abstraction and provide a string builder
old_check_defined_variables = Skillchain.__dict__.get('check_defined_variables')

def check_defined_variables(cls, values):
    assignments = values.get("assignments", {})
    steps = values.get("steps", [])
    defined = set(assignments.keys())
    used_vars = set()

    def check_args_defined(args, defined):
        if not args:
            return
        for val in args.values():
            # If the argument is an emoji variable and not a literal
            if isinstance(val, str) and is_emoji_var(val) and val not in defined:
                raise ValueError(f"Argument variable {val} used before definition.")

    for emoji_var in assignments.keys():
        if not is_emoji_var(emoji_var):
            raise ValueError(f"Assignment variable {emoji_var} is not a valid emoji variable.")

    for step in steps:
        if not is_emoji_var(step.caller):
            raise ValueError(f"Caller {step.caller} is not a valid emoji variable.")
        if not is_emoji_var(step.callee):
            raise ValueError(f"Callee {step.callee} is not a valid emoji variable.")
        check_args_defined(step.args, defined)
        if step.result and not is_emoji_var(step.result):
            raise ValueError(f"Result variable {step.result} is not a valid emoji variable.")
        if step.result:
            defined.add(step.result)
        # Recursively validate nested steps in loop bodies
        if step.action == "loop" and step.args and 'body' in step.args:
            for substep in step.args['body']:
                # Pass a copy of defined for scoping (or modify as needed for your scoping rules)
                check_args_defined(substep.args, defined)
                if substep.result and not is_emoji_var(substep.result):
                    raise ValueError(f"Result variable {substep.result} is not a valid emoji variable.")
                if substep.result:
                    defined.add(substep.result)
    if values.get("final_return") not in defined:
        raise ValueError("Final return variable is not defined in the skillchain.")
    return values

Skillchain.check_defined_variables = classmethod(check_defined_variables)

# Add a string builder method to Skillchain

def to_skillchain_string(self):
    lines = []
    for emoji, entity in self.assignments.items():
        lines.append(f"{emoji} = {entity}")
    for step in self.steps:
        if step.action == "loop":
            lines.append(f"🔁 (for {step.loop_var} in {step.args['iterable']}):")
            # Assume substeps indented in step.args['body'] if needed
            for s in step.args.get('body', []):
                call_str = f"  {s.caller} >> {s.callee}("
                if s.args:
                    call_str += ", ".join(f"{k}={v}" for k, v in s.args.items())
                call_str += ")"
                if s.result:
                    call_str += f" = {s.result}"
                lines.append(call_str)
        else:
            call_str = f"{step.caller} >> {step.callee}("
            if step.args:
                call_str += ", ".join(f"{k}={v}" for k, v in step.args.items())
            call_str += ")"
            if step.result:
                call_str += f" = {step.result}"
            lines.append(call_str)
    lines.append(f"{self.final_return} ==> ")
    phrase = f"{self.domain}.{self.subdomain}.{self.name}"
    header = f"\n<{phrase}>\n"
    skillchain_lines = "\n".join(lines)
    footer = f"\n</{phrase}>\n"
    skillchain_final = header + skillchain_lines + footer
    if self.write_prompt_block:
        prompt_block = write_prompt_block(name=self.name, text=skillchain_final, domain=self.domain, subdomain=self.subdomain)
        return prompt_block
    else:
        return skillchain_final
      
Skillchain.to_skillchain_string = to_skillchain_string


def make_skillchain_prompt_block(
    name: str,
    domain: str,
    subdomain: str,
    assignments: Dict[str, str],
    steps: List[SkillchainStep],
    final_return: str
) -> str:
    skillchain = Skillchain(
        name=name,
        domain=domain,
        subdomain=subdomain,
        assignments=assignments,
        steps=steps,
        final_return=final_return,
        write_prompt_block=True
    )
    return skillchain.to_skillchain_string()