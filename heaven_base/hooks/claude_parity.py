"""Claude Parity Hook — injects .claude/ rules and CLAUDE.md into Heaven agents.

PostToolUse hook that detects when an agent is working in a directory with .claude/
and injects the rules + CLAUDE.md content as a system-reminder appended to tool results.

Cooldown: re-injects every 7 tool uses per repo to avoid token bloat.
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional

from ..baseheavenagent import HookPoint, HookContext, HookRegistry

logger = logging.getLogger(__name__)

# Tools that expose file paths in their args
_PATH_ARG_KEYS = ("path", "file_path", "filename", "file")


def _extract_path_from_tool(ctx: HookContext) -> Optional[str]:
    """Extract a filesystem path from tool args."""
    tool_args = ctx.tool_args or {}
    tool_name = ctx.tool_name or ""

    # NetworkEditTool, ReadTool, etc. — check common path arg names
    for key in _PATH_ARG_KEYS:
        val = tool_args.get(key)
        if val and isinstance(val, str) and val.startswith("/"):
            return val

    # BashTool — try to extract paths from command string
    if "bash" in tool_name.lower():
        command = tool_args.get("command", "")
        # Find absolute paths in the command
        matches = re.findall(r'(/[^\s;|&"\']+)', command)
        for m in matches:
            if os.path.exists(os.path.dirname(m)) or os.path.exists(m):
                return m

    return None


def _find_claude_dir(path: str) -> Optional[str]:
    """Walk up from path to find nearest directory containing .claude/."""
    p = Path(path)
    if p.is_file():
        p = p.parent
    # Walk up max 10 levels
    for _ in range(10):
        claude_dir = p / ".claude"
        if claude_dir.is_dir():
            return str(p)
        if p.parent == p:
            break
        p = p.parent
    return None


def _read_claude_content(repo_root: str) -> Optional[str]:
    """Read .claude/ rules and CLAUDE.md, format as injection block."""
    claude_dir = os.path.join(repo_root, ".claude")
    parts = []

    # Read rules
    rules_dir = os.path.join(claude_dir, "rules")
    rule_texts = []
    if os.path.isdir(rules_dir):
        for f in sorted(os.listdir(rules_dir)):
            if f.endswith(".md"):
                try:
                    content = Path(os.path.join(rules_dir, f)).read_text()[:1500]
                    rule_texts.append(content.strip())
                except Exception:
                    pass

    if rule_texts:
        rule_separator = "\n---\n"
        parts.append(f"<rules>\n{rule_separator.join(rule_texts)}\n</rules>")

    # Read CLAUDE.md
    claude_md = os.path.join(claude_dir, "CLAUDE.md")
    if os.path.isfile(claude_md):
        try:
            md_content = Path(claude_md).read_text()[:3000]
            safe_tag = repo_root.replace("/", "_").strip("_")
            parts.append(f"<{safe_tag}-system-prompt>\n{md_content.strip()}\n</{safe_tag}-system-prompt>")
        except Exception:
            pass

    if not parts:
        return None

    return (
        f"\n<system-reminder>ATTENTION! You are working in a directory with "
        f"*CLAUDE CODE RULES*. These are rules that must be followed while working "
        f"in this directory `{repo_root}`.\n"
        + "\n".join(parts)
        + "\n</system-reminder>"
    )


def make_claude_parity_hook(cooldown: int = 7):
    """Create AFTER_TOOL_CALL hook that injects .claude/ content.

    Args:
        cooldown: Number of tool calls per repo before re-injecting (default 15)
    """
    # Track per-repo: {repo_root: tool_calls_since_last_injection}
    _repo_counters: Dict[str, int] = {}
    # Cache: {repo_root: injection_content}
    _repo_cache: Dict[str, Optional[str]] = {}

    def _claude_parity(ctx: HookContext):
        path = _extract_path_from_tool(ctx)
        if not path:
            return

        repo_root = _find_claude_dir(path)
        if not repo_root:
            return

        # Increment counter for this repo
        count = _repo_counters.get(repo_root, cooldown)  # first hit = inject immediately
        count += 1
        _repo_counters[repo_root] = count

        if count < cooldown:
            return

        # Reset counter
        _repo_counters[repo_root] = 0

        # Get or build injection content
        if repo_root not in _repo_cache:
            _repo_cache[repo_root] = _read_claude_content(repo_root)

        injection = _repo_cache[repo_root]
        if not injection:
            return

        # Append to tool result
        if ctx.tool_result and hasattr(ctx.tool_result, 'output'):
            ctx.tool_result.output = (ctx.tool_result.output or "") + injection
            logger.debug(f"Claude parity injected for {repo_root}")

    return _claude_parity


def register_claude_parity(registry: HookRegistry, cooldown: int = 7):
    """Register the claude_parity hook on AFTER_TOOL_CALL."""
    registry.register(HookPoint.AFTER_TOOL_CALL, make_claude_parity_hook(cooldown))
