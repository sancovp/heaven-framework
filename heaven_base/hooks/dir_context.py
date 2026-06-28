"""Directory-context autoloader — the capped, system-prompt-only successor to `claude_parity`.

Reactively loads a working directory's instruction files into the SYSTEM PROMPT:
  - `AGENTS.md` / `CLAUDE.md` directly in the dir (the agents.md standard + the Claude convention)
  - `.claude/CLAUDE.md` + `.claude/rules/*.md`   (back-compat)
  - `.heaven/AGENTS.md` + `.heaven/rules/*.md`     (heaven-native)
discovered by walking UP from the agent's working path to the repo root (`.git`) — root→cwd, **nearest
refines** — de-duplicated across `.claude` vs `.heaven` (identical files included once), each file labeled
with its source path.

Two hooks + one piece of state:
  AFTER_TOOL_CALL  (detector)        — a tool's path reveals the current dir → set `active` dir. You cannot
                                       preload this; a Read *activates* it, exactly like Claude activates a
                                       CLAUDE.md only when it reads that dir. Switching dirs swaps it.
  BEFORE_SYSTEM_PROMPT (renderer)    — inject `active`'s context into the system prompt, idempotently (strip
                                       the prior block, re-append) and hot (re-read every turn). SYSTEM PROMPT
                                       ONLY — never appended to tool results (that pays for the same tokens
                                       twice; `claude_parity` did that — this supersedes it).

Caps: <= 40k chars/file, <= 400k chars total (~100k tokens) per render. On exceed: a LOUD notice to the LLM
naming the file — NEVER silent truncation.
"""
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

from ..baseheavenagent import HookPoint, HookContext, HookRegistry

logger = logging.getLogger(__name__)

PER_FILE_CHARS = 40_000          # ~10k tokens per file
TOTAL_CHARS = 400_000            # ~100k tokens per render
_MAX_WALKUP = 12

_PATH_ARG_KEYS = ("path", "file_path", "filename", "file", "directory", "cwd")


def _extract_path_from_tool(ctx: HookContext) -> Optional[str]:
    """Pull a filesystem path out of the tool's args (or a bash command string)."""
    tool_args = ctx.tool_args or {}
    tool_name = (ctx.tool_name or "").lower()
    for key in _PATH_ARG_KEYS:
        val = tool_args.get(key)
        if val and isinstance(val, str) and val.startswith("/"):
            return val
    if "bash" in tool_name:
        command = tool_args.get("command", "") or ""
        for m in re.findall(r'(/[^\s;|&"\']+)', command):
            if os.path.exists(m) or os.path.exists(os.path.dirname(m)):
                return m
    return None


def _walk_levels(work_dir: str) -> list:
    """Dirs from repo-root (or filesystem root / max depth) down to work_dir — nearest LAST."""
    p = Path(work_dir)
    if p.is_file():
        p = p.parent
    chain, cur = [], p
    for _ in range(_MAX_WALKUP):
        chain.append(cur)
        if (cur / ".git").exists() or cur.parent == cur:
            break
        cur = cur.parent
    return list(reversed(chain))  # root-first → cwd-last (cwd refines)


def collect_context(work_dir: str) -> Optional[str]:
    """Assemble the dir-context block for `work_dir` (deduped, source-labeled, capped, loud-on-exceed)."""
    parts, notices, seen, total = [], [], set(), 0

    def add(path: str):
        nonlocal total
        try:
            content = Path(path).read_text()
        except Exception:
            return
        h = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
        if h in seen:                                  # identical file in .claude AND .heaven → once
            return
        if len(content) > PER_FILE_CHARS:
            notices.append(f"{path} append skipped — {len(content)} chars exceeds the {PER_FILE_CHARS//1000}k/file limit")
            return
        if total + len(content) > TOTAL_CHARS:
            notices.append(f"{path} append skipped — would exceed the {TOTAL_CHARS//1000}k-char (~100k token) total")
            return
        seen.add(h); total += len(content)
        parts.append(f"### [from {path}]\n{content.strip()}")

    for lvl in _walk_levels(work_dir):
        for fn in ("AGENTS.md", "CLAUDE.md"):
            fp = lvl / fn
            if fp.is_file():
                add(str(fp))
        for sub, root_md in ((".claude", "CLAUDE.md"), (".heaven", "AGENTS.md")):
            d = lvl / sub
            if not d.is_dir():
                continue
            if (d / root_md).is_file():
                add(str(d / root_md))
            rules = d / "rules"
            if rules.is_dir():
                for r in sorted(rules.glob("*.md")):
                    add(str(r))

    if not parts and not notices:
        return None
    body = "\n\n".join(parts)
    if notices:
        body += "\n\n<system-reminder>\n" + "\n".join(notices) + "\n</system-reminder>"
    return (
        f'\n\n<DIR_CONTEXT root="{work_dir}">\n'
        f"You are working in `{work_dir}`. The instruction files below apply here "
        f"(nearest-last refines; each is labeled with its source path):\n\n"
        f"{body}\n</DIR_CONTEXT>"
    )


def make_dir_context_hooks():
    """Return (detector, renderer) sharing one `active` dir. Register both via register_dir_context()."""
    state = {"active": None}

    def detector(ctx: HookContext):
        try:
            path = _extract_path_from_tool(ctx)
            if not path:
                return
            d = path if os.path.isdir(path) else os.path.dirname(path)
            if d and os.path.isdir(d):
                state["active"] = d        # switch: later tool in a new dir replaces the old
        except Exception:
            pass  # detection is best-effort, never block the agent

    def renderer(ctx: HookContext):
        try:
            current = ctx.data.get("system_prompt", ctx.prompt or "")
            current = re.sub(r'\n*<DIR_CONTEXT\b.*?</DIR_CONTEXT>', '', current, flags=re.DOTALL)  # idempotent
            root = state["active"]
            block = collect_context(root) if root else None        # hot: re-read each turn
            ctx.data["system_prompt"] = current + block if block else current
        except Exception:
            pass

    return detector, renderer


def register_dir_context(registry: HookRegistry):
    """Wire the dir-context autoloader: AFTER_TOOL_CALL detector + BEFORE_SYSTEM_PROMPT renderer."""
    detector, renderer = make_dir_context_hooks()
    registry.register(HookPoint.AFTER_TOOL_CALL, detector)
    registry.register(HookPoint.BEFORE_SYSTEM_PROMPT, renderer)
