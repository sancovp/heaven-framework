"""hook_script — the SCRIPT hook contract (Claude Code parity: `type:'command'`).

A devdir hook may be a STANDALONE SCRIPT instead of an in-process module. A hook is a script
with an envelope: JSON on STDIN, JSON on STDOUT, imports NOTHING of heaven — any language.
It declares its events in a TEXT HEADER, so the loader never has to exec it to know what it
binds (the navigability law: read the file, know everything):

    # hook-events: before_tool_call, after_run
    # hook-timeout: 10          (optional; seconds; default 10)

Events are heaven's HookPoint names or values (before_run · after_run · before_iteration ·
after_iteration · before_tool_call · after_tool_call · before_system_prompt ·
on_block_report · on_error).

THE INPUT envelope (one JSON object on stdin):
    { "hook_event_name": "<point value>", "agent_name": str, "cwd": str,
      "prompt": str, "iteration": int,
      "tool_name": str, "tool_input": dict, "tool_result": str, "error": str }
(absent fields omitted; values JSON-safe-coerced.)

THE OUTPUT envelope (one JSON object on stdout; every field optional; empty/no output = no-op):
    { "decision": "approve" | "block",      ← block at before_tool_call = the heaven veto
      "reason": str,                         ← the block message
      "systemMessage": str,                  ← appended to the system prompt next render
      "hookSpecificOutput": { "additionalContext": str } }
EXIT CODE 2 also means block, with stderr as the reason (the Claude Code command-hook
convention, kept for parity). Any other failure — timeout, crash, bad JSON — is FAIL-OPEN:
logged, the turn untouched.

The other Claude Code mechanisms are RECORDED, not owed (CLAUDE-PARITY.md §1): `prompt` /
`http` / `agent` are declarable LLM/remote hooks there; here only `command` (this file) and
the in-process module forms (register(registry) / hook(ctx)+POINT) exist.

Consumer: `baseheavenagent._register_devdir_hook_file` — a hook file whose header declares
`hook-events:` registers through THIS module; otherwise it imports as a module hook.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10
_HEADER_SCAN_LINES = 30          # the declaration must live near the top — headers, not burials
_RESULT_CAP = 4000               # tool_result truncation in the payload


class ScriptHookSpecificOutput(BaseModel):
    additionalContext: Optional[str] = None


class ScriptVerdict(BaseModel):
    """The parsed stdout envelope of one script-hook run."""
    decision: Optional[str] = None            # 'approve' | 'block'
    reason: Optional[str] = None
    systemMessage: Optional[str] = None
    hookSpecificOutput: Optional[ScriptHookSpecificOutput] = None

    @property
    def blocks(self) -> bool:
        return self.decision == "block"

    @property
    def inject_text(self) -> str:
        parts = []
        if self.systemMessage:
            parts.append(self.systemMessage)
        if self.hookSpecificOutput and self.hookSpecificOutput.additionalContext:
            parts.append(self.hookSpecificOutput.additionalContext)
        return "\n".join(parts)


class ScriptHookDecl(BaseModel):
    """What a script file's text header declares."""
    events: List[str] = Field(default_factory=list)
    timeout_s: int = DEFAULT_TIMEOUT_S


def parse_decl(content: str) -> Optional[ScriptHookDecl]:
    """Parse the `# hook-events:` / `# hook-timeout:` header from a file's text. Returns None
    if the file declares no events (⇒ it is a MODULE hook, not a script hook)."""
    events: List[str] = []
    timeout_s = DEFAULT_TIMEOUT_S
    for ln in content.splitlines()[:_HEADER_SCAN_LINES]:
        s = ln.strip()
        if s.lower().startswith("# hook-events:"):
            events = [e.strip() for e in s.split(":", 1)[1].split(",") if e.strip()]
        elif s.lower().startswith("# hook-timeout:"):
            try:
                timeout_s = max(1, int(s.split(":", 1)[1].strip()))
            except ValueError:
                pass
    if not events:
        return None
    return ScriptHookDecl(events=events, timeout_s=timeout_s)


def build_payload(point_value: str, ctx) -> dict:
    """The stdin envelope from a heaven HookContext. JSON-safe; absent fields omitted."""
    payload: dict = {"hook_event_name": point_value, "cwd": os.getcwd()}
    agent = getattr(ctx, "agent", None)
    name = getattr(getattr(agent, "config", None), "name", None)
    if name:
        payload["agent_name"] = str(name)
    prompt = getattr(ctx, "prompt", None)
    if prompt:
        payload["prompt"] = str(prompt)
    iteration = getattr(ctx, "iteration", None)
    if iteration is not None:
        payload["iteration"] = iteration
    tool_name = getattr(ctx, "tool_name", None)
    if tool_name:
        payload["tool_name"] = str(tool_name)
    tool_args = getattr(ctx, "tool_args", None)
    if tool_args is not None:
        try:
            payload["tool_input"] = json.loads(json.dumps(tool_args, default=str))
        except Exception:
            payload["tool_input"] = {}
    tool_result = getattr(ctx, "tool_result", None)
    if tool_result is not None:
        err = getattr(tool_result, "error", None)
        out = getattr(tool_result, "output", None)
        payload["tool_result"] = str(err if err else (out if out is not None else tool_result))[:_RESULT_CAP]
    error = getattr(ctx, "error", None)
    if error is not None:
        payload["error"] = str(error)[:_RESULT_CAP]
    return payload


def run_script(path: Path, payload: dict, timeout_s: int = DEFAULT_TIMEOUT_S) -> Optional[ScriptVerdict]:
    """One script-hook run: payload on stdin → verdict from stdout (or exit-2 = block with
    stderr as reason). FAIL-OPEN: any failure returns None and logs."""
    cmd = [str(path)] if os.access(path, os.X_OK) and path.suffix != ".py" else [sys.executable, str(path)]
    try:
        proc = subprocess.run(
            cmd, input=json.dumps(payload).encode(),
            capture_output=True, timeout=timeout_s,
        )
    except Exception as e:
        logger.info("script hook %s failed to run (fail-open): %s", path.name, e)
        return None
    if proc.returncode == 2:      # the CC command-hook convention: exit 2 = block
        return ScriptVerdict(decision="block",
                             reason=(proc.stderr.decode(errors="replace").strip()
                                     or f"blocked by hook {path.name}"))
    if proc.returncode != 0:
        logger.info("script hook %s exited %s (fail-open): %s",
                    path.name, proc.returncode, proc.stderr.decode(errors="replace")[:500])
        return None
    raw = proc.stdout.decode(errors="replace").strip()
    if not raw:
        return None
    try:
        return ScriptVerdict.model_validate(json.loads(raw))
    except Exception as e:
        logger.info("script hook %s emitted unparseable output (fail-open): %s", path.name, e)
        return None


def apply_verdict(ctx, verdict: Optional[ScriptVerdict], hook_name: str) -> None:
    """Map a verdict onto heaven's seams: block → the BEFORE_TOOL_CALL veto
    (ctx.data['block'] + block_message); inject text → the agent's script-inject buffer
    (flushed into the system prompt by the flusher registered alongside the hook)."""
    if verdict is None:
        return
    if verdict.blocks:
        ctx.data["block"] = True
        ctx.data["block_message"] = verdict.reason or f"blocked by hook {hook_name}"
    text = verdict.inject_text
    if text:
        agent = getattr(ctx, "agent", None)
        if agent is not None:
            if not hasattr(agent, "_script_hook_inject"):
                agent._script_hook_inject = []
            agent._script_hook_inject.append(text)


def make_script_hook(path: Path, decl: ScriptHookDecl, point_value: str):
    """The registered callable for one (script, event) pair."""
    def script_hook(ctx):
        try:
            verdict = run_script(path, build_payload(point_value, ctx), decl.timeout_s)
            apply_verdict(ctx, verdict, path.name)
        except Exception as e:      # the whole mechanism is fail-open
            logger.info("script hook %s (%s) failed (fail-open): %s", path.name, point_value, e)
    script_hook.__name__ = f"script_hook__{path.stem}__{point_value}"
    return script_hook


def make_inject_flusher():
    """A BEFORE_SYSTEM_PROMPT hook draining the agent's script-inject buffer into the prompt
    (registered ONCE per agent, by the first script hook that registers)."""
    def flush_script_inject(ctx):
        agent = getattr(ctx, "agent", None)
        buf = getattr(agent, "_script_hook_inject", None) if agent is not None else None
        if not buf:
            return
        base = ctx.data.get("system_prompt", getattr(ctx, "prompt", "") or "")
        ctx.data["system_prompt"] = (base + "\n\n<SCRIPT_HOOK_CONTEXT>\n"
                                     + "\n".join(buf) + "\n</SCRIPT_HOOK_CONTEXT>")
        buf.clear()
    return flush_script_inject


__all__ = ["ScriptVerdict", "ScriptHookDecl", "parse_decl", "build_payload", "run_script",
           "apply_verdict", "make_script_hook", "make_inject_flusher", "DEFAULT_TIMEOUT_S"]
