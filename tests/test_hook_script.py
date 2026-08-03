"""Tests for heaven_base.hook_script — the SCRIPT hook contract (goal devdir-parity stage 3).

The live proof ran a real script hook (imports nothing) through resolve_devdirs' registration
path — veto + pass-through + inject all held. These tests pin the CONTRACT: the text-header
declaration, the stdin/stdout envelope, exit-2 = block, fail-open, and the flusher.
"""
import json
import stat
from pathlib import Path

import pytest

from heaven_base.hook_script import (
    ScriptVerdict, parse_decl, run_script, apply_verdict, make_inject_flusher,
)


def test_parse_decl():
    d = parse_decl("# hook-events: before_tool_call, after_run\n# hook-timeout: 3\nprint('x')\n")
    assert d.events == ["before_tool_call", "after_run"]
    assert d.timeout_s == 3
    assert parse_decl("print('no header')\n") is None          # no events = a MODULE hook
    assert parse_decl("# HOOK-EVENTS: before_run\n").events == ["before_run"]   # case-insensitive


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_run_script_json_verdict(tmp_path):
    p = _write(tmp_path, "h.py",
               "import json,sys\n"
               "d=json.load(sys.stdin)\n"
               "print(json.dumps({'decision':'block','reason':'no '+d['tool_name']}))\n")
    v = run_script(p, {"tool_name": "X"}, timeout_s=5)
    assert v.blocks and v.reason == "no X"


def test_run_script_exit_2_blocks_with_stderr(tmp_path):
    p = _write(tmp_path, "h.py", "import sys\nsys.stderr.write('nope')\nsys.exit(2)\n")
    v = run_script(p, {}, timeout_s=5)
    assert v.blocks and v.reason == "nope"


def test_run_script_fail_open(tmp_path):
    crash = _write(tmp_path, "crash.py", "raise RuntimeError('boom')\n")
    assert run_script(crash, {}, timeout_s=5) is None
    garbage = _write(tmp_path, "garbage.py", "print('not json at all')\n")
    assert run_script(garbage, {}, timeout_s=5) is None
    silent = _write(tmp_path, "silent.py", "pass\n")
    assert run_script(silent, {}, timeout_s=5) is None


def test_non_python_executable_runs_itself(tmp_path):
    p = tmp_path / "h.sh"
    p.write_text("#!/bin/sh\necho '{\"systemMessage\": \"from shell\"}'\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    v = run_script(p, {}, timeout_s=5)
    assert v.inject_text == "from shell"


class _Ctx:
    def __init__(self, agent=None, prompt=""):
        self.agent = agent
        self.prompt = prompt
        self.data = {}


class _Agent:
    pass


def test_apply_verdict_and_flusher():
    agent = _Agent()
    ctx = _Ctx(agent=agent)
    apply_verdict(ctx, ScriptVerdict(decision="block", reason="r", systemMessage="ctx-note"),
                  "h.py")
    assert ctx.data["block"] is True and ctx.data["block_message"] == "r"
    assert agent._script_hook_inject == ["ctx-note"]
    flush = make_inject_flusher()
    c2 = _Ctx(agent=agent, prompt="BASE")
    flush(c2)
    assert "<SCRIPT_HOOK_CONTEXT>" in c2.data["system_prompt"]
    assert "ctx-note" in c2.data["system_prompt"]
    assert agent._script_hook_inject == []          # drained — turn-scoped
    c3 = _Ctx(agent=agent, prompt="BASE")
    flush(c3)
    assert "system_prompt" not in c3.data           # nothing pending = untouched
