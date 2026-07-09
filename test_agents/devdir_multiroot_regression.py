"""Regression test for devdir resolution: an agent loads its configured/launch cwd's rules (always-on)
PLUS the rules of the SINGLE dir it is currently working in (its most recent read/bash). The active dir
SWAPS on move — leaving a dir drops its rules — mirroring how a CLAUDE.md is active only while you are in
that dir. Fixes devdir loading being keyed only to os.getcwd() (the launch dir, not where the agent works)."""
from pathlib import Path
from types import SimpleNamespace

from heaven_base.baseheavenagent import BaseHeavenAgent, HookRegistry, _extract_tool_path


def _agent(monkeypatch, configured_cwd=None, active=None):
    agent = object.__new__(BaseHeavenAgent)
    agent.hooks = HookRegistry()
    agent.config = SimpleNamespace(name="devdir-multiroot-test")
    agent._devdir_hook_keys = set()
    if configured_cwd is not None:
        agent._configured_cwd = str(configured_cwd)
    if active is not None:
        agent._active_work_dir = str(active)
    monkeypatch.setattr(BaseHeavenAgent, "_equipped_skill_summaries", lambda self: [])
    return agent


def _mk_repo(base: Path, marker: str, body: str = None) -> Path:
    (base / ".git").mkdir(parents=True)
    (base / ".claude" / "rules").mkdir(parents=True)
    (base / ".claude" / "rules" / f"{marker}.md").write_text(body if body is not None else f"RULE BODY FOR {marker}")
    return base


# ---- the core fix: BOTH the always-on configured cwd AND the current read/bash dir load ----

def test_loads_both_configured_cwd_and_active_dir(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "home_aios", "HOME")       # the agent's own AIOS (always-on)
    b = _mk_repo(tmp_path / "reviewed_repo", "REPO")    # the dir it is currently in
    resolved = _agent(monkeypatch, configured_cwd=a, active=b).resolve_devdirs("BASE")
    assert "RULE BODY FOR HOME" in resolved, "configured cwd's rules must load"
    assert "RULE BODY FOR REPO" in resolved, "current read/bash dir's rules must ALSO load (the fix)"
    assert resolved.startswith("BASE"), "base prompt preserved"


def test_overlapping_root_bodies_deduped_once(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "a", "A_ONLY")
    b = _mk_repo(tmp_path / "b", "B_ONLY")
    (a / ".claude" / "rules" / "shared.md").write_text("IDENTICAL SHARED BODY")
    (b / ".claude" / "rules" / "shared.md").write_text("IDENTICAL SHARED BODY")
    resolved = _agent(monkeypatch, configured_cwd=a, active=b).resolve_devdirs("BASE")
    assert resolved.count("IDENTICAL SHARED BODY") == 1
    assert "RULE BODY FOR A_ONLY" in resolved and "RULE BODY FOR B_ONLY" in resolved


def test_no_active_dir_falls_back_to_configured_cwd_only(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "solo", "SOLO")
    resolved = _agent(monkeypatch, configured_cwd=a, active=None).resolve_devdirs("BASE")
    assert "RULE BODY FOR SOLO" in resolved


# ---- the walk boundary: phase-1 (skip leading gap) + phase-2 (gap above entry stops); NOT .git-bounded ----

def test_deep_agent_reaches_root_devdir_across_leading_gap(tmp_path, monkeypatch):
    # repo has .claude ONLY at the root; agent works 2 levels deep with no .claude in between.
    # Phase 1 must climb PAST the non-devdir dirs to the repo-root .claude (the old .git-walk did this;
    # a strict "stop at first parent without a devdir" would wrongly return nothing).
    repo = _mk_repo(tmp_path / "repo", "ROOT")
    feature = repo / "pkg" / "feature"; feature.mkdir(parents=True)
    resolved = _agent(monkeypatch, configured_cwd=tmp_path / "elsewhere", active=feature).resolve_devdirs("BASE")
    assert "RULE BODY FOR ROOT" in resolved, "deep agent in a root-only-.claude repo still gets the root rules"


def test_nested_contiguous_devdirs_all_load(tmp_path, monkeypatch):
    root = _mk_repo(tmp_path / "r", "R_TOP")
    mid = _mk_repo(root / "mid", "R_MID")
    leaf = _mk_repo(mid / "leaf", "R_LEAF")
    resolved = _agent(monkeypatch, configured_cwd=tmp_path / "home", active=leaf).resolve_devdirs("BASE")
    for m in ("R_TOP", "R_MID", "R_LEAF"):
        assert f"RULE BODY FOR {m}" in resolved, f"{m} in the contiguous chain must load"


def test_gap_above_entry_stops_the_chain(tmp_path, monkeypatch):
    # top/.claude(TOP) -> top/gap (NO devdir) -> top/gap/leaf/.claude(LEAF); agent at leaf.
    # entry = leaf; its parent 'gap' has no devdir -> STOP. TOP is above the gap and must NOT load.
    _mk_repo(tmp_path / "top", "TOP")
    gap = tmp_path / "top" / "gap"; gap.mkdir()
    leaf = _mk_repo(gap / "leaf", "LEAF")
    resolved = _agent(monkeypatch, configured_cwd=tmp_path / "home", active=leaf).resolve_devdirs("BASE")
    assert "RULE BODY FOR LEAF" in resolved
    assert "RULE BODY FOR TOP" not in resolved, "a gap ABOVE the entry ends the chain -> TOP not loaded"


# ---- the tracker: READ + BASH set the current dir, and it SWAPS on move (leave → gone) ----

def _bare_agent():
    agent = object.__new__(BaseHeavenAgent)
    agent._active_work_dir = None
    return agent


def test_tracker_sets_current_dir_from_bash_command(tmp_path):
    d = tmp_path / "somerepo" / "src"
    d.mkdir(parents=True)
    f = d / "x.py"; f.write_text("x")
    agent = _bare_agent()
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"cat {f}"}))
    assert agent._active_work_dir == str(d)


def test_tracker_sets_current_dir_from_read_file_path(tmp_path):
    d = tmp_path / "repo2"; d.mkdir()
    f = d / "y.py"; f.write_text("y")
    agent = _bare_agent()
    agent._track_active_work_dir(SimpleNamespace(tool_name="ReadTool", tool_args={"file_path": str(f)}))
    assert agent._active_work_dir == str(d)


def test_tracker_swaps_on_move_so_leaving_a_dir_drops_it(tmp_path, monkeypatch):
    # The heart of the behavior: read in dir A, then move to dir B -> only B is active (A's rules gone).
    a = _mk_repo(tmp_path / "dirA", "A_MARKER")
    b = _mk_repo(tmp_path / "dirB", "B_MARKER")
    (a / "f.txt").write_text("a"); (b / "f.txt").write_text("b")  # root-level files to cat
    (tmp_path / "home").mkdir()
    agent = _bare_agent()
    agent.hooks = HookRegistry()
    agent.config = SimpleNamespace(name="swap-test")
    agent._devdir_hook_keys = set()
    agent._configured_cwd = str(tmp_path / "home")
    monkeypatch.setattr(BaseHeavenAgent, "_equipped_skill_summaries", lambda self: [])

    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"cat {a}/f.txt"}))
    assert agent._active_work_dir == str(a)
    r1 = agent.resolve_devdirs("BASE")
    assert "RULE BODY FOR A_MARKER" in r1 and "RULE BODY FOR B_MARKER" not in r1

    # move to B — A must go away
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"cat {b}/f.txt"}))
    assert agent._active_work_dir == str(b)
    r2 = agent.resolve_devdirs("BASE")
    assert "RULE BODY FOR B_MARKER" in r2, "the new dir's rules load"
    assert "RULE BODY FOR A_MARKER" not in r2, "LEFT dir A -> its rules are GONE (swap-on-leave)"


def test_tracker_never_raises_on_junk(tmp_path):
    agent = _bare_agent()
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": "echo hi"}))
    agent._track_active_work_dir(SimpleNamespace(tool_name="ReadTool", tool_args={"file_path": "/no/such/dir/x"}))
    agent._track_active_work_dir(SimpleNamespace(tool_name="", tool_args=None))
    assert agent._active_work_dir is None


def test_extract_tool_path_helper(tmp_path):
    f = tmp_path / "z.txt"; f.write_text("z")
    assert _extract_tool_path("ReadTool", {"file_path": str(f)}) == str(f)
    assert _extract_tool_path("BashTool", {"command": f"git -C {tmp_path} status"}) == str(tmp_path)
    assert _extract_tool_path("BashTool", {"command": "ls"}) is None
    assert _extract_tool_path("SomeTool", {"note": "no path here"}) is None
