"""Regression test for multi-root devdir resolution: an agent loads BOTH its configured/launch cwd's
rules AND the rules of the dirs it reads/bashes into (tracked via AFTER_TOOL_CALL). This is the fix for
devdir loading being keyed only to os.getcwd() (which is the launch dir, NOT where the agent works)."""
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
        agent._active_work_dirs = [str(d) for d in active]
    monkeypatch.setattr(BaseHeavenAgent, "_equipped_skill_summaries", lambda self: [])
    return agent


def _mk_repo(base: Path, marker: str, body: str = None) -> Path:
    (base / ".git").mkdir(parents=True)
    (base / ".claude" / "rules").mkdir(parents=True)
    (base / ".claude" / "rules" / f"{marker}.md").write_text(body if body is not None else f"RULE BODY FOR {marker}")
    return base


# ---- the core fix: BOTH the configured cwd AND the read/bash-tracked dirs load ----

def test_loads_both_configured_cwd_and_active_dirs(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "home_aios", "HOME")       # the agent's own AIOS (its launch/working dir)
    b = _mk_repo(tmp_path / "reviewed_repo", "REPO")    # a repo it reads/bashes into
    resolved = _agent(monkeypatch, configured_cwd=a, active=[b]).resolve_devdirs("BASE")
    assert "RULE BODY FOR HOME" in resolved, "configured cwd's rules must load"
    assert "RULE BODY FOR REPO" in resolved, "read/bash-tracked dir's rules must ALSO load (the fix)"
    assert resolved.startswith("BASE"), "base prompt preserved"


def test_overlapping_root_bodies_deduped_once(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "a", "A_ONLY")
    b = _mk_repo(tmp_path / "b", "B_ONLY")
    # both also carry an identical shared rule body → content-hash dedup must include it once
    (a / ".claude" / "rules" / "shared.md").write_text("IDENTICAL SHARED BODY")
    (b / ".claude" / "rules" / "shared.md").write_text("IDENTICAL SHARED BODY")
    resolved = _agent(monkeypatch, configured_cwd=a, active=[b]).resolve_devdirs("BASE")
    assert resolved.count("IDENTICAL SHARED BODY") == 1
    assert "RULE BODY FOR A_ONLY" in resolved and "RULE BODY FOR B_ONLY" in resolved


def test_no_active_dirs_falls_back_to_configured_cwd_only(tmp_path, monkeypatch):
    a = _mk_repo(tmp_path / "solo", "SOLO")
    resolved = _agent(monkeypatch, configured_cwd=a, active=[]).resolve_devdirs("BASE")
    assert "RULE BODY FOR SOLO" in resolved


# ---- the tracker: READ + BASH tool use updates _active_work_dirs ----

def _bare_agent():
    agent = object.__new__(BaseHeavenAgent)
    agent._active_work_dirs = []
    return agent


def test_tracker_records_dir_from_bash_command(tmp_path):
    d = tmp_path / "somerepo" / "src"
    d.mkdir(parents=True)
    f = d / "x.py"
    f.write_text("x")
    agent = _bare_agent()
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"cat {f}"}))
    assert str(d) in agent._active_work_dirs


def test_tracker_records_dir_from_read_file_path(tmp_path):
    d = tmp_path / "repo2"
    d.mkdir()
    f = d / "y.py"
    f.write_text("y")
    agent = _bare_agent()
    agent._track_active_work_dir(SimpleNamespace(tool_name="ReadTool", tool_args={"file_path": str(f)}))
    assert str(d) in agent._active_work_dirs


def test_tracker_cap_and_newest_refines(tmp_path):
    agent = _bare_agent()
    dirs = []
    for i in range(10):
        di = tmp_path / f"d{i}"
        di.mkdir()
        dirs.append(di)
        agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"ls {di}"}))
    assert len(agent._active_work_dirs) == BaseHeavenAgent._MAX_ACTIVE_WORK_DIRS  # 8
    assert str(dirs[0]) not in agent._active_work_dirs, "oldest dropped past the cap"
    assert agent._active_work_dirs[-1] == str(dirs[9]), "newest is last"
    # re-touching a still-tracked dir moves it to the end (newest-refines)
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": f"ls {dirs[5]}"}))
    assert agent._active_work_dirs[-1] == str(dirs[5])


def test_tracker_never_raises_on_junk(tmp_path):
    agent = _bare_agent()
    # no path in args, nonexistent path, empty — none should raise or record
    agent._track_active_work_dir(SimpleNamespace(tool_name="BashTool", tool_args={"command": "echo hi"}))
    agent._track_active_work_dir(SimpleNamespace(tool_name="ReadTool", tool_args={"file_path": "/no/such/dir/x"}))
    agent._track_active_work_dir(SimpleNamespace(tool_name="", tool_args=None))
    assert agent._active_work_dirs == []


def test_extract_tool_path_helper(tmp_path):
    f = tmp_path / "z.txt"
    f.write_text("z")
    assert _extract_tool_path("ReadTool", {"file_path": str(f)}) == str(f)
    assert _extract_tool_path("BashTool", {"command": f"git -C {tmp_path} status"}) == str(tmp_path)
    assert _extract_tool_path("BashTool", {"command": "ls"}) is None
    assert _extract_tool_path("SomeTool", {"note": "no path here"}) is None
