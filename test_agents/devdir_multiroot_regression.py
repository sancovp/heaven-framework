"""Regression test for devdir resolution: an agent loads its configured/launch cwd's rules (always-on)
PLUS the rules of the SINGLE dir it is currently working in (its most recent read/bash). The active dir
SWAPS on move — leaving a dir drops its rules — mirroring how a CLAUDE.md is active only while you are in
that dir. Fixes devdir loading being keyed only to os.getcwd() (the launch dir, not where the agent works)."""
from pathlib import Path
from types import SimpleNamespace

from heaven_base.baseheavenagent import (
    BaseHeavenAgent, HookRegistry, _extract_tool_path, _scan_persona_directive,
)


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


# ---- persona declarations: skillmanager_persona= (FORCE, sticky, composes) / absolute_ (ignores parents) ----

def test_scan_persona_directive_force():
    assert _scan_persona_directive("blah\nskillmanager_persona=Gnosys\nmore") == ("Gnosys", False)


def test_scan_persona_directive_absolute_wins_over_the_force_substring_inside_it():
    # absolute_skillmanager_persona= literally CONTAINS skillmanager_persona= — absolute must still win,
    # and the plain FORCE regex must NOT match the absolute form (fixed-width negative lookbehind).
    assert _scan_persona_directive("absolute_skillmanager_persona=Isolated") == ("Isolated", True)


def test_scan_persona_directive_none():
    assert _scan_persona_directive("nothing to see") is None
    assert _scan_persona_directive("") is None
    assert _scan_persona_directive(None) is None


def test_forced_persona_is_STICKY_and_not_lost_when_the_agent_moves(tmp_path, monkeypatch):
    # dir A's CLAUDE.md declares a persona; after resolving there, moving to dir B (no declaration) KEEPS
    # the forced persona — the defining property: a forced persona does NOT get lost when the agent moves.
    a = _mk_repo(tmp_path / "declares", "A_RULE")
    (a / "CLAUDE.md").write_text("skillmanager_persona=Gnosys\n")
    b = _mk_repo(tmp_path / "plain", "B_RULE")
    agent = _agent(monkeypatch, configured_cwd=tmp_path / "home", active=a)
    agent.resolve_devdirs("BASE")
    assert agent._forced_persona == "Gnosys"
    assert agent._forced_persona_absolute is False
    # move to B — no declaration there — the persona must PERSIST
    agent._active_work_dir = str(b)
    agent.resolve_devdirs("BASE")
    assert agent._forced_persona == "Gnosys", "forced persona must NOT be lost when the agent moves"


def test_absolute_persona_IGNORES_PARENTS_suppressing_ambient_devdir_rules(tmp_path, monkeypatch):
    # leaf declares an ABSOLUTE persona; the parent chain's rules (and the leaf's own ambient rules) must
    # be SUPPRESSED — absolute = "only this persona, ignore all ambient/parent devdir context".
    root = _mk_repo(tmp_path / "r", "ROOT_RULE")
    leaf = _mk_repo(root / "leaf", "LEAF_RULE")
    (leaf / "CLAUDE.md").write_text("absolute_skillmanager_persona=Isolated\n")
    agent = _agent(monkeypatch, configured_cwd=tmp_path / "home", active=leaf)
    resolved = agent.resolve_devdirs("BASE")
    assert agent._forced_persona == "Isolated" and agent._forced_persona_absolute is True
    assert "RULE BODY FOR ROOT_RULE" not in resolved, "absolute persona ignores PARENT rules"
    assert "RULE BODY FOR LEAF_RULE" not in resolved, "absolute persona suppresses ALL ambient devdir rules"
    assert resolved.startswith("BASE"), "base prompt preserved"


def test_force_persona_COMPOSES_with_parent_rules(tmp_path, monkeypatch):
    # a plain FORCE persona is sticky but STILL composes with the ambient parent devdir rules.
    root = _mk_repo(tmp_path / "r2", "ROOT2_RULE")
    leaf = _mk_repo(root / "leaf2", "LEAF2_RULE")
    (leaf / "CLAUDE.md").write_text("skillmanager_persona=Composed\n")
    agent = _agent(monkeypatch, configured_cwd=tmp_path / "home", active=leaf)
    resolved = agent.resolve_devdirs("BASE")
    assert agent._forced_persona == "Composed" and agent._forced_persona_absolute is False
    assert "RULE BODY FOR ROOT2_RULE" in resolved, "force persona still loads parent rules"
    assert "RULE BODY FOR LEAF2_RULE" in resolved, "force persona still loads the current dir's rules"
