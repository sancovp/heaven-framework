"""Tests for heaven_base.devdir — THE ONE RESOLVER (goal devdir-parity, stage 1).

The stage-1 live proof ran on real dirs (~/.heaven + the monorepo); these tests pin the
CONTRACT so it cannot silently drift: provenance stamping, the ENTER+CLIMB walk with its gap
policy, path+content dedup, the default grammar for a never-seen slot, and frontmatter.
"""
import os
from pathlib import Path

import pytest

from heaven_base.devdir import (
    DevdirFile, resolve, devdir_levels, dir_has_devdir, parse_frontmatter,
)


@pytest.fixture()
def world(tmp_path):
    """repo/ (devdir) > mid/ (NO devdir — the gap) > deep/ (devdir). Plus a sibling place/."""
    repo = tmp_path / "repo"
    (repo / ".heaven" / "rules").mkdir(parents=True)
    (repo / ".heaven" / "rules" / "r1.md").write_text("repo rule one")
    (repo / ".heaven" / "hooks").mkdir(parents=True)
    (repo / ".heaven" / "hooks" / "h1.py").write_text("POINT='before_run'\ndef hook(ctx): pass\n")
    (repo / ".heaven" / "hooks" / "h2.py.inactive").write_text("POINT='before_run'\ndef hook(ctx): pass\n")
    (repo / "CLAUDE.md").write_text("repo claude md")
    mid = repo / "mid"
    mid.mkdir()
    deep = mid / "deep"
    (deep / ".claude" / "rules").mkdir(parents=True)
    (deep / ".claude" / "rules" / "d1.md").write_text("deep rule")
    place = tmp_path / "place"
    (place / ".heaven" / "widgets").mkdir(parents=True)
    (place / ".heaven" / "widgets" / "w.json").write_text('{"a": 1}')
    (place / ".heaven" / "widgets" / "sub").mkdir()
    (place / ".heaven" / "widgets" / "sub" / "inner.txt").write_text("inner")
    (place / ".heaven" / "widgets" / "_private.json").write_text("{}")
    return tmp_path


def test_walk_gap_policy(world):
    # ENTER from deep finds deep; CLIMB stops at mid (no devdir) — the gap ends the chain.
    levels = devdir_levels(world / "repo" / "mid" / "deep")
    assert levels == [(world / "repo" / "mid" / "deep").resolve()]
    # ENTER skips a leading gap: starting IN mid climbs to repo.
    levels = devdir_levels(world / "repo" / "mid")
    assert levels == [(world / "repo").resolve()]
    assert dir_has_devdir(world / "repo") and not dir_has_devdir(world / "repo" / "mid")


def test_resolve_provenance_and_active_root(world):
    launch = world / "repo"
    files = resolve(launch, None, "rules")
    launch_files = [f for f in files if f.source.root == "launch"]
    assert {os.path.basename(f.path) for f in launch_files} == {"CLAUDE.md", "r1.md"}
    for f in launch_files:
        assert f.source.level == str(launch.resolve())
        assert f.source.devdir in (".claude", ".heaven")
    # standing in another place ADDS its files (the AMBIENT shape), stamped root="active"
    files2 = resolve(launch, world / "repo" / "mid" / "deep", "rules")
    active = [f for f in files2 if f.source.root == "active"]
    assert {os.path.basename(f.path) for f in active} == {"d1.md"}


def test_hooks_presence_is_active(world):
    files = resolve(world / "repo", None, "hooks")
    names = {os.path.basename(f.path) for f in files if f.source.root == "launch"}
    assert names == {"h1.py"}          # .py.inactive structurally excluded


def test_never_seen_slot_default_grammar(world):
    files = resolve(world / "place", None, "widgets")
    mine = [f for f in files if f.source.root == "launch"]
    names = {os.path.basename(f.path) for f in mine}
    assert names == {"w.json", "inner.txt"}   # one + two levels deep; _private excluded
    assert all(isinstance(f, DevdirFile) for f in mine)


def test_content_dedup(world):
    dup = world / "repo" / ".heaven" / "rules" / "r1_copy.md"
    dup.write_text("repo rule one")     # identical body
    files = resolve(world / "repo", None, "rules")
    bodies = [f.content for f in files if f.source.root == "launch"]
    assert bodies.count("repo rule one") == 1


def test_frontmatter():
    fm = parse_frontmatter("---\nname: x\ndescription: y z\n---\nbody")
    assert fm == {"name": "x", "description": "y z"}
    fm2 = parse_frontmatter("# title\nname: loose\n")
    assert fm2["name"] == "loose"
