from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heaven_base.baseheavenagent import BaseHeavenAgent, HookPoint, HookRegistry


def _agent(monkeypatch):
    agent = object.__new__(BaseHeavenAgent)
    agent.hooks = HookRegistry()
    agent.config = SimpleNamespace(name="devdir-test")
    agent._devdir_hook_keys = set()
    monkeypatch.setattr(BaseHeavenAgent, "_equipped_skill_summaries", lambda self: [])
    return agent


def test_resolve_devdirs_gathers_claude_then_heaven_and_dedupes(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    leaf = repo / "pkg" / "feature"
    leaf.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text("root claude instructions")

    (repo / ".claude" / "rules").mkdir(parents=True)
    (repo / ".claude" / "rules" / "shared.md").write_text("shared rule body")
    (repo / ".claude" / "skills" / "alpha").mkdir(parents=True)
    (repo / ".claude" / "skills" / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill\n---\n\nAlpha body"
    )

    (repo / ".heaven" / "rules").mkdir(parents=True)
    (repo / ".heaven" / "rules" / "shared.md").write_text("shared rule body")
    (repo / ".heaven" / "rules" / "heaven.md").write_text("heaven-only rule")
    (repo / ".heaven" / "skills" / "beta").mkdir(parents=True)
    (repo / ".heaven" / "skills" / "beta" / "SKILL.md").write_text(
        "---\nname: beta\ndescription: Beta skill\n---\n\nBeta body"
    )

    monkeypatch.chdir(leaf)
    resolved = _agent(monkeypatch).resolve_devdirs("BASE")

    assert "root claude instructions" in resolved
    assert resolved.count("shared rule body") == 1
    assert "heaven-only rule" in resolved
    assert "**alpha**" in resolved
    assert "**beta**" in resolved
    assert resolved.index("root claude instructions") < resolved.index("heaven-only rule")


def test_resolve_devdirs_registers_claude_and_heaven_hooks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".claude" / "hooks").mkdir(parents=True)
    (repo / ".heaven" / "hooks").mkdir(parents=True)
    hook_source = (
        'POINT = "before_system_prompt"\n'
        "def hook(ctx):\n"
        "    ctx.data['system_prompt'] = ctx.data.get('system_prompt', ctx.prompt) + '\\nHOOKED'\n"
    )
    (repo / ".claude" / "hooks" / "prompt_hook.py").write_text(hook_source)
    (repo / ".heaven" / "hooks" / "same_prompt_hook.py").write_text(hook_source)

    monkeypatch.chdir(repo)
    agent = _agent(monkeypatch)
    resolved = agent.resolve_devdirs("BASE")
    ctx = agent._fire_hook(HookPoint.BEFORE_SYSTEM_PROMPT, prompt=resolved)

    assert ctx.data["system_prompt"].endswith("HOOKED")
    assert ctx.data["system_prompt"].count("HOOKED") == 1
