"""ON_BLOCK_REPORT fires when an agent files a block report.

The hook is the first-class seam for triage automations / the SI backlog (OM DESIGN §26.3):
a block report = a voluntary agent halt, and subscribers must be able to react to it without
polling /tmp/block_report.json. Before this, HookPoint.ON_BLOCK_REPORT was DECLARED but never fired.
"""
import json
import os

from heaven_base.baseheavenagent import BaseHeavenAgent, HookRegistry, HookPoint


def test_on_block_report_fires_with_data_and_md(tmp_path, monkeypatch):
    # create_block_report reads the fixed /tmp path written by WriteBlockReportTool
    report = {
        "completed_tasks": ["did A", "did B"],
        "current_task": "stuck on C",
        "explanation": "the widget API returned 500",
        "blocked_reason": "human_required: need a credential",
        "timestamp": "2026-07-06T00:00:00",
    }
    with open("/tmp/block_report.json", "w") as f:
        json.dump(report, f)

    captured = {}

    def handler(ctx):
        captured["fired"] = True
        captured["md"] = ctx.data.get("block_report_md")
        captured["report"] = ctx.data.get("block_report")
        captured["agent_name"] = ctx.agent.name

    reg = HookRegistry()
    reg.register(HookPoint.ON_BLOCK_REPORT, handler)

    # exercise the REAL create_block_report + _fire_hook without a full agent construction
    agent = BaseHeavenAgent.__new__(BaseHeavenAgent)
    agent.hooks = reg
    agent.name = "tester"
    agent.current_task = "stuck on C"
    agent.goal = "build the thing"

    md = agent.create_block_report()

    # the hook fired, with both the structured report and the rendered markdown
    assert captured.get("fired") is True, "ON_BLOCK_REPORT did not fire"
    assert captured["report"]["blocked_reason"] == "human_required: need a credential"
    assert captured["report"]["completed_tasks"] == ["did A", "did B"]
    assert "BLOCKED REPORT" in captured["md"]
    assert captured["agent_name"] == "tester"
    # method still returns the markdown and cleans up the world file
    assert "BLOCKED REPORT" in md
    assert not os.path.exists("/tmp/block_report.json"), "block report file should be cleaned up"


def test_no_block_report_file_no_fire():
    """No /tmp/block_report.json → create_block_report returns None and does not fire the hook."""
    if os.path.exists("/tmp/block_report.json"):
        os.remove("/tmp/block_report.json")

    fired = {"count": 0}
    reg = HookRegistry()
    reg.register(HookPoint.ON_BLOCK_REPORT, lambda ctx: fired.__setitem__("count", fired["count"] + 1))

    agent = BaseHeavenAgent.__new__(BaseHeavenAgent)
    agent.hooks = reg
    agent.name = "tester"
    agent.current_task = None
    agent.goal = None

    assert agent.create_block_report() is None
    assert fired["count"] == 0
