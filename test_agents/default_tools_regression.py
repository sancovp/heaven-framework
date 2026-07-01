from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heaven_base.baseheavenagent import BaseHeavenAgent
from heaven_base.tools.skill_tool import SkillTool
from heaven_base.tools.task_system_tool import TaskSystemTool
from heaven_base.tools.write_block_report_tool import WriteBlockReportTool


def test_resolve_tools_adds_default_tools_including_skill_tool():
    agent = object.__new__(BaseHeavenAgent)
    agent.config_tools = []

    resolved_tools = agent.resolve_tools()

    assert WriteBlockReportTool in resolved_tools
    assert TaskSystemTool in resolved_tools
    assert SkillTool in resolved_tools
