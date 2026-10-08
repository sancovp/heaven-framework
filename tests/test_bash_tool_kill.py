"""Tests that a bash session can be killed with the command it runs: the shell leads its own process group
(`setsid`), so stopping the await that waits on it leaves the command running until the group is killed."""
import asyncio
import subprocess
import time

from heaven_base.tools.bash_tool import BashTool, kill_all_sessions


def _alive(marker: str) -> bool:
    out = subprocess.run(["ps", "-axo", "stat=,command="], capture_output=True, text=True).stdout
    return any(marker in line and not line.lstrip().startswith("Z") for line in out.splitlines())


def _gone(marker: str, within: float = 3.0) -> bool:
    """Whether the process is gone within `within` seconds (a killed process takes a moment to be reaped)."""
    end = time.time() + within
    while time.time() < end:
        if not _alive(marker):
            return True
        time.sleep(0.1)
    return False


def test_a_stopped_command_runs_on_until_its_session_is_killed():
    tool = BashTool.create()

    async def go():
        task = asyncio.create_task(tool.base_tool.ainvoke({"command": "sleep 51.731"}))
        await asyncio.sleep(1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert _alive("sleep 51.731")                  # cancelling the await does not stop the command
        assert tool.kill_session() is True
        assert _gone("sleep 51.731")
        assert tool.kill_session() is False            # nothing left to kill
    asyncio.run(go())


def test_every_live_session_can_be_killed_at_exit():
    tools = [BashTool.create(), BashTool.create()]

    async def go():
        tasks = [asyncio.create_task(t.base_tool.ainvoke({"command": f"sleep 52.73{i}"}))
                 for i, t in enumerate(tools)]
        await asyncio.sleep(1.0)
        assert kill_all_sessions() >= 2
        await asyncio.gather(*tasks, return_exceptions=True)
        assert _gone("sleep 52.730") and _gone("sleep 52.731")
    asyncio.run(go())
