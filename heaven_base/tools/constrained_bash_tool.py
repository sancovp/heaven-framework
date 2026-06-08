"""
ConstrainedBashTool — BashTool with destructive command guards.

Same as BashTool but blocks dangerous commands like rm -rf, dd, mkfs, etc.
Use this for agents in swarms or any context where you want safety rails.
"""

import asyncio
import os
import re
from typing import Optional, Dict, Any, ClassVar, Type
from langchain.tools import Tool, BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.schema.runnable import RunnableConfig
from collections.abc import Callable

from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema


# ── Dangerous command patterns ──
# These are checked as substrings or regex against the command string.
BLOCKED_COMMANDS = [
    "rm -rf",
    "rm -r ",
    "rm -fr",
    "> /dev/",
    "mkfs",
    "dd if=",
    "dd of=",
    ":(){ :|:& };:",    # fork bomb
    "chmod -R 777",
    "chown -R",
    "kill -9 1",        # kill init
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "mv /* ",
    "mv / ",
]

# Regex patterns for more nuanced checks
BLOCKED_PATTERNS = [
    r"rm\s+(-\w*[rf]\w*\s+)?/\s",          # rm anything starting with /
    r">\s*/etc/",                             # overwrite system files
    r"pip\s+install\s+--force",              # force pip installs
    r"chmod\s+.*\s+/",                       # chmod on root paths
    r"(curl|wget)\s+.*\|\s*(sh|bash)",       # pipe downloads to shell
]


def _check_command_safety(command: str) -> None:
    """Raise ToolError if command matches any blocked pattern."""
    cmd_lower = command.lower().strip()

    for blocked in BLOCKED_COMMANDS:
        if blocked.lower() in cmd_lower:
            raise ToolError(
                f"🚫 BLOCKED: '{blocked}' is a destructive command. "
                f"ConstrainedBashTool does not allow this. "
                f"If you need to do this, ask the user directly."
            )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            raise ToolError(
                f"🚫 BLOCKED: Command matches destructive pattern '{pattern}'. "
                f"ConstrainedBashTool does not allow this."
            )


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("ERROR: Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("ERROR: Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"ERROR: timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output line-by-line using async readline() until sentinel is found
        try:
            output_lines = []
            async with asyncio.timeout(self._timeout):
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode()
                    if self._sentinel in decoded:
                        before_sentinel = decoded[:decoded.index(self._sentinel)]
                        if before_sentinel:
                            output_lines.append(before_sentinel)
                        break
                    output_lines.append(decoded)
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"ERROR: timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        output = "".join(output_lines)
        if output.endswith("\n"):
            output = output[:-1]

        error = ""
        if self._process.stderr._buffer:  # pyright: ignore[reportAttributeAccessIssue]
            error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
            self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
            if error.endswith("\n"):
                error = error[:-1]

        return CLIResult(output=output, error=error)


class ConstrainedBashToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'command': {
            'name': 'command',
            'type': 'str',
            'description': 'The bash command to run. Destructive commands (rm -rf, dd, mkfs, etc.) are blocked.',
            'required': False
        },
        'restart': {
            'name': 'restart',
            'type': 'bool',
            'description': 'Specifying true will restart this tool. If you receive a timeout error, the tool must be restarted before it can be used again.',
            'required': False
        }
    }


class ConstrainedBashTool(BaseHeavenTool):
    name = "ConstrainedBashTool"
    description = (
        "Run commands in a bash shell with safety guards. "
        "Destructive commands (rm -rf, dd, mkfs, chmod -R 777, etc.) are blocked. "
        "Use tmux send-keys/capture-pane to communicate with teammates."
    )
    args_schema = ConstrainedBashToolArgsSchema
    is_async = True

    def __init__(self, base_tool: BaseTool, args_schema: Type[ToolArgsSchema], is_async: bool = False):
        super().__init__(base_tool=base_tool, args_schema=args_schema, is_async=is_async)
        self._bash_session = _BashSession()

    @classmethod
    def create(cls, adk: bool = False):
        session = _BashSession()

        async def wrapped_func(command: Optional[str] = None,
                               restart: Optional[bool] = None):
            nonlocal session
            if restart:
                if session._started:
                    session.stop()
                session = _BashSession()
                await session.start()
                return "tool has been restarted."
            if not session._started:
                await session.start()
            if command is not None:
                # ── Safety gate ──
                _check_command_safety(command)
                return await session.run(command)
            raise ToolError("ERROR: no command provided.")

        cls.func = wrapped_func
        instance = super().create(adk=adk)
        wrapped_func.__self__ = instance
        return instance