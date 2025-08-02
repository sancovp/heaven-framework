import asyncio
import os
from typing import Optional, Dict, Any, ClassVar, Type
from langchain.tools import Tool, BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.schema.runnable import RunnableConfig
from collections.abc import Callable

from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema

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

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"ERROR: timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)

        # # read output from the process, until the sentinel is found
        # try:
        #     async with asyncio.timeout(self._timeout):
        #         while True:
        #             await asyncio.sleep(self._output_delay)
        #             # if we read directly from stdout/stderr, it will wait forever for
        #             # EOF. use the StreamReader buffer directly instead.
        #             output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        #             if self._sentinel in output:
        #                 # strip the sentinel and break
        #                 output = output[: output.index(self._sentinel)]
        #                 break
        # except asyncio.TimeoutError:
        #     self._timed_out = True
        #     raise ToolError(
        #         f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
        #     ) from None

        # if output.endswith("\n"):
        #     output = output[:-1]

        # error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        # if error.endswith("\n"):
        #     error = error[:-1]

        # # clear the buffers so that the next output can be read correctly
        # self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        # self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        # return CLIResult(output=output, error=error)


# OLD
# class BashToolArgsSchema(ToolArgsSchema):
#     arguments: Dict[str, Dict[str, Any]] = {
#         'command': {
#             'name': 'command',
#             'type': 'str',
#             'description': 'The bash command to run. Required unless the tool is being restarted. For running files inside /core/, you may need to set PYTHONPATH=/home/GOD/core'
#         },
#         'restart': {
#             'name': 'restart',
#             'type': 'bool',
#             'description': 'Specifying true will restart this tool. Otherwise, leave this unspecified. If you receive a timeout error, BashTool must be restarted without any command before it can be used again.'
#         }
#     }

class BashToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'command': {
            'name': 'command',
            'type': 'str',
            'description': 'The bash command to run. Required unless restart is set to true. For running files inside /core/, you may need to set PYTHONPATH=/home/GOD/core',
            'required': False  # Not required when restarting
        },
        'restart': {
            'name': 'restart',
            'type': 'bool',
            'description': 'Specifying true will restart this tool. Otherwise, leave this unspecified or false. If you receive a timeout error, BashTool must be restarted without any command before it can be used again.',
            'required': False  # Defaults to false
        }
    }

# class BashTool(BaseHeavenTool):
#     name = "BashTool"
#     description = "Run commands in a bash shell"
#     args_schema = BashToolArgsSchema
#     is_async = True

#     def __init__(self, base_tool: BaseTool, args_schema: Type[ToolArgsSchema], is_async: bool = False):
#         super().__init__(base_tool=base_tool, args_schema=args_schema, is_async=is_async)
#         self._bash_session = _BashSession()  # Each instance gets its own session

#     @classmethod
#     def create(cls, adk: bool = False):
#         async def wrapped_func(command: str | None = None, restart: bool = False):
#             # Get the instance from the bound function
#             self = wrapped_func.__self__

#             if restart:
#                 if self._bash_session and self._bash_session._started:
#                     self._bash_session.stop()
#                 self._bash_session = _BashSession()
#                 await self._bash_session.start()
#                 return "tool has been restarted."

#             if not self._bash_session._started:
#                 await self._bash_session.start()

#             if command is not None:
#                 return await self._bash_session.run(command)

#             raise ToolError("ERROR: no command provided.")

#         cls.func = wrapped_func
#         instance = super().create()
#         wrapped_func.__self__ = instance
#         return instance

# With ADK
class BashTool(BaseHeavenTool):
    name = "BashTool"
    description = "Run commands in a bash shell"
    args_schema = BashToolArgsSchema
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
                return await session.run(command)
            raise ToolError("ERROR: no command provided.")
    
        cls.func = wrapped_func
        instance = super().create(adk=adk)
        wrapped_func.__self__ = instance
        return instance





      
      
# #### Has timeout problem
# class BashTool(BaseHeavenTool):
#     name = "BashTool"
#     description = "Run commands in a bash shell"
#     args_schema = BashToolArgsSchema
#     is_async = True

#     _bash_session = None  # Class-level singleton

#     @classmethod
#     def get_session(cls):
#         if cls._bash_session is None:
#             cls._bash_session = _BashSession()
#         return cls._bash_session

#     @classmethod
#     def create(cls):
#         # Get/create the persistent session
#         session = cls.get_session()

#         async def wrapped_func(command: str | None = None, restart: bool = False):
#             if restart:
#                 if cls._bash_session and cls._bash_session._started:
#                     cls._bash_session.stop()
#                 cls._bash_session = None
#                 cls._bash_session = cls.get_session()
#                 await cls._bash_session.start()
#                 return "tool has been restarted."

#             if not cls._bash_session._started:
#                 await cls._bash_session.start()

#             if command is not None:
#                 return await cls._bash_session.run(command)

#             raise ToolError("no command provided.")

#         cls.func = wrapped_func
#         return super().create()

# class BashTool(BaseHeavenTool):
#     name = "BashTool"
#     description = "Run commands in a bash shell"
#     args_schema = BashToolArgsSchema
#     is_async = True

#     # Dictionary to store sessions per container
#     _bash_sessions: ClassVar[Dict[str, _BashSession]] = {}

#     @classmethod
#     def get_session(cls, container_id: str = "default"):
#         if container_id not in cls._bash_sessions:
#             cls._bash_sessions[container_id] = _BashSession()
#         return cls._bash_sessions[container_id]

#     @classmethod
#     def create(cls):
#         async def wrapped_func(command: str | None = None, restart: bool = False, container_id: str = "default"):
#             session = cls.get_session(container_id)

#             if restart:
#                 if session and session._started:
#                     session.stop()
#                 cls._bash_sessions[container_id] = _BashSession()
#                 session = cls._bash_sessions[container_id]
#                 await session.start()
#                 return "tool has been restarted."

#             if not session._started:
#                 await session.start()

#             if command is not None:
#                 return await session.run(command)

#             raise ToolError("no command provided.")

#         cls.func = wrapped_func
#         return super().create()



  
# # Global session management
# _bash_session = None

# # The main function that our tool will wrap
# async def execute_bash_command(command: str | None = None, restart: bool = False) -> str:
#     """Main function that handles all bash execution logic"""
#     global _bash_session

#     try:
#                 # Add safety checks
#         if command is not None:
#             # Check for rm -rf
#             if "rm -rf" in command:
#                 raise ToolError("SAFETY ERROR: 'rm -rf' commands are forbidden to prevent accidental data loss. If you need to delete something large, ask the user to help you by doing it themselves.")

#             # Could also add other dangerous commands to check for
#             dangerous_commands = [
#                 "rm -rf",
#                 "rm -r",
#                 "rmdir",  # Maybe allow this one with specific checks
#                 "> /dev/",  # Prevent direct device writes
#                 "mkfs",    # Prevent filesystem formatting
#                 "dd",      # Prevent direct disk operations
#             ]

#             for dangerous_cmd in dangerous_commands:
#                 if dangerous_cmd in command:
#                     raise ToolError(f"SAFETY ERROR: '{dangerous_cmd}' is a protected command that could cause data loss. Ask the user to help you if you are tangled.")

#         if _bash_session is None:
#             _bash_session = _BashSession()
#             await _bash_session.start()

#         if restart:
#             if _bash_session:
#                 _bash_session.stop()
#             _bash_session = _BashSession()
#             await _bash_session.start()
#             return "tool has been restarted."

#         if command is not None:
#             result = await _bash_session.run(command)
#             # Convert CLIResult to string format for the tool
#             return f"Output: {result.output}\nError: {result.error}" if result.error else result.output

#         raise ToolError("no command provided.")
#     except Exception as e:
#         raise ToolError(f"Error in bash command: {e}")

# class BashTool(BaseHeavenTool):
#     """A tool that allows the agent to run bash commands."""
#     name = "BashTool"
#     description ="Run commands in a bash shell"
#     func = execute_bash_command
#     args_schema = BashToolArgsSchema
#     is_async = True