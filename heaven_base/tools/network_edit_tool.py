# network edit tool


import subprocess
import base64
from typing import Union
from langchain_core.tools import BaseTool, Tool
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args, Dict, Any, ClassVar, Type, Optional, Union
from collections.abc import Callable
import ast
import shlex
import difflib

from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema, ToolFailure

from .run import maybe_truncate, run


tool_log_path = "/tmp/tool_debug.log"

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4

guide_manual = """
• “Always capture the full block you intend to replace—including every leading/trailing newline, space, indentation level and delimiter—in your old_str so the tool can find an exact verbatim match.”

• “Before calling str_replace, run a view on the exact line range and copy-paste that output directly into your old_str; this eliminates manual typos or missing characters.”

• “Remember that old_str is matched 100% literally: any missing or extra space, newline, quote or backslash will cause a mismatch error.”

• “When your target spans multiple lines, wrap old_str in a multi-line JSON string (triple-quoted in your mind) so you’re replacing the entire contiguous block, not just fragments.”

• “Escape any double quotes inside your JSON string as \" and backslashes as \\ so that your JSON payload remains valid and matches the file content exactly.”

• “If you see a mismatch error, inspect the reported line/column and adjust your old_str—pay special attention to whitespace and escape sequences that may not be obvious at a glance.”

• “Use the same quoting style in new_str as the file uses (e.g. single quotes inside double-quoted strings) to avoid introducing syntax errors after replacement.”

• “For very large templates, draft and verify your old_str/new_str in an external editor, then paste them wholesale into the tool call to minimize hand-escaped mistakes.”

• “Include the opening and closing delimiters (triple quotes, single quotes, etc.) in your old_str so you truly replace the entire template block and don’t leave orphaned quotes behind.”

• “Label your edit calls with a short comment (e.g. ‘# replace user_processor_template block’) to keep track in code reviews which block is being targeted.”

• “Whenever possible, treat str_replace as a last-resort fallback—if you can parameterize or patch smaller pieces (insert, delete, etc.) you’ll reduce the odds of a brittle full-block replace.”

- "If str_replace fails, use this bash command to see invisible characters: python3 -c "with open('filename', 'r') as f: print(repr(f.read()))"

This shows all hidden characters like \r, \t, \u00a0, etc."
"""

def get_container_name() -> str:
    """
    Gets the name of the current docker container.
    Returns:
        str: Name of the container without leading slash
    """
    try:
        # Get the hostname (container ID)
        hostname = subprocess.check_output(['cat', '/etc/hostname']).decode('utf-8').strip()
        
        # Use docker inspect to get the container name
        container_name = subprocess.check_output(
            ['docker', 'inspect', '-f', '{{.Name}}', hostname]
        ).decode('utf-8').strip()
        
        # Remove leading slash if present
        return container_name.lstrip('/')
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fallback to hostname if docker command fails
        try:
            hostname = subprocess.check_output(['cat', '/etc/hostname']).decode('utf-8').strip()
            return hostname
        except:
            return "unknown_container"

def clean_networkedit_args(args) -> dict:
    """Specifically cleans NetworkEditTool arguments.
    Handles both:
    1. Entire dict being a string
    2. command_arguments being a string that should be a dict
    """
    # First handle if the entire thing is a string
    if isinstance(args, str):
        start = args.find('{')
        end = args.rfind('}')
        if start == -1 or end == -1:
            raise ValueError(f"No dict structure found in: {args}")
        args = ast.literal_eval(args[start:end + 1])
    
    # Now we should have a dict. Check if command_arguments is a string
    if isinstance(args.get('command_arguments'), str):
        cmd_args = args['command_arguments']
        start = cmd_args.find('{')
        end = cmd_args.rfind('}')
        if start == -1 or end == -1:
            raise ValueError(f"No dict structure found in command_arguments: {cmd_args}")
        args['command_arguments'] = ast.literal_eval(cmd_args[start:end + 1])
    
    return args

class EditHelper:
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """
    COMMAND_TEMPLATES = {
    "create": {"file_text": "your_content"},
    "str_replace": {"old_str": "text_to_replace", "new_str": "new_text"},
    "insert": {"insert_line": 42, "new_str": "text_to_insert"},
    "view": {},
    "undo_edit": {}
    }

    name: Literal["NetworkEditTool"] = "NetworkEditTool"

    _file_history: dict[Path, list[str]]

    def __init__(self):
        with open(tool_log_path, 'a') as f:
            f.write("\n\nEditHelper init called!\n")
        self._file_history = defaultdict(list)
        


    async def use_edit_helper(
        self,
        *,
        command: Command,
        path: str,
        # file_text: str | None = None,
        # view_range: list[int] | None = None,
        # old_str: str | None = None,
        # new_str: str | None = None,
        # insert_line: int | None = None,
        command_arguments: Optional[dict] = None,
        target_container: Optional[str] = None,
        # display_guide_manual: Optional[bool] = None,
        **kwargs,
    ):
        # if display_guide_manual:
        #     return guide_manual
        # First clean any stringified arguments
        if isinstance(command_arguments, str):
            try:
                command_arguments = clean_networkedit_args(command_arguments)
            except ValueError as e:
                raise ToolError(f"Invalid command_arguments format: {e}")
        if not isinstance(command_arguments, dict):
            raise ToolError(f"command_arguments must be a dictionary, got {type(command_arguments)}")
        
        # If we got a flat structure, convert it to nested
        if command_arguments is not None and command not in command_arguments:
            # Take the existing args and nest them under the command key
            command_arguments = {command: command_arguments}
        
        
        # Convert floats to ints
        if (command == 'view' and command_arguments 
            and 'view' in command_arguments 
            and 'view_range' in command_arguments['view']):
                view_range = command_arguments['view']['view_range']
                if isinstance(view_range, list):
                    command_arguments['view']['view_range'] = [int(x) if isinstance(x, float) else x for x in view_range]
        # If undo_edit, force empty dict regardless of what they sent
        if command == "undo_edit":
            command_arguments[command] = {}
        # Validate the structure is now correct
        if not isinstance(command_arguments[command], dict):
            raise ToolError(f"Arguments for '{command}' must be a dictionary, got {type(command_arguments[command])}")
        with open(tool_log_path, 'a') as f:
            f.write("\n\nuse_edit_helper entered!\n")
        _path = Path(path)
        with open(tool_log_path, 'a') as f:
            f.write("\n\nuse_edit_helper: path set!\n")
        self.validate_path(command, _path, target_container)
        with open(tool_log_path, 'a') as f:
            f.write("\n\nuse_edit_helper: path validated!\n")
        


        ##### ADD ARG DICT

        if command in ["view", "undo_edit"] and command_arguments is None:
            raise ToolError(
              "ERROR: Parameter `command_arguments` is required."
              "Example usage:\n"
              "- For view: {'command': 'view', 'path': '/path/to/file', 'command_arguments': {}}\n <-- notice the command_arguments\n"
              "- For undo_edit: {'command': 'undo_edit', 'path': '/path/to/file', 'command_arguments': {}}\n <-- notice the command_arguments\n\n"
            )
        if command in ["create", "str_replace", "insert"] and not command_arguments:
            raise ToolError(
                "ERROR: Parameter `command_arguments` is required.\n" 
                "\nINFO: This error may be due to chunking problems regarding the token limit OR dict escaping problems. The data for this tool is transferred through a dict. Read this entire error message -- the warning below contains extremely important context.\n\n"
                r"""
                ⚠️ WARNING! ANTHROPIC MAY HAVE EXPERIENCED A `CATCH` ERROR WHICH COULD HAVE DROPPED THE COMMAND_ARGUMENTS EVEN IF THEY WERE INCLUDED. As a result, they would appear not to have been included:
                `Anthropic may have failed to catch the tool call properly. This can happen if 1) the input exceeds the token limit OR 2) content inside any argument is not properly escaped and breaks the dict structure inside of the XML-like structure for the tool call parameters. When this occurs, Anthropic drops the un-escaped argument from the history and their models enter vicious loops. In other words, Anthropic fails to `catch`. The only way to fix it is by debugging it by chunking the input to make sure it is not a token length issue (while properly escaping the content inside the dict inside the XML-like tool call parameters). This must be fixed immediately and only the model performing the tool call can fix it. Pressure is on!`.

XML style example for Claude:
`str_replace` and `create` can be very tricky and the user will cry if you dont use it correctly and Anthropic fails to catch and loses the tool call. The important part is that inside command_arguments:
* It's valid dict\n* Special characters in the content (like quotes, newlines) need to be escaped
* But the dict structure itself (the curly braces and property names) doesn't need escaping
* The whole thing goes inside the parameter tag as is
* It must be within the token length limit
                
✅ So this style works (if using xml tags):
(Note: These examples assume the structure for the rest of the function call is already known)

```xml
<parameter name="command_arguments">{"old_str": "text with \"quotes\" and \n newlines", "new_str": "replacement text"}</parameter>
```

```xml
<parameter name="command_arguments">{"file_text": "text with \"quotes\" and \n newlines"}</parameter>
```

```xml
<parameter name="command_arguments">{"insert_line": 33, "new_str": "\"...\""}</parameter>
```

Example of python content:
```python
"def example():\n    print(\"Hello, world!\")" # notice how it is properly escaped and written with `\n`, which is rendered in python modules instead of being transferred literally
```

You can also use triple quotes to encase text.

❌ But this style is wrong:

```xml
<parameter name="command_arguments">{\"old_str\": \"text with "quotes"\"}</parameter>
```
^^ Escapes the dict and fails to escape inside dict
```xml
<parameter name"command_arguments">{\"file_text\": \"text with \"quotes\"\"}</parameter>
```
^^ Escapes the dict itself
```xml
<parameter name="insert_line">33</parameter>
<parameter name="command_arguments">{"new_str": ""...""}</parameter> 
```
^^ Nothing escaped
Chunking Instructions:
* If creating a file, use create for the first part and then insert... 
* If editing a file use str_replace/insert accordingly, depending on the context. 
* Escaping issues should've produced a `get` failure instead of this error, so try chunking it and see if that works, first.
              """
              "WARNING: CONTINUE WORKING ON THE SAME FILE TASK. MAKING A NEW FILE ARBITRARILY WILL NOT HELP YOU RESOLVE THIS ERROR."
              "\nFORMALLY ACKNOWLEDGE THIS ERROR AND APPLY ANY CHANGES, ACCORDINGLY. ARGUMENTS MUST BE USED PROPERLY.\nThe key is: Escape the CONTENT -- not the dict structure -- and CHUNK IT IF IT'S LONG!!!! **YOU SHOULD IMMEDIATELY BEGIN CHUNKING. BETTER TO GET THE WORK DONE THAN GET MORE ERRORS!!!**\n\n"
            )
        
        # # Extract args from command_arguments based on command type
        # file_text = command_arguments.get('file_text', None)
        # view_range = command_arguments.get('view_range', None)
        # old_str = command_arguments.get('old_str', None)
        # new_str = command_arguments.get('new_str', None)
        # insert_line = command_arguments.get('insert_line', None)
        # Add template definitions

        
        # Then in the function, before the .get() calls:
        try:
            # First check if command_arguments has a nested dictionary under the command name
            if command in command_arguments and isinstance(command_arguments[command], dict):
                # Use the nested dictionary for extraction
                command_specific_args = command_arguments[command]
                file_text = command_specific_args.get('file_text', None)
                view_range = command_specific_args.get('view_range', None)
                old_str = command_specific_args.get('old_str', None)
                new_str = command_specific_args.get('new_str', None)
                insert_line = command_specific_args.get('insert_line', None)
            else:
                # Extract args from command_arguments based on command type
                file_text = command_arguments.get('file_text', None)
                view_range = command_arguments.get('view_range', None)
                old_str = command_arguments.get('old_str', None)
                new_str = command_arguments.get('new_str', None)
                insert_line = command_arguments.get('insert_line', None)
        except TypeError as e:
            if str(e) == "'str' object has no attribute 'get'":
                # Detect common patterns
                hint = ""
                if isinstance(command_arguments, str):
                    if command_arguments.startswith("{") and command_arguments.endswith("}"):
                        hint = "\nLooks like you wrapped your dict in quotes. Remove the outer quotes!"
                    elif command_arguments in ("null", "None", ""):
                        hint = f"\nFor {command}, you need a proper dict structure."
        
                raise ToolError(
                    f"ERROR: command_arguments must be a dict, but you provided: {command_arguments}\n"
                    f"For the '{command}' command, use this format:\n"
                    f"{COMMAND_TEMPLATES[command]}"
                    f"{hint}\n\n"
                    "Common mistakes:\n"
                    "1. Wrapping the dict in quotes (BAD: \"{'key': 'value'}\")\n"
                    "2. Using null or empty string instead of {} for view/undo_edit\n"
                    "3. Not escaping quotes in content strings\n\n"
                    "If you're still having issues after fixing this, you may need to check the chunking guidelines."
                )
            raise
        if command == "view":
            with open(tool_log_path, 'a') as f:
                f.write("\n\nuse_edit_helper: entering command 'view'!\n")
            return await self.view(_path, view_range, target_container)

        elif command == "create":
            with open(tool_log_path, 'a') as f:
                f.write("\n\nuse_edit_helper: entering command 'create'!\n")
            if file_text is None:
                raise ToolError("ERROR: Parameter `file_text` is required for command: create")
            self.write_file(_path, file_text, target_container)
            self._file_history[_path].append(file_text)
            return ToolResult(output=f"File created successfully at: {_path}")





        elif command == "str_replace":
            if old_str is None:
                raise ToolError(
                    "ERROR: Parameter `old_str` is required for command: str_replace"
                )
            return self.str_replace(_path, old_str, new_str, target_container)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "ERROR: Parameter `insert_line` is required for command: insert"
                )
            if new_str is None:
                raise ToolError("ERROR: Parameter `new_str` is required for command: insert")
            return self.insert(_path, insert_line, new_str, target_container)
        elif command == "undo_edit":
            return self.undo_edit(_path, target_container)
        raise ToolError(
            f'ERROR: Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command))}'
        )

    def validate_path(self, command: str, path: Path, target_container: str | None = None):
        """
        Check that the path/command combination is valid.
        """
        # Check if its an absolute path
        with open(tool_log_path, 'a') as f:
                f.write("\n\nuse_edit_helper_validate_path: entered!\n")
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ToolError(
                f"ERROR: The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
    
        # For container paths, use docker exec to check existence and type
        if target_container:
            # Check if path exists
            result = subprocess.run(
                f"docker exec {target_container} test -e {path}",
                shell=True,
                capture_output=True
            )
            path_exists = result.returncode == 0
    
            # Check if it's a directory
            result = subprocess.run(
                f"docker exec {target_container} test -d {path}",
                shell=True,
                capture_output=True
            )
            is_directory = result.returncode == 0
    
            if not path_exists and command != "create":
                raise ToolError(
                    f"ERROR: The path {path} does not exist in container {target_container}. Please provide a valid path."
                )
            if path_exists and command == "create":
                raise ToolError(
                    f"ERROR: File already exists at: {path} in container {target_container}. `create` cannot be used to edit or overwrite files.  Use `str_replace`, `insert`, or make a new file with a `*_v*` suffix (i.e. hello_world_v2.py), depending on use case. Versioning is the safest."
                )
            if is_directory and command != "view":
                raise ToolError(
                    f"ERROR: The path {path} is a directory in container {target_container} and only the `view` command can be used on directories"
                )
        else:
            # Local path checks remain the same
            if not path.exists() and command != "create":
                raise ToolError(
                    f"ERROR: The path {path} does not exist. Please provide a valid path."
                )
            if path.exists() and command == "create":
                raise ToolError(
                    f"ERROR: File already exists at: {path}. Cannot overwrite files using command `create`."
                )
            if path.is_dir():
                if command != "view":
                    raise ToolError(
                        f"ERROR: The path {path} is a directory and only the `view` command can be used on directories"
                    )

    async def view(self, path: Path, view_range: list[int] | None = None, target_container: str | None = None):
        """Implement the view command"""
        if path.is_dir():  # This needs to change for container paths!
            if view_range:
                raise ToolError(
                    "ERROR: The `view_range` parameter is not allowed when `path` points to a directory."
                )
    
            if target_container:
                cmd = f"docker exec {target_container} find {path} -maxdepth 2 -not -path '*/\.*'"
            else:
                cmd = rf"find {path} -maxdepth 2 -not -path '*/\.*'"
    
            _, stdout, stderr = await run(cmd)
            if not stderr:
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            return CLIResult(output=stdout, error=stderr)
    
        file_content = self.read_file(path, target_container)
        
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError(
                    "ERROR: Invalid `view_range`. It should be a list of two integers."
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"ERROR: Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                # raise ToolError(
                #     f"ERROR: Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                # )
                # lol!
                
                final_line = n_lines_file

            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"ERROR: Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )



    def _represent_char(self, char):

        """Provides a readable representation of special characters."""

        if char == '\n':

            return '\\n'

        if char == '\t':

            return '\\t'

        if char == ' ':

            return ' ' # Represent space explicitly if needed, but often not necessary

        if not char.isprintable():

            return repr(char)

        return char

    def _generate_diff_error_v2(self, file_content: str, old_str: str, path: str) -> str:
        """
        Finds the best-matching region for old_str in file_content using SequenceMatcher
        and reports the first mismatched character with context.
        """
        # Use SequenceMatcher to find the best alignment of old_str within file_content
        s = difflib.SequenceMatcher(None, old_str, file_content, autojunk=False)
        match = s.find_longest_match(0, len(old_str), 0, len(file_content))
        
        # Determine the start of the slice in the file to compare against
        start_in_file = max(0, match.b - match.a)
        end_in_file = start_in_file + len(old_str)
        file_slice = file_content[start_in_file:end_in_file]
        
        # Find the first difference between old_str and the aligned file_slice
        mismatch_offset = -1
        for i in range(len(old_str)):
            if i >= len(file_slice):
                # old_str is longer than file_slice
                mismatch_offset = i
                break
            if old_str[i] != file_slice[i]:
                mismatch_offset = i
                break
        
        if mismatch_offset == -1:
            # This shouldn't happen if count was 0, but handle gracefully
            return f"ERROR: Your old_str input did not appear verbatim in {path}."
        
        # Calculate line and column of the mismatch in the file
        mismatch_pos_in_file = start_in_file + mismatch_offset
        lines_before_mismatch = file_content[:mismatch_pos_in_file].split('\n')
        line_number = len(lines_before_mismatch)
        column_number = len(lines_before_mismatch[-1]) + 1
        
        # Get context: last 10 characters before mismatch in old_str
        context_start = max(0, mismatch_offset - 10)
        file_context_start = max(0, mismatch_pos_in_file - 10)
        context_before = file_content[file_context_start:mismatch_pos_in_file]
        # context_before = old_str[context_start:mismatch_offset]
        
        
        # Get the mismatched characters
        expected_char = self._represent_char(old_str[mismatch_offset]) if mismatch_offset < len(old_str) else "<NOTE: END_OF_OLD_STR>"
        actual_char = self._represent_char(file_slice[mismatch_offset]) if mismatch_offset < len(file_slice) else "<NOTE: END_OF_FILE_CONTENT>"
        
        # Get a snippet of the surrounding file content for additional context
        snippet_start = max(0, mismatch_pos_in_file - 20)
        snippet_end = min(len(file_content), mismatch_pos_in_file + 20)
        file_snippet = file_content[snippet_start:snippet_end]
        
        error_message = (
            f"ERROR: old_str input did not appear verbatim in {path}.\n"
            f"Mismatch at line {line_number}, column {column_number}\n"
            f"Last 10 chars before mismatch: '{context_before}'\n"
            f"Expected next character: '{expected_char}'\n"
            f"Instead, received character: '{actual_char}'\n"
            f"File context around mismatch: '{file_snippet}'"
        )
        
        return error_message
    # def _generate_diff_error_v2(self, file_content: str, old_str: str, path: str) -> str:
    #     """
    #     Finds the best-matching region for old_str in file_content using SequenceMatcher
    #     and reports the first mismatched character.
    #     """
    #     # Use SequenceMatcher to find the best alignment of old_str within file_content
    #     s = difflib.SequenceMatcher(None, old_str, file_content, autojunk=False)
    #     match = s.find_longest_match(0, len(old_str), 0, len(file_content))
        
    #     # Determine the start of the slice in the file to compare against
    #     start_in_file = max(0, match.b - match.a)
    #     end_in_file = start_in_file + len(old_str)
    #     file_slice = file_content[start_in_file:end_in_file]
        
    #     # Find the first difference between old_str and the aligned file_slice
    #     for i in range(len(old_str)):
    #         if i >= len(file_slice):
    #             # old_str is longer than file_slice
    #             return (
    #                 f"ERROR: Your old_str input did not appear verbatim in {path}.\n"
    #                 f"Match stopped because old_str is longer than the file content.\n"
    #                 f"Expected more characters but reached end of file."
    #             )
    #         if old_str[i] != file_slice[i]:
    #             # Found the mismatch
    #             expected = self._represent_char(old_str[i])
    #             actual = self._represent_char(file_slice[i])
    #             return (
    #                 f"ERROR: Your old_str input did not appear verbatim in {path}.\n"
    #                 f"Expected: '{expected}'\n"
    #                 f"Found: '{actual}'"
    #             )
        
    #     # This shouldn't happen if count was 0
    #     return f"ERROR: Your old_str input did not appear verbatim in {path}."

    def _generate_diff_error(self, file_content: str, old_str: str, path: str) -> str:
        """
        Finds the best-matching region for old_str in file_content and generates a
        detailed error message pointing to the first character mismatch in that region.
        """
        # Use SequenceMatcher to find the best alignment of old_str within file_content
        s = difflib.SequenceMatcher(None, old_str, file_content, autojunk=False)
        match = s.find_longest_match(0, len(old_str), 0, len(file_content))

        # Determine the start of the slice in the file to compare against.
        # This aligns the start of old_str with the file content based on the best match.
        start_in_file = max(0, match.b - match.a)
        end_in_file = start_in_file + len(old_str)
        file_slice = file_content[start_in_file:end_in_file]

        # Now, find the first difference between old_str and the aligned file_slice
        mismatch_offset = -1
        for i, (char_old, char_slice) in enumerate(zip(old_str, file_slice)):
            if char_old != char_slice:
                mismatch_offset = i
                break
        
        # Handle case where the difference is due to length mismatch at the end
        if mismatch_offset == -1 and len(old_str) != len(file_slice):
            mismatch_offset = len(file_slice)

        # If no mismatch is found at all (should be rare if count==0), provide a generic error.
        if mismatch_offset == -1:
            return (
                f"ERROR: Your old_str input did not appear verbatim in {path}, "
                "but a detailed difference could not be determined. "
                "Consider using this bash command to see invisible characters: python3 -c \"with open('filename', 'r') as f: print(repr(f.read()))\""
            )

        # Calculate the line and column of the mismatch
        mismatch_abs_pos = start_in_file + mismatch_offset
        line_content_up_to_mismatch = file_content[:mismatch_abs_pos]
        line = line_content_up_to_mismatch.count('') + 1
        last_newline_pos = line_content_up_to_mismatch.rfind('')
        col = mismatch_abs_pos - last_newline_pos if last_newline_pos != -1 else mismatch_abs_pos + 1

        # Get the mismatched characters for the report
        expected_char = self._represent_char(old_str[mismatch_offset]) if mismatch_offset < len(old_str) else "EOF"
        actual_char_in_file = self._represent_char(file_slice[mismatch_offset]) if mismatch_offset < len(file_slice) else "EOF"

        error_message = (
            f"ERROR: Your old_str input did not appear verbatim in {path}."
            f"The closest match has a mismatch at Line {line}, Column {col}."
            f"Expected: '{expected_char}'"
            f"Received: '{actual_char_in_file}'"
            "This bash command to see invisible characters can be used to reveal if any hidden characters are preventing the old_str from matching: `python3 -c \"with open('filename', 'r') as f: print(repr(f.read()))\"`"
        )

        return error_message



    def str_replace(self, path: Path, old_str: str, new_str: str | None, target_container: str | None = None):

        """Implement the str_replace command, which replaces old_str with new_str in the file content"""

        # Read the file content

        file_content = self.read_file(path, target_container).expandtabs()

        old_str = old_str.expandtabs()

        new_str = new_str.expandtabs() if new_str is not None else ""


        # Check if old_str is unique in the file

        occurrences = file_content.count(old_str)

        if occurrences == 0:

            # Generate the detailed diff error instead of the generic one.

            error_message = self._generate_diff_error_v2(file_content, old_str, path)

            raise ToolError(error_message)

        elif occurrences > 1:

            file_content_lines = file_content.split("\n")

            lines = [

                idx + 1

                for idx, line in enumerate(file_content_lines)

                if old_str in line

            ]

            raise ToolError(

                f"ERROR: No replacement was performed. Multiple occurrences of old_str. Please ensure it is unique. The uniqueness depends on the lines above and below. Include more lines until the replacement is unique."
            )


        # Replace old_str with new_str

        new_file_content = file_content.replace(old_str, new_str)


        # Write the new content to the file

        self.write_file(path, new_file_content, target_container)


        # Save the content to history

        self._file_history[path].append(file_content)


        # Create a snippet of the edited section

        replacement_line = file_content.split(old_str)[0].count("\n")

        start_line = max(0, replacement_line - 4) # Using a fixed number for clarity

        end_line = replacement_line + 4 + new_str.count("\n")

        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])


        # Prepare the success message

        success_msg = f"The file {path} has been edited. "

        success_msg += self._make_output(

            snippet, f"a snippet of {path}", start_line + 1

        )

        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."


        return CLIResult(output=success_msg)
    # def str_replace(self, path: Path, old_str: str, new_str: str | None, target_container: str | None = None):
    #     """Implement the str_replace command, which replaces old_str with new_str in the file content"""
    #     # Read the file content
    #     file_content = self.read_file(path, target_container).expandtabs()
    #     old_str = old_str.expandtabs()
    #     new_str = new_str.expandtabs() if new_str is not None else ""

    #     # Check if old_str is unique in the file
    #     occurrences = file_content.count(old_str)
    #     if occurrences == 0:
    #         raise ToolError(
    #             f"ERROR: No replacement was performed, old_str did not appear verbatim in {path}. Make sure to exclude line numbers and line number tabs from the strings. View command autogenerates them."
    #         )
    #     elif occurrences > 1:
    #         file_content_lines = file_content.split("\n")
    #         lines = [
    #             idx + 1
    #             for idx, line in enumerate(file_content_lines)
    #             if old_str in line
    #         ]
    #         raise ToolError(
    #             f"ERROR: No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique. The uniqueness depends on the lines above and below. Include more lines until the replacement is unique."
    #         )

    #     # Replace old_str with new_str
    #     new_file_content = file_content.replace(old_str, new_str)

    #     # Write the new content to the file
    #     self.write_file(path, new_file_content, target_container)

    #     # Save the content to history
    #     self._file_history[path].append(file_content)

    #     # Create a snippet of the edited section
    #     replacement_line = file_content.split(old_str)[0].count("\n")
    #     start_line = max(0, replacement_line - SNIPPET_LINES)
    #     end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
    #     snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

    #     # Prepare the success message
    #     success_msg = f"The file {path} has been edited. "
    #     success_msg += self._make_output(
    #         snippet, f"a snippet of {path}", start_line + 1
    #     )
    #     success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

    #     return CLIResult(output=success_msg)

    def insert(self, path: Path, insert_line: int, new_str: str, target_container: str | None = None):
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        # Force insert_line to be an integer
        try:
            insert_line = int(insert_line)
        except (TypeError, ValueError):
            raise ToolError(f"ERROR: insert_line must be an integer, got {type(insert_line)}: {insert_line}")

        file_text = self.read_file(path, target_container).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"ERROR: Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text, target_container)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return CLIResult(output=success_msg)

    def undo_edit(self, path: Path, target_container: str | None = None):
        """Implement the undo_edit command."""
        if not self._file_history[path]:
            raise ToolError(f"ERROR: No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text, target_container)

        return CLIResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path, target_container: str | None = None):
        """Read the content of a file from a path or container path"""
        try:
            if target_container is None:
                return path.read_text()
            else:
                # Use docker exec to read from target container
                cmd = f"docker exec {target_container} cat {path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Docker exec failed: {result.stderr}")
                return result.stdout
        except Exception as e:
            raise ToolError(f"ERROR: Ran into {e} while trying to read {path}") from None
    
    def write_file(self, path: Path, content: str, target_container: str | None = None):
        try:
            # ---- local write ---------------------------------------------------
            if target_container is None:
                path.write_text(content)
                return

            # ---- remote write --------------------------------------------------
            encoded     = base64.b64encode(content.encode()).decode()
            quoted_path = shlex.quote(str(path))
            script      = f"echo '{encoded}' | base64 --decode > {quoted_path}"

            # Try three shells in turn: sudo-bash → bash → sh
            for cmd in (
                ["docker", "exec", "-i", target_container, "sudo", "bash", "-c", script],
                ["docker", "exec", "-i", target_container, "bash", "-c", script],
                ["docker", "exec", "-i", target_container, "sh",  "-c", script],
            ):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    break  # success
            else:
                # all attempts failed
                raise Exception(
                    result.stderr.strip() or f"docker exec exited {result.returncode}"
                )

        except Exception as e:
            raise ToolError(f"ERROR: Ran into {e} while trying to write to {path}") from None

    # def write_file(self, path: Path, file: str, target_container: Union[str, None] = None):
    #     """Write file content either locally or to target container"""
    #     try:
    #         if target_container is None:
    #             path.write_text(file)
    #         else:
    #             # Encode the file content to Base64
    #             encoded_file = base64.b64encode(file.encode('utf-8')).decode('utf-8')
                
    #             # Construct a docker exec command that decodes the Base64 string and writes it to the file
    #             cmd = f"docker exec {target_container} sudo bash -c 'echo {encoded_file!r} | base64 --decode > {path}'"
                
                
    #             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    #             if result.returncode != 0:
    #                 raise Exception(f"Docker exec failed: {result.stderr}")
    #     except Exception as e:
    #         raise ToolError(f"ERROR: Ran into {e} while trying to write to {path}") from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )

# Lazy initialization - get container name when needed
container_name = None

def get_current_container_name():
    global container_name
    if container_name is None:
        container_name = get_container_name()
    return container_name

def get_container_description():
    """Get description of current container for tool schema"""
    try:
        current_container = get_current_container_name()
        if current_container == "unknown_container":
            return "You are in a local container; source container = local."
        else:
            return f"The current container is `{current_container}`."
    except:
        return "You are in a local container; source container = local."

class NetworkEditToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'target_container': {
            'name': 'target_container',
            'type': 'str', 
            'description': f'Required container name to edit files in. For NetworkEditTool, container must be specified. {get_container_description()}',
            'required': True
        },
        'command': {
            'name': 'command',
            'type': 'str',
            'description': 'The command to execute. One of: view, create, str_replace, insert, undo_edit',
            'required': True
        },
        'path': {
            'name': 'path',
            'type': 'str',
            'description': 'Absolute path to file or directory',
            'required': True
        },
        'command_arguments': {
            'name': 'command_arguments',
            'type': 'dict',
            'description': 'Arguments specific to the selected command. To simply call view or undo_edit, include `command_arguments: {}`',
            'required': True,
            'nested': {
                'view': {
                    'view_range': {
                        'name': 'view_range',
                        'type': 'list',
                        'description': 'If provided, the file will be shown in the indicated line number range. e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting [start_line, -1] shows all lines from start_line to the end of the file.',
                        'items': {
                        'type': 'int'
                        },
                        'required': False
                    }
                },
                'create': {
                    'file_text': {
                        'name': 'file_text',
                        'type': 'str',
                        'description': 'The content of the file to be created. Cannot be empty and cannot overwrite existing files.',
                        'required': True
                    }
                },
                'str_replace': {
                    'old_str': {
                        'name': 'old_str',
                        'type': 'str',
                        'description': 'The string in the file to be replaced. Always try to fix by deleting and replacing an entire block in one shot, rather than isolating changes piecemeal and sequencing them.',
                        'required': True
                    },
                    'new_str': {
                        'name': 'new_str',
                        'type': 'str',
                        'description': 'The new string to replace the old string with',
                        'required': True
                    }
                },
                'insert': {
                    'insert_line': {
                        'name': 'insert_line',
                        'type': 'int',
                        'description': 'The new_str will be inserted AFTER this line',
                        'required': True
                    },
                    'new_str': {
                        'name': 'new_str',
                        'type': 'str',
                        'description': 'The string to insert',
                        'required': True
                    }
                },
                'undo_edit': {}  # No additional arguments needed for undo_edit
                
            }
        }
        # display_guide_manual
    }

  # Create the wrapper function
async def use_edit_tool_in_heaven(
    command: Command,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
    target_container: str = None,
    **kwargs,
) -> ToolResult:
    """Wrapper function for the edit tool to be used with BaseHeavenTool"""
    with open(tool_log_path, 'a') as f:
        f.write("\n\nEntering use_edit_tool_in_heaven\n")
    edit_func = EditHelper() # need to fix THIS
    with open(tool_log_path, 'a') as f:
        f.write("\n\nEditHelper() init called!!\n")
    return await edit_func.use_edit_helper(
        
        command=command,
        path=path,
        file_text=file_text,
        view_range=view_range,
        old_str=old_str,
        new_str=new_str,
        insert_line=insert_line,
        target_container=target_container,
        **kwargs
    )



class NetworkEditTool(BaseHeavenTool):
    name = "NetworkEditTool"
    description = (
        "Custom editing tool for viewing, creating and editing files with optional target container in the docker network. "
        "* State is persistent across command calls and discussions with the user\n"
        "* All command-specific arguments must be provided in the `command_arguments` dictionary\n"
        "* If `path` is a file, `view` displays the result of applying `cat -n`. "
        "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
        "* The `create` command cannot be used if the specified `path` already exists as a file\n"
        "* The `create` command cannot be used to edit files\n"
        "* If a command generates a long output, it will be truncated and marked with `<response clipped>`\n"
        "* The `undo_edit` command will revert the last edit made to the file at `path`\n"
        "* Important directories: 1. `your pwd/core/computer_use_demo/tools/base/tools/` to see tools, 2. `<your pwd>.../base/agents/<specific agent>/memories/...` to view `history_id`s"
        "* Long inputs for create, str_replace, and insert must be chunked into multiple operations (eg. create -> insert; str_replace -> insert; chain of str_replace, etc.)"
        "\nCOMMAND REQUIREMENTS:\n"
        "When inputting text, the dict must always be properly formatted."
        "This error `Error in tool 'NetworkEditTool': 'str' object has no attribute 'get'` means the dict was not properly formatted. It must be properly formatted, meaning that the dict itself must ALWAYS be a dict, not a string (ie the curlies of the dict cannot be wrapped in quotes)."
#         # This causes user anxiety 😭: 
# "command_arguments": "{'file_text': 'content'}"  # A STRING! NO NO NO
# # This makes user very happy:
# "command_arguments": {"file_text": "content"}    # A DICT! YES YES YES
        """
Escape Information:
For multiline content in file_text, use \n to indicate line breaks; they will be translated to real newlines.

Inside that JSON string:

• Escape double quotes as \".

• Single quotes need no escaping.

• Backslash must be doubled (\\) to produce a single literal backslash.

str_replace behaves identically: both old_str and new_str are ordinary JSON strings, so same escaping rules apply.

"""
        "Command Maps:"
        "1. 'view' command:\n"
        "   - Required: path\n"
        "   - Optional: view_range (list of 2 integers for line range)\n"
        "\n2. 'create' command:\n"
        "   - Required: path, file_text (cannot be empty - use placeholder word or code comment if needed)\n"
        "\n3. 'str_replace' command:\n"
        "   - Required: path, old_str, new_str\n"
        "\n4. 'insert' command:\n"
        "   - Required: path, new_str, insert_line\n"
        "\n5. 'undo_edit' command:\n"
        "   - Required: path\n"
    )
    args_schema = NetworkEditToolArgsSchema
    is_async = True
    
    _edit_helper = None  # Class-level singleton

    @classmethod
    def get_helper(cls):
        if cls._edit_helper is None:
            cls._edit_helper = EditHelper()
        return cls._edit_helper
      
    @classmethod     
    def create(cls, adk: bool = False):
        # Create the persistent helper first
        helper = cls.get_helper()
        
        # Create the actual implementation function that uses **kwargs
        async def implementation_func(**kwargs):
            try:
                # First clean the entire argument set if it's a string
                if isinstance(kwargs, str):
                    kwargs = clean_networkedit_args(kwargs)
                    
                # Now we know kwargs is a dict, check if command_arguments is a string
                if isinstance(kwargs.get('command_arguments'), str):
                    try:
                        kwargs = clean_networkedit_args(kwargs)
                    except SyntaxError as e:
                        if "unexpected character after line continuation character" in str(e):
                            raise ToolError(
                                r"""ERROR: unexpected character after line continuation character. Usually due to escaped dict keys and/or values, themselves, instead of properly escaped content. Don't do: `{\"key\": \"value\"}`"""
                                "\n\n"
                                r"SOLUTION: It needs to be a python dict, so just write it normally: `{'key': 'value'}`"
                            )
                        raise
                    
                return await helper.use_edit_helper(**kwargs)
            except Exception as e:
                raise
        
        # Set this as our internal implementation
        _implementation = implementation_func
        
        if adk:
            # For ADK, create a wrapper with properly typed parameters
            from typing import Dict, Any, Optional, List
            
            # Get parameter information from schema
            schema_instance = cls.args_schema()
            
            # Create parameter string based on schema
            param_list = []
            for param_name, param_info in schema_instance.arguments.items():
                param_list.append(param_name)
            
            # Create a function with proper parameter names
            func_def = f"""
async def {cls.name}({', '.join(param_list)}):
    '''{cls.description}'''
    kwargs = locals()
    return await _implementation(**kwargs)
    """
            
            # Execute to create the function
            namespace = {"_implementation": _implementation}
            exec(func_def, namespace)
            typed_func = namespace[cls.name]
            
            # Apply type annotations
            annotations = {}
            for param_name, param_info in schema_instance.arguments.items():
                if param_info['type'] in ('str', 'string'):
                    param_type = str
                elif param_info['type'] in ('int', 'integer'):
                    param_type = int
                elif param_info['type'] in ('float', 'number'):
                    param_type = float
                elif param_info['type'] in ('bool', 'boolean'):
                    param_type = bool
                elif param_info['type'] in ('list', 'array'):
                    param_type = List[str]  # Simplified for now
                elif param_info['type'] in ('dict', 'object'):
                    param_type = Dict[str, Any]
                else:
                    param_type = Any
                    
                # Make optional if not required
                if not param_info.get('required', True):
                    from typing import Optional
                    param_type = Optional[param_type]
                    
                annotations[param_name] = param_type
            
            typed_func.__annotations__ = annotations
            
            # Set as class function
            cls.func = typed_func
        else:
            # For LangChain, use the original implementation
            cls.func = implementation_func
        
        # Let parent create() do its thing
        return super().create(adk)

### OLD/WORKING
    # @classmethod
    # def create(cls, adk: bool = False):
    #     # Create the persistent helper first
    #     helper = cls.get_helper()
    #     schema_instance = cls.args_schema()
    #     pydantic_schema = cls.args_schema.to_pydantic_schema(schema_instance.arguments)
    
    #     async def wrapped_func(**kwargs):
    #         try:
    #             # First clean the entire argument set if it's a string
    #             if isinstance(kwargs, str):
    #                 kwargs = clean_networkedit_args(kwargs)
        
    #             # Now we know kwargs is a dict, check if command_arguments is a string
    #             if isinstance(kwargs.get('command_arguments'), str):
    #                 try:
    #                     kwargs = clean_networkedit_args(kwargs)
    #                 except SyntaxError as e:
    #                     if "unexpected character after line continuation character" in str(e):
    #                         raise ToolError(
    #                             r"""ERROR: unexpected character after line continuation character. Usually due to escaped dict keys and/or values, themselves, instead of properly escaped content. Don't do: `{\"key\": \"value\"}`"""
    #                             "\n\n"
    #                             r"SOLUTION: It needs to be a python dict, so just write it normally: `{'key': 'value'}`"
    #                         )
    #                     raise
        
    #             return await helper.use_edit_helper(**kwargs)
    #         except Exception as e:
    #             raise
    #     # Set this as our function
    #     wrapped_func.__name__ = cls.name
    #     wrapped_func.__doc__ = cls.description

    #     # Add type annotations from the Pydantic model
    #     annotations = {}
    #     from typing import Dict, List, Optional, Any
        
    #     # Extract annotations from Pydantic model fields
    #     for field_name, field in pydantic_schema.__fields__.items():
    #         # Get the field type - handle both Pydantic v1 and v2
    #         if hasattr(field, 'outer_type_'):
    #             field_type = field.outer_type_
    #         elif hasattr(field, 'annotation'):
    #             field_type = field.annotation
    #         else:
    #             # Fallback to schema
    #             arg_info = schema_instance.arguments.get(field_name, {})
    #             if arg_info.get('type') in ('str', 'string'):
    #                 field_type = str
    #             elif arg_info.get('type') in ('int', 'integer'):
    #                 field_type = int
    #             elif arg_info.get('type') in ('float', 'number'):
    #                 field_type = float
    #             elif arg_info.get('type') in ('bool', 'boolean'):
    #                 field_type = bool
    #             elif arg_info.get('type') in ('list', 'array'):
    #                 # Check for item type
    #                 item_type = arg_info.get('items', {}).get('type', 'string')
    #                 if item_type in ('str', 'string'):
    #                     field_type = List[str]
    #                 elif item_type in ('int', 'integer'):
    #                     field_type = List[int]
    #                 elif item_type in ('float', 'number'):
    #                     field_type = List[float]
    #                 elif item_type in ('bool', 'boolean'):
    #                     field_type = List[bool]
    #                 else:
    #                     field_type = List[str]
    #             elif arg_info.get('type') in ('dict', 'object'):
    #                 field_type = Dict[str, Any]
    #             else:
    #                 field_type = Any
                    
    #         # Check if it's a CustomList
    #         if str(type(field_type)).endswith('CustomList'):
    #             # Replace with standard list type
    #             arg_info = schema_instance.arguments.get(field_name, {})
    #             if arg_info.get('type') in ('list', 'array') and 'items' in arg_info:
    #                 item_type = arg_info['items'].get('type')
    #                 if item_type in ('str', 'string'):
    #                     field_type = List[str]
    #                 elif item_type in ('int', 'integer'):
    #                     field_type = List[int]
    #                 elif item_type in ('float', 'number'):
    #                     field_type = List[float]
    #                 elif item_type in ('bool', 'boolean'):
    #                     field_type = List[bool]
    #                 else:
    #                     field_type = List[str]
    #             else:
    #                 field_type = List[str]
                    
    #         annotations[field_name] = field_type
        
    #     # Apply annotations
    #     wrapped_func.__annotations__ = annotations
        
    #     cls.func = wrapped_func
        
    #     # Let parent create() do its thing
    #     return super().create(adk)

