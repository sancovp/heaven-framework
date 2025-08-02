# Utility for getting environment values in get_env_value.py

import os
import subprocess
import re
from enum import Enum

class KnownEnvVar(Enum):
    # GITHUB CONFIG
    GIT_USERNAME="GITHUB_USERNAME"
    GIT_TOKEN="GITHUB_TOKEN"
    GIT_CODE_REPO_BASE="GITHUB_CODE_REPO_BASE"
    GIT_REPO_URL="GITHUB_REPO_URL"
    GIT_MEMORY_REPO_BASE="GITHUB_MEMORY_REPO_BASE"
    GIT_MEMORY_REPO_URL="GITHUB_MEMORY_REPO_URL"
    GIT_MAIN_CODE_BRANCH_URL="GITHUB_MAIN_CODE_BRANCH_URL"
    GIT_BACKUP_CODE_BRANCH_URL="GITHUB_BACKUP_CODE_BRANCH_URL"
    GIT_MAIN_MEMORY_BRANCH_URL="GITHUB_MAIN_MEMORY_BRANCH_URL"
    GIT_BACKUP_MEMORY_BRANCH_URL="GITHUB_BACKUP_MEMORY_BRANCH_URL"
    # LLM MODEL CONFIG
    ANTHROPIC_KEY="ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL="ANTHROPIC_MODEL"
    OPENAI_API_KEY="OPENAI_API_KEY"
    OPENAI_MODEL="OPENAI_MODEL"
    GOOGLE_API_KEY="GOOGLE_API_KEY"
    GOOGLE_MODEL="GOOGLE_MODEL"
    GROQ_API_KEY="GROQ_API_KEY"
    GROQ_MODEL="GROQ_MODEL"
    DEEPSEEK_API_KEY="DEEPSEEK_API_KEY"
    DEEPSEEK_MODEL="DEEPSEEK_MODEL"
    # AI PERSONA CONFIG
    MODEL_DEFAULT_TEMP="MODEL_DEFAULT_TEMP"
    EVOLUTION_SYSTEM_NAME="EVOLUTION_SYSTEM_NAME"
    CONTAINER_AGENT_TYPE="CONTAINER_AGENT_TYPE"
    SYSTEM_CONNECTOR="SYSTEM_CONNECTOR"
    NETWORK_TYPE="NETWORK_TYPE"
    EVOLUTION_SYSTEM_MEANING="EVOLUTION_SYSTEM_MEANING"
    # HEAVEN DATA & STORAGE CONFIG
    HEAVEN_DATA_DIR="HEAVEN_DATA_DIR"
    HEAVEN_STORAGE_BACKEND="HEAVEN_STORAGE_BACKEND"
    HEAVEN_API_KEY="HEAVEN_API_KEY"
    # OTHER
    PWD="PWD" #/home/$USERNAME
    USERNAME="USERNAME"
    HOME="HOME" #/home/$USERNAME
    # Add more known environment variable keys as needed

class EnvConfigUtil:
    
    # Global constant with the list of target container names.
    TARGET_CONTAINERS = ["image_of_god", "creation_of_god"]
    CONFIG_FILE = "/home/GOD/system_config.sh"

    @staticmethod
    def _update_env_val():
        result = subprocess.run(
            'bash -c "source ~/system_config.sh && env"',
            shell=True, capture_output=True, text=True
        )
        # Update the current process's environment
        for line in result.stdout.splitlines():
            key, _, value = line.partition("=")
            os.environ[key] = value


    @staticmethod
    def get_env_value(key, default=None):
        """
        Retrieve an environment variable's value.

        Args:
            key (str or KnownEnvVar): The environment variable's name or enum member.
            default (any, optional): A default value if the env variable is not set.

        Returns:
            The value of the environment variable, or the default if not found.
        """
        EnvConfigUtil._update_env_val()
        # If key is an Enum member, extract its value.
        if isinstance(key, KnownEnvVar):
            key = key.value    
        return os.environ.get(key, default)

    @staticmethod
    def propagate_system_config():
        """
        Copies /home/GOD/system_config.sh from the current container to each target container
        defined in the TARGET_CONTAINERS global variable. The file is copied to /home/GOD
        in each target container.

        Returns:
            dict: A mapping of container name to a tuple (success, message).
                  'success' (bool) indicates if the copy was successful,
                  'message' contains stdout or error message.
        """
        results = {}
        for container in EnvConfigUtil.TARGET_CONTAINERS:
            command = f"docker cp /home/GOD/system_config.sh {container}:/home/GOD"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"File copied successfully to {container}!")
                results[container] = (True, result.stdout.strip())
            else:
                error_msg = result.stderr.strip()
                print(f"Error copying file to {container}: {error_msg}")
                results[container] = (False, error_msg)
        return results

    
    @staticmethod
    def put_env_value(key:str, value:str=None):
        """
        Updates the configuration file with the provided export variable.
        The function preserves all blank lines and comment lines. If an export line
        for the given key already exists, it is replaced with the new value.
        Otherwise, a new export line is appended to the end of the file.

        Args:
            key (str): The environment variable name.
            default (str, optional): The value to set for the environment variable.
                                     If None, no update is made.
        """
        if value is None:
            return
        
        # The new export line we want to have in the file.
        new_line = f'export {key}="{value}"\n'

        # Try to read the existing configuration file.
        try:
            with open(EnvConfigUtil.CONFIG_FILE, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            lines = []

        # Regex to match export lines that are not commented out.
        # This pattern allows for leading whitespace before 'export'.
        pattern = re.compile(r"^\s*export\s+(\w+)\s*=.*$")
        found = False
        new_lines = []

        for line in lines:
            match = pattern.match(line)
            if match and match.group(1) == key:
                # Replace the line for this key with the new value.
                new_lines.append(new_line)
                found = True
            else:
                # Retain the original line (be it a blank line, comment, or another export).
                new_lines.append(line)
                
        if not found:
            # If the variable was not found, append the new export line.
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(new_line)

        # Write the updated content back to the configuration file.
        with open(EnvConfigUtil.CONFIG_FILE, "w") as file:
            file.writelines(new_lines)
    
    @staticmethod
    def get_heaven_data_dir():
        """Get the heaven data directory, ensuring it exists and is properly configured."""
        data_dir = EnvConfigUtil.get_env_value(KnownEnvVar.HEAVEN_DATA_DIR)
        if not data_dir:
            raise EnvironmentError(
                "🚨 HEAVEN_DATA_DIR environment variable is REQUIRED!\n\n"
                "No ~/.heaven fallback - explicit data directory prevents deployment disasters.\n"
                "Set: export HEAVEN_DATA_DIR=/path/to/project/heaven-data"
            )
        
        # Ensure directory structure exists
        subdirs = [
            'agents',                    # Agent histories and persistence
            'registry',                  # Registries (brain configs, personas, modes)  
            'progenitor',               # Agent DNA and system prompts
            'modules',                  # Generated Python modules (must be local)
            'generated',                # AI-created Python files (must be local)
            'configs/hermes_configs'    # Hermes workflow configurations
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
        
        return data_dir
    
    @staticmethod
    def get_heaven_storage_backend():
        """Get the storage backend type with validation."""
        backend = EnvConfigUtil.get_env_value(KnownEnvVar.HEAVEN_STORAGE_BACKEND, 'local')
        api_key = EnvConfigUtil.get_env_value(KnownEnvVar.HEAVEN_API_KEY)
        
        if backend == 'premium_api' and not api_key:
            raise EnvironmentError(
                "🚨 CANNOT CONNECT TO HEAVEN_API WHILE STORAGE_BACKEND=premium_api!\n\n"
                "PLEASE CONFIGURE YOUR API KEY:\n"
                "export HEAVEN_API_KEY=your_premium_key\n\n"
                "Or switch to local storage:\n"
                "export HEAVEN_STORAGE_BACKEND=local"
            )
        
        if backend == 'local' and api_key:
            print("⚠️  WARNING: HEAVEN_API_KEY is set but STORAGE_BACKEND=local")
            print("    Did you mean: export HEAVEN_STORAGE_BACKEND=premium_api ?")
        
        if backend not in ['local', 'premium_api']:
            raise ValueError(f"Invalid HEAVEN_STORAGE_BACKEND: {backend}. Must be 'local' or 'premium_api'")
        
        return backend
    
    @staticmethod
    def is_premium_storage():
        """Check if using premium API storage."""
        return EnvConfigUtil.get_heaven_storage_backend() == 'premium_api'

# wrapper class to allow the env properties to be loaded when used
class DynamicString(str):
    def __new__(cls, func, *args, **kwargs):
        # Call the function once for the initial value
        initial_value = func(*args, **kwargs)
        # Create the str instance with the initial value
        obj = super().__new__(cls, initial_value)
        # Save the callable and its arguments for future dynamic conversion
        obj._func = func
        obj._args = args
        obj._kwargs = kwargs
        return obj

    def __str__(self):
        # Every time __str__ is called, re-run the function for the most up-to-date value
        return self._func(*self._args, **self._kwargs)


# Example usage:
if __name__ == "__main__":
    # Using an enum
    print("GIT_REPO:", EnvConfigUtil.get_env_value(KnownEnvVar.GIT_REPO))
    # Using a direct string (for variables not in the enum)
    print("CUSTOM_VAR:", EnvConfigUtil.get_env_value("ANTHROPIC_API_KEY"))
    EnvConfigUtil.put_env_value("ANTHROPIC_API_KEY", "NA")
    EnvConfigUtil.propagate_system_config()