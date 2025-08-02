# Base git container utilities in git_container_base_util.py

#!/usr/bin/env python3
import subprocess
import sys
import argparse
from enum import Enum
import os
# from .get_env_value import EnvConfigUtil

# Define an enum for repo types.
class RepoType(Enum):
    CODE = "code"
    MEMORY = "memory"

# mapping of the container_name to the assoicated folder in github directory
mapping = {
    "mind_of_god": "mind_of_god",
    "image_of_god": "image_of_god",
    "creation_of_god": "creation_of_god"
}
auth_username = os.getenv("GITHUB_USERNAME")    # EnvConfigUtil.get_env_value(KnownEnvVar.GIT_USERNAME)
auth_token = os.getenv("GITHUB_TOKEN")          # EnvConfigUtil.get_env_value(KnownEnvVar.GIT_TOKEN)

# List of file/folder names to ignore (delete) from the final content. so they are not committed to github
ignore_patterns = [".DS_Store", ".zed", ".venv", ".pytest_cache", "__pycache__", "index.html", ".anthropic", ".mozilla"]
# Memory file filter pattern.
memory_file_patterns = ["202[4-9]*.md", "20[3-9][0-9]*.md", "*.txt", "*memories.json", "*_summary.md"]
# Global whitelist: files that should never be removed from the code repository. but excluded from the memory repo
code_repo_whitelist = ["requirements.txt", "dev_requirements.txt", "dev-requirements.txt"]


def _run_command(command, capture_output=False, check=True):
    result = subprocess.run(
        command, shell=True, text=True,
        capture_output=capture_output
    )
    if check and result.returncode != 0:
        err = result.stderr.strip() if result.stderr else "Unknown error"
        raise Exception(f"Command failed: {command}\nError: {err}")
    if capture_output:
        return result.stdout.strip()
    return None


def _run_container_command(container, command, capture_output=False, check=True):
    """
    Runs a command inside the specified container using docker exec.
    The command is executed in bash.
    """
    # Wrap the command in single quotes so that it is interpreted by bash inside the container.
    full_cmd = f"docker exec {container} bash -c '{command}'"
    return _run_command(full_cmd, capture_output=capture_output, check=check)


def _is_directory_in_container(container, path):
    """
    Determines if a given path in the container is a directory.
    """
    cmd = f"if [ -d '{path}' ]; then echo yes; else echo no; fi"
    output = _run_container_command(container, cmd, capture_output=True)
    return output.strip() == "yes"


def get_container_user_name(target_container):
    return _run_container_command(target_container, "whoami", capture_output=True)


def push_target_container_content_to_target_folder_on_repo(
    target_container,      # Docker container name in which operations will run.
    commit_message,        # Commit message for the git commit.
    repo_url,              # GitHub repository URL to push changes to.
    target_folder,         # Target folder or file inside the container to push; ignored if push_all_container_contents is True.
    push_all_core_contents: bool,  # If True, use content from /home/{username}/core.
    target_repo_folder: str = None,  # Optional: Override the mapped folder in the GitHub repo.
    repo_type: RepoType = RepoType.CODE  # New parameter as an enum: RepoType.CODE or RepoType.MEMORY.
):
    # Get the container's username (expected to be one of the three)
    username = get_container_user_name(target_container)
    
    # Determine the source path in the container.
    if push_all_core_contents:
        container_source = f"/home/{username}/core"
    else:
        container_source = target_folder

    # Compute the relative path by stripping the "/home/{username}" prefix if present.
    user_home_prefix = f"/home/{username}"
    if container_source.startswith(user_home_prefix):
        relative_path = container_source[len(user_home_prefix):]  # e.g. "/core/test.txt"
    else:
        relative_path = container_source

    # Prepare the repository clone location inside the container.
    repo_clone_dir = "/tmp/repo_clone"

    # Remove any previous clone and create a fresh directory.
    print("Cleaning up previous clone (if any) inside container...")
    _run_container_command(target_container, f"rm -rf {repo_clone_dir}")
    _run_container_command(target_container, f"mkdir -p {repo_clone_dir}")

    # Check if a branch is specified in the URL (using '#' as delimiter).
    branch_name = None
    if '#' in repo_url:
        repo_url, branch_name = repo_url.split('#', 1)

    # Embed credentials for the private repo.
    if repo_url.startswith("https://github.com/"):
        auth_repo_url = f"https://{auth_username}:{auth_token}@" + repo_url[len("https://"):]
    else:
        auth_repo_url = repo_url

    # Clone the repo inside the container, using -b if a branch is specified.
    if branch_name:
        clone_cmd = f"git clone -b {branch_name} {auth_repo_url} {repo_clone_dir}"
    else:
        clone_cmd = f"git clone {auth_repo_url} {repo_clone_dir}"
    print(f"Cloning repo {repo_url} (branch: {branch_name if branch_name else 'default'}) inside container into {repo_clone_dir}...")
    _run_container_command(target_container, clone_cmd)

    # Set git user configuration inside the cloned repository to avoid identity errors.
    print("Setting git user configuration inside container...")
    _run_container_command(
        target_container,
        f"cd {repo_clone_dir} && git config user.email '{auth_username}' && git config user.name '{auth_username}'"
    )
    # Disable file mode tracking so that permission changes are ignored.
    _run_container_command(target_container, f"cd {repo_clone_dir} && git config core.fileMode false")

    # Compute the mapped folder in the Git repo.
    # If target_repo_folder is provided, use it; otherwise use the mapping based on container name (target_container).
    if target_repo_folder is not None:
        mapped_folder = target_repo_folder
    else:
        mapped_folder = mapping[target_container]
    # Final destination in the cloned repo.
    final_dest = f"{repo_clone_dir}/{mapped_folder}{relative_path}"

    print(f"Mapping container path '{container_source}' to repo destination '{final_dest}'...")

    # Remove the destination so deletions are preserved.
    _run_container_command(target_container, f"rm -rf '{final_dest}'")

    # Create parent directory for final_dest.
    _run_container_command(target_container, f"mkdir -p $(dirname '{final_dest}')")

    # Now, copy the content from container_source to final_dest.
    if _is_directory_in_container(target_container, container_source):
        # If it's a directory, copy its contents.
        # The trailing '/.' ensures we copy the content inside container_source.
        _run_container_command(target_container, f"mkdir -p '{final_dest}'")
        _run_container_command(target_container, f"cp -r '{container_source}/.' '{final_dest}'")
    else:
        # Otherwise, copy the single file.
        _run_container_command(target_container, f"cp '{container_source}' '{final_dest}'")

    # Apply filtering based on repo type.
    if repo_type == RepoType.CODE:
        # Derive non-.txt memory patterns from the global memory_file_patterns list.
        non_txt_memory_patterns = [pattern for pattern in memory_file_patterns if pattern != "*.txt"]
        for pattern in non_txt_memory_patterns:
            print(f"Filtering out files matching non-.txt pattern '{pattern}' for code repo...")
            _run_container_command(
                target_container,
                f'bash -c "if [ -d \\"{final_dest}\\" ]; then find \\"{final_dest}\\" -name \\"{pattern}\\" -exec rm -rf {{}} +; fi"'
            )
        # Build a whitelist expression for .txt files using the global code_repo_whitelist.
        # This will produce a string like: -not -name "requirements.txt" -not -name "dev_requirements.txt" -not -name "dev-requirements.txt"
        whitelist_expr = " ".join([f'-not -name "{w}"' for w in code_repo_whitelist])
        print("Filtering out .txt memory files for code repo (excluding whitelisted files)...")
        _run_container_command(
            target_container,
            f'bash -c "if [ -d \\"{final_dest}\\" ]; then find \\"{final_dest}\\" -type f -name \\"*.txt\\" {whitelist_expr} -exec rm -rf {{}} +; fi"'
        )
        # Remove any additional files/folders matching the standard ignore patterns.
        for pattern in ignore_patterns:
            print(f"Filtering out ignore pattern '{pattern}' for code repo...")
            _run_container_command(
                target_container,
                f'bash -c "if [ -d \\"{final_dest}\\" ]; then find \\"{final_dest}\\" -name \\"{pattern}\\" -exec rm -rf {{}} +; fi"'
            )
    elif repo_type == RepoType.MEMORY:
        # Build allowed expression dynamically from memory_file_patterns and code_repo_whitelist.
        allowed_expr_parts = []
        for pattern in memory_file_patterns:
            if pattern == "*.txt":
                # For .txt files, allow if they match "*.txt" AND are NOT one of the whitelisted names.
                txt_predicate = '-name "*.txt" ' + " ".join([f'-and -not -name "{w}"' for w in code_repo_whitelist])
                allowed_expr_parts.append(f"\\( {txt_predicate} \\)")
            else:
                allowed_expr_parts.append(f'-name "{pattern}"')
        allowed_expr = " -o ".join(allowed_expr_parts)
        print(f"Allowed expression for memory repo: {allowed_expr}")
        print("Filtering out files that are not allowed for memory repo...")
        _run_container_command(
            target_container,
            f'bash -c "if [ -d \\"{final_dest}\\" ]; then find \\"{final_dest}\\" -type f -not \\( {allowed_expr} \\) -exec rm -rf {{}} +; fi"'
        )
        # Also remove any files/folders matching the global ignore patterns.
        for pattern in ignore_patterns:
            print(f"Filtering out ignore pattern '{pattern}' for memory repo...")
            _run_container_command(
                target_container,
                f'bash -c "if [ -d \\"{final_dest}\\" ]; then find \\"{final_dest}\\" -name \\"{pattern}\\" -exec rm -rf {{}} +; fi"'
            )
    else:
        raise Exception(f"Unknown repo_type: {repo_type}")

    # Remove all symlinks.
    print("Removing all symlinks so not to be committed to github...")
    _run_container_command(
        target_container,
        f"find '{final_dest}' -type l -delete"
    )

    # Stage, commit, and push the changes inside the container.
    print("Staging changes in the repository inside container...")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git add -A")
    try:
        _run_container_command(target_container, f"cd {repo_clone_dir} && git diff-index --quiet HEAD --")
        print("No changes detected. Exiting without committing.")
        return
    except Exception:
        # Exception indicates that changes exist.
        pass

    print("Committing changes inside container...")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git commit -m \"{commit_message}\"")
    print("Pushing changes from container to remote repo...")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git push")
    print("Changes pushed successfully.")


def pull_from_target_folder_in_repo_to_target_container(
    target_container,           # Docker container name in which operations will run.
    target_path_in_container,   # The destination path in the container to update.
    repo_url,                   # Remote repository URL to pull from.
    repo_folder,              # The folder in the repository (inside the cloned repo) to pull.
    replace_container_contents: bool  # If True, remove the existing container content at target_path_in_container.
):
    # Prepare the repository clone location inside the container.
    repo_clone_dir = "/tmp/repo_clone"
    print("Cleaning up previous clone (if any) inside container...")
    _run_container_command(target_container, f"rm -rf {repo_clone_dir}")
    _run_container_command(target_container, f"mkdir -p {repo_clone_dir}")

    # Check if a branch is specified in the URL.
    branch_name = None
    print(f"Repo URL: {repo_url}")
    if '#' in repo_url:
        repo_url, branch_name = repo_url.split('#', 1)

    # Embed credentials.
    if repo_url.startswith("https://"):
        auth_repo_url = f"https://{auth_username}:{auth_token}@" + repo_url[len("https://"):]
    else:
        auth_repo_url = repo_url

    # Clone the repo inside the container.
    if branch_name:
        clone_cmd = f"git clone -b {branch_name} {auth_repo_url} {repo_clone_dir}"
    else:
        clone_cmd = f"git clone {auth_repo_url} {repo_clone_dir}"
    print(f"Cloning repo {repo_url} (branch: {branch_name if branch_name else 'default'}) inside container into {repo_clone_dir}...")
    _run_container_command(target_container, clone_cmd)

    # Determine the source path inside the cloned repo.
    # Here, target_folder is the folder (relative to the repo clone root) that we want to pull.
    source_path = f"{repo_clone_dir}/{repo_folder}"
    print(f"Source path in repo: '{source_path}'")

    # Optionally, if replace_container_contents is True, remove the target path in the container.
    if replace_container_contents:
        print(f"Removing existing content at container path '{target_path_in_container}'...")
        _run_container_command(target_container, f"rm -rf '{target_path_in_container}'")
        _run_container_command(target_container, f"mkdir -p '{target_path_in_container}'")
    else:
        _run_container_command(target_container, f"mkdir -p $(dirname '{target_path_in_container}')")

    # Copy content from the repo's target folder to the container target path.
    if _is_directory_in_container(target_container, source_path):
        print(f"Copying directory contents from '{source_path}' to '{target_path_in_container}'...")
        _run_container_command(target_container, f"cp -r '{source_path}/.' '{target_path_in_container}'")
    else:
        print(f"Copying file from '{source_path}' to '{target_path_in_container}'...")
        _run_container_command(target_container, f"cp '{source_path}' '{target_path_in_container}'")

    # Clean up the cloned repository.
    _run_container_command(target_container, f"rm -rf {repo_clone_dir}")
    print("Pull operation completed successfully.")


def erase_target_repo_folder(
    target_container,      # Container name (e.g. "creation_of_god")
    commit_message,        # Commit message for the erase operation
    repo_url,              # Remote repository URL to operate on
    target_repo_folder: str = None  # Optional override; if None, use mapping[target_container]
):
    """
    Erases (empties) the top-level folder in the repository corresponding to target_container.
    This function clones the repository into a temporary location inside the container,
    removes the mapped folder, recreates it empty, stages the changes, commits and pushes.
    Only the folder for target_container is modified.
    """

    # Determine the mapped folder.
    if target_repo_folder is not None:
        mapped_folder = target_repo_folder
    else:
        mapped_folder = mapping[target_container]

    # Set the local clone directory.
    repo_clone_dir = "/tmp/erase_repo_clone"
    print(f"[{target_container}] Cleaning up previous clone at {repo_clone_dir}...")
    _run_container_command(target_container, f"rm -rf {repo_clone_dir} && mkdir -p {repo_clone_dir}")

    # Check if a branch is specified in the URL.
    branch_name = None
    if '#' in repo_url:
        repo_url, branch_name = repo_url.split('#', 1)

    # Embed credentials.
    if repo_url.startswith("https://"):
        auth_repo_url = f"https://{auth_username}:{auth_token}@" + repo_url[len("https://"):]
    else:
        auth_repo_url = repo_url

    # Clone the repo into repo_clone_dir.
    if branch_name:
        clone_cmd = f"git clone -b {branch_name} {auth_repo_url} {repo_clone_dir}"
    else:
        clone_cmd = f"git clone {auth_repo_url} {repo_clone_dir}"
    print(f"[{target_container}] Cloning repo into {repo_clone_dir}...")
    _run_container_command(target_container, clone_cmd)

    # Compute the target folder path inside the clone.
    target_folder_path = f"{repo_clone_dir}/{mapped_folder}"
    print(f"[{target_container}] Erasing folder: {target_folder_path}")
    _run_container_command(target_container, f"rm -rf '{target_folder_path}'")
    print(f"[{target_container}] Recreating folder {target_folder_path} as empty...")
    _run_container_command(target_container, f"mkdir -p '{target_folder_path}'")
    _run_container_command(target_container, f"touch '{target_folder_path}/README.md'")

    # Set git user configuration inside the cloned repository to avoid identity errors.
    print("Setting git user configuration inside container...")
    _run_container_command(
        target_container,
        f"cd {repo_clone_dir} && git config user.email '{auth_username}' && git config user.name '{auth_username}'"
    )
    # Stage, commit, and push the change.
    print(f"[{target_container}] Staging changes for erase operation...")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git add -A")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git commit -m \"{commit_message}\"")
    _run_container_command(target_container, f"cd {repo_clone_dir} && git push")
    print(f"[{target_container}] Erase operation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push, pull, or erase container content on a remote repo (all operations run inside the container) with path remapping."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run: push, pull, or erase.")

    # Push subcommand
    push_parser = subparsers.add_parser("push", help="Push container content to a remote repo.")
    push_parser.add_argument("target_container", help="Name of the Docker container")
    push_parser.add_argument("commit_message", help="Git commit message")
    push_parser.add_argument("repo_url", help="Remote repository URL")
    push_parser.add_argument("target_folder", help="Target folder or file in the container to push (ignored if push_all is set)")
    push_parser.add_argument("--push_all", action="store_true", help="If set, push all content from /home/{username}/core instead of target_folder")
    push_parser.add_argument("--target_repo_folder", default=None, help="Optional: Override the mapped folder in the GitHub repo.")
    push_parser.add_argument("--repo_type", default="code", type=lambda s: RepoType(s.lower()), choices=[x.value for x in RepoType],
                             help="Type of repository: 'code' or 'memory'. Default is 'code'.")
    # Pull subcommand
    pull_parser = subparsers.add_parser("pull", help="Pull content from a remote repo into the container.")
    pull_parser.add_argument("target_container", help="Name of the Docker container")
    pull_parser.add_argument("target_path_in_container", help="The destination path in the container to update")
    pull_parser.add_argument("repo_url", help="Remote repository URL")
    pull_parser.add_argument("target_folder", help="Folder in the repository (relative to the repo root) to pull")
    pull_parser.add_argument("--replace", action="store_true", help="If set, replace the existing container content at the target path")
    # Erase subcommand
    erase_parser = subparsers.add_parser("erase", help="Erase (empty) the target repo folder corresponding to a container.")
    erase_parser.add_argument("target_container", help="Name of the Docker container")
    erase_parser.add_argument("commit_message", help="Git commit message for the erase operation")
    erase_parser.add_argument("repo_url", help="Remote repository URL")
    erase_parser.add_argument("--target_repo_folder", default=None, help="Optional: Override the mapped folder in the GitHub repo.")

    args = parser.parse_args()

    try:
        if args.command == "push":
            push_target_container_content_to_target_folder_on_repo(
                target_container=args.target_container,
                commit_message=args.commit_message,
                repo_url=args.repo_url,
                target_folder=args.target_folder,
                push_all_core_contents=args.push_all,
                target_repo_folder=args.target_repo_folder
            )
        elif args.command == "pull":
            pull_from_target_folder_in_repo_to_target_container(
                target_container=args.target_container,
                target_path_in_container=args.target_path_in_container,
                repo_url=args.repo_url,
                repo_folder=args.target_folder,
                replace_container_contents=args.replace
            )
        elif args.command == "erase":
            erase_target_repo_folder(
                target_container=args.target_container,
                commit_message=args.commit_message,
                repo_url=args.repo_url,
                target_repo_folder=args.target_repo_folder
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)