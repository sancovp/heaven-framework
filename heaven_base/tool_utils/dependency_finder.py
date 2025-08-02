#!/usr/bin/env python3

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional


class TargetFinder(ast.NodeVisitor):
    """AST visitor to find a specific function or class in a Python module."""
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.found = False
        self.target_node = None
        self.line_range = None
        
    def visit_FunctionDef(self, node):
        if node.name == self.target_name:
            self.found = True
            self.target_node = node
            self.line_range = (node.lineno, node.end_lineno)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        if node.name == self.target_name:
            self.found = True
            self.target_node = node
            self.line_range = (node.lineno, node.end_lineno)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        if node.name == self.target_name:
            self.found = True
            self.target_node = node
            self.line_range = (node.lineno, node.end_lineno)
        self.generic_visit(node)

class DependencyCollector(ast.NodeVisitor):
    """Collects external dependencies in a Python module."""
    
    def __init__(self, current_file: Path):
        self.current_file = current_file
        self.function_calls = []
        self.class_refs = []
        self.imports = []
        self.references = []

    def visit_Assign(self, node):
        # Check if the value is a name that might be a function or class
        if isinstance(node.value, ast.Name):
            # Add to a new list of potential function/class references
            self.references.append(node.value.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.function_calls.append(f"{node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        # Track parent classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.class_refs.append(base.id)
        self.generic_visit(node)
        
    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for name in node.names:
                self.imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)


class CrossFileDependencyAnalyzer:
    """Analyzes cross-file dependencies for a given function or class."""
    
    def __init__(self, target_name: str, search_dirs: List[str]):
        self.target_name = target_name
        self.search_dirs = [Path(d) for d in search_dirs]
        self.target_file = None
        self.target_line_range = None
        self.external_dependencies = []
        self.visited = set()
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the search directories."""
        python_files = []
        for directory in self.search_dirs:
            if not directory.exists():
                continue
            for file_path in directory.glob('**/*.py'):
                python_files.append(file_path)
        return python_files
        
    def find_target(self) -> bool:
        """Find the target function or class in the codebase."""
        python_files = self.find_python_files()
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                tree = ast.parse(code)
                finder = TargetFinder(self.target_name)
                finder.visit(tree)
                if finder.found:
                    self.target_file = file_path
                    self.target_line_range = finder.line_range
                    return True
            except Exception as e:
                if self.target_name in str(file_path):
                    print(f"Error parsing {file_path}: {e}")
        return False

    def find_file_for_item(self, item_name: str) -> Optional[Tuple[Path, Tuple[int, int]]]:
        """Find the file containing a specific item (function/class)."""
        if item_name in self.visited:
            return None
            
        self.visited.add(item_name)
        python_files = self.find_python_files()
        
        for file_path in python_files:
            if file_path == self.target_file:
                continue  # Skip the file we're analyzing
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                tree = ast.parse(code)
                finder = TargetFinder(item_name)
                finder.visit(tree)
                if finder.found:
                    return (file_path, finder.line_range)
            except Exception:
                pass
        return None
        
    def collect_dependencies(self) -> List[Dict]:
        """Collect dependencies for the target function or class."""
        if not self.find_target():
            return []
            
        # Analyze the target file
        with open(self.target_file, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        collector = DependencyCollector(self.target_file)
        collector.visit(tree)
        
        # Find external dependencies (items in other files)
        for item in collector.function_calls + collector.class_refs + collector.references:
            result = self.find_file_for_item(item)
            if result:
                file_path, line_range = result
                if file_path != self.target_file:  # Only include external dependencies
                    # Determine type based on which list it's in
                    if item in collector.function_calls:
                        item_type = "function"
                    elif item in collector.class_refs:
                        item_type = "class"
                    else:
                        # For references, we can't be sure if it's a function or class
                        # until we find it, so we'll just call it a reference
                        item_type = "reference"
    
                    self.external_dependencies.append({
                        "name": item,
                        "type": item_type,
                        "file": str(file_path),
                        "line_range": line_range
                    })
            # if result:
            #     file_path, line_range = result
            #     if file_path != self.target_file:  # Only include external dependencies
            #         self.external_dependencies.append({
            #             "name": item,
            #             "type": "function" if item in collector.function_calls else "class",
            #             "file": str(file_path),
            #             "line_range": line_range
            #         })
        
        # Sort dependencies by file and line number
        self.external_dependencies.sort(key=lambda x: (x["file"], x["line_range"][0]))
        return self.external_dependencies
        
    def analyze(self) -> Dict:
        """Analyze the target and return the results."""
        dependencies = self.collect_dependencies()
        
        if not self.target_file:
            return {
                "status": "not_found",
                "message": f"Could not find {self.target_name} in the specified directories"
            }
            
        return {
            "status": "found",
            "target": self.target_name,
            "file": str(self.target_file),
            "line_range": self.target_line_range,
            "dependencies": dependencies
        }


def analyze_dependencies(target_name: str, 
                         search_dirs: Optional[List[str]] = None, # None
                         contextualizer: Optional[bool] = False, # False
                         exclude_from_contextualizer: Optional[List[str]] = None # None
                        ) -> str:
    """Analyze dependencies for a given function or class name.
    
    Args:
        target_name: Name of the function or class to analyze
        search_dirs: List of directories to search in
        
    Returns:
        Str containing dependency information or 'not_found' message
    """
    if search_dirs is None: 
        # Use current working directory as the primary search location for code analysis
        search_dirs = [os.getcwd(), "/tmp/"]
    print(f"Searching for '{target_name}' in {search_dirs}")
    analyzer = CrossFileDependencyAnalyzer(target_name, search_dirs)
    result = analyzer.analyze()
    
    if result["status"] == "found":
        my_dict = f"Found {target_name} in {result['file']} (lines {result['line_range'][0]}-{result['line_range'][1]})"
        
        if result["dependencies"]:
            my_dict += f"\nExternal dependencies required to understand {target_name}:"
            for i, dep in enumerate(result["dependencies"], 1):
                my_dict += f"  {i}. {dep['name']} ({dep['type']}) - {dep['file']} lines {dep['line_range'][0]}-{dep['line_range'][1]}"
        else:
            my_dict = f"\nNo external dependencies found for {target_name}"
    else:
        my_dict = f"\nERROR: {result['message']}"
        
    my_dict = str(result)
    
    
    if contextualizer:
        result["context"] = {}
        # Add the target file first
        if result["status"] == "found":
            target_file = result["file"]
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    # Extract only the lines in the range
                    start, end = result["line_range"]
                    # Adjust for 0-based indexing
                    relevant_lines = all_lines[start-1:end]
                    target_content = ''.join(relevant_lines)
                    
                result["context"][target_file] = {
                    "content": target_content,
                    "line_range": result["line_range"]
                }
                my_dict += f"\n\nFile: {target_file} (lines {result['line_range'][0]}-{result['line_range'][1]})\n```python\n{target_content}\n```"
            except Exception as e:
                my_dict += f"\n\nFile: {target_file} (ERROR: {str(e)})"
            
            # Add dependency files
            for dep in result.get("dependencies", []):
                dep_file = dep["file"]
                # Skip if in exclude list
                if exclude_from_contextualizer and any(exclude in dep_file for exclude in exclude_from_contextualizer):
                    continue
                try:
                    with open(dep_file, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                        # Extract only the lines in the range
                        start, end = dep["line_range"]
                        # Adjust for 0-based indexing
                        relevant_lines = all_lines[start-1:end]
                        dep_content = ''.join(relevant_lines)
                        
                    result["context"][dep_file] = {
                        "content": dep_content,
                        "line_range": dep["line_range"]
                    }
                    my_dict += f"\n\nDependency: {dep_file} (lines {dep['line_range'][0]}-{dep['line_range'][1]})\n```python\n{dep_content}\n```"
                except Exception as e:
                    my_dict += f"\n\nDependency: {dep_file} (ERROR: {str(e)})"
    
    if contextualizer:
        return f"===💻🗺️CodeLocalization===\n\nNote: CodeLocalizerTool only returns dependencies that are local to the codebase and not in the same file. If the target imports from libraries, it will not explain those imports. If the target has dependencies in the same file, it will also not explain those. If `dependencies` k has empty v, you only need to read the file containing the target.\n\nRead the file the target is in in its entirety (to understand the imports and any local definitions it uses), unless you've already done so or it doesn't seem necessary.\n\nResults:\n\n{my_dict}\n\n===/💻🗺️CodeLocalization==="
    else:
        return f"===💻🗺️CodeLocalization===\n\nNote: CodeLocalizerTool only returns dependencies that are local to the codebase and not in the same file. If the target imports from libraries, it will not explain those imports. If the target has dependencies in the same file, it will also not explain those. If `dependencies` k has empty v, you only need to read the file containing the target. Read the file the target is in in its entirety (to understand the imports and any local definitions it uses), unless you've already done so or it doesn't seem necessary. To contextualize the target, you should view the target file and then view the files with the ranges provided in the dependencies.\n\nResults:\n\n{my_dict}\n\n===/💻🗺️CodeLocalization==="


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze cross-file dependencies for a function or class.')
    parser.add_argument('target_name', help='Name of the function or class to analyze')
    parser.add_argument('dirs', nargs='+', help='Directories to search in')
    
    args = parser.parse_args()
    
    result = analyze_dependencies(args.target_name, args.dirs)