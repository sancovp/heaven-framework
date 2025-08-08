from typing import List, Dict, Type, Any
import os
import importlib
import importlib.util
import inspect

def get_tool_modules() -> str:
    """Get formatted string of available tool classes from tools.__all__ and HEAVEN_DATA_DIR/tools/"""
    available_tools = []
    
    # Get built-in tools from heaven_base
    try:
        tools_module = importlib.import_module("heaven_base.tools")
        heaven_tools = [tool for tool in getattr(tools_module, '__all__', []) if tool != "HermesTool"]
        available_tools.extend(heaven_tools)
    except (ImportError, AttributeError):
        # Fallback to basic tools we know exist
        available_tools.extend(["StraightforwardSummarizerTool", "WriteBlockReportTool"])
    
    # Get custom tools from HEAVEN_DATA_DIR/tools/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_tools_path = os.path.join(heaven_data_dir, 'tools')
        if os.path.exists(custom_tools_path):
            for filename in os.listdir(custom_tools_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    # Try to import and find tool classes in the file
                    try:
                        spec = importlib.util.spec_from_file_location(
                            filename[:-3], os.path.join(custom_tools_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Look for classes ending in 'Tool'
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if name.endswith('Tool') and name != 'Tool':
                                available_tools.append(name)
                    except Exception:
                        pass  # Skip files that can't be imported
    
    return ", ".join(sorted(set(available_tools)))  # Remove duplicates

def get_agent_modules() -> str:
    """Get formatted string of available agent directories from heaven_base and HEAVEN_DATA_DIR"""
    modules = []
    
    # Get built-in agents from heaven_base
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_path = os.path.join(os.path.dirname(current_dir), "agents")
    
    if os.path.exists(agents_path):
        for item in os.listdir(agents_path):
            full_path = os.path.join(agents_path, item)
            if (os.path.isdir(full_path) and 
                not item.startswith('__') and 
                item != '__pycache__'):
                modules.append(item)
    
    # Get custom agents from HEAVEN_DATA_DIR/agents/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_agents_path = os.path.join(heaven_data_dir, 'agents')
        if os.path.exists(custom_agents_path):
            for item in os.listdir(custom_agents_path):
                full_path = os.path.join(custom_agents_path, item)
                if (os.path.isdir(full_path) and 
                    not item.startswith('__') and 
                    item != '__pycache__'):
                    modules.append(item)
    
    # Fallback if no agents directory exists
    if not modules:
        modules = ["summary_agent"]
    
    return ", ".join(sorted(set(modules)))  # Remove duplicates


def get_tool_class(tool_name: str) -> Type:
    """Get tool class by name from heaven_base.tools or HEAVEN_DATA_DIR/tools/"""
    # Try heaven_base.tools first
    try:
        tools_module = importlib.import_module("heaven_base.tools")
        tool_class = getattr(tools_module, tool_name)
        return tool_class
    except (ImportError, AttributeError):
        pass
    
    # Try custom tools from HEAVEN_DATA_DIR/tools/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_tools_path = os.path.join(heaven_data_dir, 'tools')
        if os.path.exists(custom_tools_path):
            for filename in os.listdir(custom_tools_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            filename[:-3], os.path.join(custom_tools_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        if hasattr(module, tool_name):
                            return getattr(module, tool_name)
                    except Exception:
                        continue
    
    raise ImportError(f"Could not find tool class: {tool_name}")


def get_all_tool_classes() -> Dict[str, Type]:
    """Get all tool classes from both heaven_base.tools and HEAVEN_DATA_DIR/tools/"""
    all_tools = {}
    
    # Import from heaven_base.tools
    try:
        tools_module = importlib.import_module("heaven_base.tools")
        for tool_name in getattr(tools_module, '__all__', []):
            if tool_name != "HermesTool":
                all_tools[tool_name] = getattr(tools_module, tool_name)
    except (ImportError, AttributeError):
        pass
    
    # Import from HEAVEN_DATA_DIR/tools/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_tools_path = os.path.join(heaven_data_dir, 'tools')
        if os.path.exists(custom_tools_path):
            for filename in os.listdir(custom_tools_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            filename[:-3], os.path.join(custom_tools_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Get all classes ending in 'Tool'
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if name.endswith('Tool') and name != 'Tool':
                                all_tools[name] = obj
                    except Exception:
                        continue
    
    return all_tools


def get_agent_class(agent_name: str) -> Type:
    """Get agent class by name from heaven_base.agents or HEAVEN_DATA_DIR/agents/"""
    # Try heaven_base.agents first
    try:
        agent_module = importlib.import_module(f"heaven_base.agents.{agent_name}")
        # Look for main agent class (usually named after the module)
        for name, obj in inspect.getmembers(agent_module, inspect.isclass):
            if name.lower().replace('_', '') == agent_name.lower().replace('_', ''):
                return obj
    except (ImportError, AttributeError):
        pass
    
    # Try custom agents from HEAVEN_DATA_DIR/agents/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_agents_path = os.path.join(heaven_data_dir, 'agents', agent_name)
        if os.path.exists(custom_agents_path):
            # Look for main agent file
            for filename in os.listdir(custom_agents_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            filename[:-3], os.path.join(custom_agents_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Look for agent classes
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if name.lower().replace('_', '') == agent_name.lower().replace('_', ''):
                                return obj
                    except Exception:
                        continue
    
    raise ImportError(f"Could not find agent class: {agent_name}")


def get_all_agent_classes() -> Dict[str, Type]:
    """Get all agent classes from both heaven_base.agents and HEAVEN_DATA_DIR/agents/"""
    all_agents = {}
    
    # Import from heaven_base.agents
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_path = os.path.join(os.path.dirname(current_dir), "agents")
        
        if os.path.exists(agents_path):
            for agent_dir in os.listdir(agents_path):
                if (os.path.isdir(os.path.join(agents_path, agent_dir)) and 
                    not agent_dir.startswith('__')):
                    try:
                        agent_module = importlib.import_module(f"heaven_base.agents.{agent_dir}")
                        for name, obj in inspect.getmembers(agent_module, inspect.isclass):
                            if name.lower().replace('_', '') == agent_dir.lower().replace('_', ''):
                                all_agents[name] = obj
                    except Exception:
                        continue
    except Exception:
        pass
    
    # Import from HEAVEN_DATA_DIR/agents/
    heaven_data_dir = os.environ.get('HEAVEN_DATA_DIR')
    if heaven_data_dir:
        custom_agents_path = os.path.join(heaven_data_dir, 'agents')
        if os.path.exists(custom_agents_path):
            for agent_dir in os.listdir(custom_agents_path):
                agent_full_path = os.path.join(custom_agents_path, agent_dir)
                if (os.path.isdir(agent_full_path) and not agent_dir.startswith('__')):
                    for filename in os.listdir(agent_full_path):
                        if filename.endswith('.py') and not filename.startswith('__'):
                            try:
                                spec = importlib.util.spec_from_file_location(
                                    filename[:-3], os.path.join(agent_full_path, filename)
                                )
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                for name, obj in inspect.getmembers(module, inspect.isclass):
                                    if name.lower().replace('_', '') == agent_dir.lower().replace('_', ''):
                                        all_agents[name] = obj
                            except Exception:
                                continue
    
    return all_agents