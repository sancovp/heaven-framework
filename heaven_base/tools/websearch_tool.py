# websearch_tool.py

from typing import Dict, Any
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema, ToolResult, ToolError
from ..utils.get_env_value import EnvConfigUtil, KnownEnvVar
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

tool_log_path = "/tmp/tool_debug.log"

async def _perform_websearch(
    search_goal: str,
    history_id: str = None
):
    try:
        # llm = init_chat_model("openai:gpt-4o-mini", 
        llm = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True, api_key=EnvConfigUtil.get_env_value(KnownEnvVar.OPENAI_API_KEY))
        tool = {"type": "web_search_preview"}
        llm_with_tools = llm.bind_tools([tool])
        if history_id:
            response = llm_with_tools.invoke(
                search_goal,
                previous_response_id=history_id
            )
        else:
            response = llm_with_tools.invoke(search_goal)
        
        search_result = response.text()
        search_id = response.response_metadata["id"]
        # with open(tool_log_path, 'a') as f:
        #     f.write(f"\n\nUsing websearch for {search_goal}\n"
        #             f"search id: {search_id}"
        #             f"search result: {search_result}")
        return ToolResult(output=f"WebsearchTool Respone:\n\n" 
                        f"Search result:\n\n{search_result}\n\n\n"
                        f"Search_id: {search_id}\n")
    except Exception as e:
        raise ToolError(f"WebsearchTool ran into error: {e} This occurred while trying to search for: {search_goal}")


class WebsearchArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'search_goal': {
            'name': 'search_goal',
            'type': 'str',
            'description': 'A sentence or two describing what you want to search for. Be specific and concise.',
            'required': True
        },
        'history_id': {
            'name': 'history_id',
            'type': 'str',
            'description': 'A id that allow you to continue a previous search. If you want to start a new search, leave this empty.',
            'required': False
        }
    }


class WebsearchTool(BaseHeavenTool):
    name = "WebsearchTool"
    description = "Use websearch to search the web for information. This tool allows you to search the web and retrieve relevant information to assist in your tasks."
    func = _perform_websearch
    args_schema = WebsearchArgsSchema
    is_async = True
