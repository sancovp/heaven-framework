from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)
import json

# def _safe_parse_arguments(arguments):
#     """Safely parse tool arguments, handling various formats and malformed JSON."""
#     if isinstance(arguments, dict):
#         return arguments
#     elif isinstance(arguments, str):
#         try:
#             return json.loads(arguments)
#         except json.JSONDecodeError as e:
#             print(f"DEBUG: JSON parse error in tool arguments: {e}")
#             print(f"DEBUG: Problematic arguments: {repr(arguments)}")
#             return {"raw_arguments": arguments}  # Preserve the original
#     else:
#         return {}
def _safe_parse_arguments(arguments):
    """Safely parse tool arguments, handling various formats and malformed JSON."""
    if isinstance(arguments, dict):
        return arguments
    elif isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            # Try to fix common issues
            try:
                # Replace single quotes with double quotes
                fixed = arguments.replace("'", '"')
                return json.loads(fixed)
            except:
                try:
                    # Try eval for Python dict format
                    return eval(arguments)
                except:
                    # Return empty dict if all fails
                    return {}
    else:
        return {}

class HeavenEvent(BaseModel):
    """Standard event format for HEAVEN system.
    
    This class provides a standardized event format that can be used across
    different agent frameworks (LangChain, ADK, etc.).
    
    SIMPLIFIED VERSION that relies on LangChain's standardized tool_calls attribute.
    """
    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "event_type": self.event_type,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeavenEvent':
        """Create a HeavenEvent from a dictionary."""
        return cls(
            event_type=data.get("event_type", "UNKNOWN"),
            data=data.get("data", {})
        )

    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> List['HeavenEvent']:
        """Convert a LangChain message to a HeavenEvent."""
        events = []
        
        # Handle System Messages
        if isinstance(message, SystemMessage):
            events.append(cls(
                event_type="SYSTEM_MESSAGE",
                data={"content": message.content}
            ))
            
        # Handle Human Messages
        elif isinstance(message, HumanMessage):
            events.append(cls(
                event_type="USER_MESSAGE",
                data={"content": message.content}
            ))
            
        # Handle AI Messages
        elif isinstance(message, AIMessage):
            reason = (
                message.additional_kwargs.get("reasoning") or
                message.additional_kwargs.get("response_metadata", {}).get("reasoning")
            )

            if reason:
                # ❶ Pull out the summary field when present
                summary_obj = reason.get("summary", reason)

                # ❷ Normalise to plain text
                if isinstance(summary_obj, list):
                    if summary_obj:  # non-empty list
                        summary_text = "\n".join(
                            part.get("text", str(part)) for part in summary_obj
                        )
                    else:            # empty list
                        summary_text = (
                            "The model generated internal reasoning but did not "
                            "expose a summary for this turn."
                        )
                elif isinstance(summary_obj, dict):              # single dict
                    summary_text = summary_obj.get("text", json.dumps(summary_obj, ensure_ascii=False))
                else:                                            # already a str
                    summary_text = str(summary_obj)

                # ❸ Emit a THINKING event only if we have non-empty text
                if summary_text.strip():
                    events.append(cls(
                        event_type="THINKING",
                        data={"content": summary_text.strip()}
                    ))
            # 1. Process content blocks in EXACT order - SKIP tool_use blocks
            if isinstance(message.content, list):
                # Handle Google list of strings
                if all(isinstance(x, str) for x in message.content):
                    events.append(cls(
                        event_type="AGENT_MESSAGE",
                        data={"content": "\n\n".join(message.content)}
                    ))
                else:
                    # Handle structured blocks
                    for block in message.content:
                        if isinstance(block, dict):
                            block_type = block.get('type')
                            
                            if block_type == 'thinking':
                                thinking_content = block.get('thinking', '')
                                if thinking_content.strip():
                                    events.append(cls(
                                        event_type="THINKING",
                                        data={"content": thinking_content}
                                    ))
                            elif block_type == 'text':
                                text_content = block.get('text', '')
                                if text_content.strip():
                                    events.append(cls(
                                        event_type="AGENT_MESSAGE",
                                        data={"content": text_content}
                                    ))
                            # SKIP tool_use blocks completely
                        elif isinstance(block, str) and block.strip():
                            events.append(cls(
                                event_type="AGENT_MESSAGE",
                                data={"content": block}
                            ))
    
            # 2. Handle string content
            elif isinstance(message.content, str) and message.content.strip():
                events.append(cls(
                    event_type="AGENT_MESSAGE",
                    data={"content": message.content}
                ))
            
            # 3. Handle tool calls from tool_calls attribute ONLY
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    events.append(cls(
                        event_type="TOOL_USE",
                        data={
                            "name": tool_call.get('name', 'unknown_tool'),
                            "id": tool_call.get('id', ''),
                            "input": tool_call.get('args', {}),
                            "provider": "LANGCHAIN"
                        }
                    ))
                    ### ADDED - sometimes tool_calls is in additional_kwargs
            # FALLBACK: Check additional_kwargs if tool_calls is empty!
            elif message.additional_kwargs.get('tool_calls'):
                for tool_call in message.additional_kwargs['tool_calls']:
                    # Handle the different format in additional_kwargs
                    if 'function' in tool_call:
                        # OpenAI format
                        events.append(cls(
                            event_type="TOOL_USE",
                            data={
                                "name": tool_call['function']['name'],
                                "id": tool_call.get('id', ''),
                                # FIXED: Handle case where arguments might already be dict or malformed JSON
                                # This was causing JSONDecodeError in AgentConfigTestTool assertion tests
                                "input": _safe_parse_arguments(tool_call['function']['arguments']),
                                "provider": "LANGCHAIN"
                            }
                        ))
                    else:
                        # Direct format
                        events.append(cls(
                            event_type="TOOL_USE",
                            data={
                                "name": tool_call.get('name', 'unknown_tool'),
                                "id": tool_call.get('id', ''),
                                "input": tool_call.get('args', {}),
                                "provider": "LANGCHAIN"
                            }
                        ))
                        ### 
    
        # Handle Tool Messages
        elif isinstance(message, ToolMessage):
            events.append(cls(
                event_type="TOOL_RESULT",
                data={
                    "output": message.content,
                    "tool_call_id": message.tool_call_id or ""
                }
            ))
    
        # Handle Unknown Messages
        else:
            events.append(cls(
                event_type="UNKNOWN",
                data={"content": str(message)}
            ))
    
        return events if events else [cls(event_type="UNKNOWN", data={"content": "Empty message"})]

  
    
    def to_langchain_message(self, provider: str = "ANTHROPIC") -> Optional[BaseMessage]:
        """Convert a HeavenEvent to a LangChain message.
        
        Args:
            provider: The provider to format the message for ("ANTHROPIC", "GOOGLE", "OPENAI")
        
        Returns:
            A LangChain message formatted appropriately for the specified provider
        """
        if self.event_type == "SYSTEM_MESSAGE":
            return SystemMessage(content=self.data.get("content", ""))
            
        elif self.event_type == "USER_MESSAGE":
            return HumanMessage(content=self.data.get("content", ""))
            
        elif self.event_type == "AGENT_MESSAGE":
            return AIMessage(content=self.data.get("content", ""))
            
        elif self.event_type == "TOOL_RESULT":
            return ToolMessage(
                content=self.data.get("output", ""),
                tool_call_id=self.data.get("tool_call_id", "")
            )
            
        elif self.event_type == "TOOL_USE":
            # Handle conversion back to provider-specific format
            event_provider = self.data.get("provider", provider)
            
            if event_provider == "ANTHROPIC" or provider == "ANTHROPIC":
                # Convert to Claude format (content list with tool_use objects)
                tool_use_block = {
                    "type": "tool_use",
                    "name": self.data.get("name", "unknown_tool"),
                    "id": self.data.get("id", ""),
                    "input": self.data.get("input", {})
                }
                return AIMessage(content=[tool_use_block])
                
            elif event_provider == "GOOGLE" or provider == "GOOGLE":
                # Create Gemini format
                func_call = {
                    "name": self.data.get("name", "unknown_tool"),
                    "arguments": json.dumps(self.data.get("input", {}))
                }
                
                # Set up tool_calls
                tool_calls = self.data.get("original_tool_calls")
                if not tool_calls:
                    tool_calls = [{
                        "name": self.data.get("name", "unknown_tool"),
                        "id": self.data.get("id", str(hash(self.data.get("name", "")))),
                        "args": self.data.get("input", {})
                    }]
                
                return AIMessage(
                    content="",
                    additional_kwargs={"function_call": func_call},
                    tool_calls=tool_calls
                )
                
            elif event_provider == "OPENAI" or provider == "OPENAI":
                # Create OpenAI format
                func_call = {
                    "name": self.data.get("name", "unknown_tool"),
                    "arguments": json.dumps(self.data.get("input", {}))
                }
                
                tool_call = {
                    "id": self.data.get("id", str(hash(self.data.get("name", "")))),
                    "function": func_call,
                    "type": "function"
                }
                
                return AIMessage(
                    content="",
                    additional_kwargs={"tool_calls": [tool_call]}
                )
                
            # Default fallback format (Claude-style)
            tool_use_block = {
                "type": "tool_use",
                "name": self.data.get("name", "unknown_tool"),
                "id": self.data.get("id", ""),
                "input": self.data.get("input", {})
            }
            return AIMessage(content=[tool_use_block])
            
        else:
            return None

# Helper functions
def is_heaven_event_dict(obj: Any) -> bool:
    """Check if an object is a HEAVEN event dictionary."""
    return (
        isinstance(obj, dict) and
        "event_type" in obj and
        "data" in obj
    )

def convert_langchain_to_heaven(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert a list of LangChain messages to HEAVEN events."""
    heaven_events = []
    for msg in messages:
        events = HeavenEvent.from_langchain_message(msg)  # This returns a list now
        for event in events:
            heaven_events.append(event.to_dict())
    return heaven_events

def convert_heaven_to_langchain(events: List[Dict[str, Any]], provider: str = "ANTHROPIC") -> List[BaseMessage]:
    """Convert a list of HEAVEN events to LangChain messages.
    
    Args:
        events: List of HeavenEvent dictionaries
        provider: The provider to format messages for ("ANTHROPIC", "GOOGLE", "OPENAI")
        
    Returns:
        List of LangChain messages formatted for the specified provider
    """
    messages = []
    for event in events:
        if is_heaven_event_dict(event):
            heaven_event = HeavenEvent.from_dict(event)
            message = heaven_event.to_langchain_message(provider=provider)
            if message:
                messages.append(message)
    return messages
    
# OLD VERSION

### Works but might be rendering stuff out of order
    # @classmethod
    # def from_langchain_message(cls, message: BaseMessage) -> List['HeavenEvent']:
    #     """Convert a LangChain message to a HeavenEvent.
        
    #     FIXED: Process content blocks in ORIGINAL ORDER instead of grouping by type.
    #     """
    #     events = []
        
    #     # Handle System Messages
    #     if isinstance(message, SystemMessage):
    #         events.append(cls(
    #             event_type="SYSTEM_MESSAGE",
    #             data={"content": message.content}
    #         ))
            
    #     # Handle Human Messages
    #     elif isinstance(message, HumanMessage):
    #         events.append(cls(
    #             event_type="USER_MESSAGE",
    #             data={"content": message.content}
    #         ))
            
    #     # Handle AI Messages
    #     elif isinstance(message, AIMessage):
    #         # 1. Handle Thinking/Reasoning
    #         if hasattr(message, 'additional_kwargs'):
    #             # OpenAI-style reasoning
    #             if 'reasoning' in message.additional_kwargs:
    #                 reasoning_data = message.additional_kwargs['reasoning']
    #                 if reasoning_data and 'summary' in reasoning_data:
    #                     reasoning_texts = []
    #                     for block in reasoning_data.get('summary', []):
    #                         if isinstance(block, dict) and 'text' in block:
    #                             reasoning_texts.append(block['text'])
    
    #                     if reasoning_texts:
    #                         events.append(cls(
    #                             event_type="THINKING",
    #                             data={"content": "\n\n".join(reasoning_texts)}
    #                         ))
    
    #         # 2. Handle Content - PROCESS IN ORIGINAL ORDER
    #         if isinstance(message.content, list):
    #             # 2a. Handle Gemini-style list of strings
    #             if all(isinstance(x, str) for x in message.content):
    #                 events.append(cls(
    #                     event_type="AGENT_MESSAGE",
    #                     data={"content": "\n".join(message.content)}
    #                 ))
    #             # 2b. Handle structured content - PROCESS BLOCKS IN ORDER
    #             else:
    #                 for block in message.content:
    #                     if isinstance(block, dict):
    #                         block_type = block.get('type')
                            
    #                         if block_type == 'thinking':
    #                             thinking_content = block.get('thinking', '')
    #                             if thinking_content.strip():
    #                                 events.append(cls(
    #                                     event_type="THINKING",
    #                                     data={"content": thinking_content}
    #                                 ))
    #                         elif block_type == 'text':
    #                             text_content = block.get('text', '')
    #                             if text_content.strip():
    #                                 events.append(cls(
    #                                     event_type="AGENT_MESSAGE",
    #                                     data={"content": text_content}
    #                                 ))
    #                         # elif block_type == 'tool_use':
    #                         #     events.append(cls(
    #                         #         event_type="TOOL_USE",
    #                         #         data={
    #                         #             "name": block.get('name', 'unknown_tool'),
    #                         #             "id": block.get('id', ''),
    #                         #             "input": block.get('input', {}),
    #                         #             "provider": "ANTHROPIC"
    #                         #         }
    #                         #     ))
    
    #         # 3. Handle Tool Calls
    #         # if hasattr(message, 'additional_kwargs'):
    #         #     # 3a. OpenAI-style tool calls in additional_kwargs
    #         #     if 'tool_calls' in message.additional_kwargs:
    #         #         for tool_call in message.additional_kwargs['tool_calls']:
    #         #             events.append(cls(
    #         #                 event_type="TOOL_USE",
    #         #                 data={
    #         #                     "name": tool_call['function'].get('name', 'unknown_tool'),
    #         #                     "id": tool_call.get('id', ''),
    #         #                     "input": json.loads(tool_call['function'].get('arguments', '{}')),
    #         #                     "provider": "OPENAI"
    #         #                 }
    #         #             ))
    #         #     # 3b. Gemini-style function call
    #         #     elif 'function_call' in message.additional_kwargs:
    #         #         function_call = message.additional_kwargs['function_call']
    #         #         events.append(cls(
    #         #             event_type="TOOL_USE",
    #         #             data={
    #         #                 "name": function_call.get('name', 'unknown_tool'),
    #         #                 "id": '',
    #         #                 "input": json.loads(function_call.get('arguments', '{}')),
    #         #                 "provider": "GOOGLE"
    #         #             }
    #         #         ))
    
    #         # 4. Standard LangChain tool_calls
    #         if hasattr(message, 'tool_calls') and message.tool_calls:
    #             for tool_call in message.tool_calls:
    #                 events.append(cls(
    #                     event_type="TOOL_USE",
    #                     data={
    #                         "name": tool_call.get('name', 'unknown_tool'),
    #                         "id": tool_call.get('id', ''),
    #                         "input": tool_call.get('args', {}),
    #                         "provider": "LANGCHAIN"
    #                     }
    #                 ))
    
    #         # 5. Handle remaining content
    #         if isinstance(message.content, str) and message.content.strip():
    #             events.append(cls(
    #                 event_type="AGENT_MESSAGE",
    #                 data={"content": message.content}
    #             ))
    
    #     # Handle Tool Messages
    #     elif isinstance(message, ToolMessage):
    #         events.append(cls(
    #             event_type="TOOL_RESULT",
    #             data={
    #                 "output": message.content,
    #                 "tool_call_id": message.tool_call_id or ""
    #             }
    #         ))
    
    #     # Handle Unknown Messages
    #     else:
    #         events.append(cls(
    #             event_type="UNKNOWN",
    #             data={"content": str(message)}
    #         ))
    
    #     return events if events else [cls(event_type="UNKNOWN", data={"content": "Empty message"})]
      
    # @classmethod
    # def from_langchain_message(cls, message: BaseMessage) -> List['HeavenEvent']:
    #     """Convert a LangChain message to a HeavenEvent.
    
    #     Handles provider-specific formats for tool calls:
    #     - Claude: Tool calls in message.content as list with 'type': 'tool_use'
    #     - OpenAI: Tool calls in additional_kwargs.tool_calls
    #     - Gemini: Tool calls in additional_kwargs.function_call and content as list of strings
    #     """
    #     events = []
        
    #     # Handle System Messages
    #     if isinstance(message, SystemMessage):
    #         events.append(cls(
    #             event_type="SYSTEM_MESSAGE",
    #             data={"content": message.content}
    #         ))
            
    #     # Handle Human Messages
    #     elif isinstance(message, HumanMessage):
    #         events.append(cls(
    #             event_type="USER_MESSAGE",
    #             data={"content": message.content}
    #         ))
            
    #     # Handle AI Messages
    #     elif isinstance(message, AIMessage):
    #         # 1. Handle Thinking/Reasoning
    #         if hasattr(message, 'additional_kwargs'):
    #             # OpenAI-style reasoning
    #             if 'reasoning' in message.additional_kwargs:
    #                 reasoning_data = message.additional_kwargs['reasoning']
    #                 if reasoning_data and 'summary' in reasoning_data:
    #                     reasoning_texts = []
    #                     for block in reasoning_data.get('summary', []):
    #                         if isinstance(block, dict) and 'text' in block:
    #                             reasoning_texts.append(block['text'])
    
    #                     if reasoning_texts:
    #                         events.append(cls(
    #                             event_type="THINKING",
    #                             data={"content": "\n\n".join(reasoning_texts)}
    #                         ))
    
    #         # 2. Handle Content
    #         if isinstance(message.content, list):
    #             # 2a. Handle Gemini-style list of strings
    #             if all(isinstance(x, str) for x in message.content):
    #                 events.append(cls(
    #                     event_type="AGENT_MESSAGE",
    #                     data={"content": "\n".join(message.content)}
    #                 ))
    #             # 2b. Handle Claude-style list of dicts
    #             else:
    #                 # Extract thinking blocks
    #                 thinking_blocks = [block for block in message.content 
    #                                  if isinstance(block, dict) and block.get('type') == 'thinking']
    #                 if thinking_blocks:
    #                     thinking_text = "\n\n".join([block.get('thinking', '') for block in thinking_blocks])
    #                     events.append(cls(
    #                         event_type="THINKING",
    #                         data={"content": thinking_text}
    #                     ))
    
    #                 # Extract text blocks
    #                 text_blocks = [block.get('text', '') for block in message.content 
    #                              if isinstance(block, dict) and block.get('type') == 'text']
    #                 if text_blocks:
    #                     events.append(cls(
    #                         event_type="AGENT_MESSAGE",
    #                         data={"content": "\n\n".join(text_blocks)}
    #                     ))
    
    #                 # Handle Claude tool_use blocks
    #                 for block in message.content:
    #                     if isinstance(block, dict) and block.get('type') == 'tool_use':
    #                         events.append(cls(
    #                             event_type="TOOL_USE",
    #                             data={
    #                                 "name": block.get('name', 'unknown_tool'),
    #                                 "id": block.get('id', ''),
    #                                 "input": block.get('input', {}),
    #                                 "provider": "ANTHROPIC"
    #                             }
    #                         ))
    
    #         # 3. Handle Tool Calls
    #         if hasattr(message, 'additional_kwargs'):
    #             # 3a. OpenAI-style tool calls in additional_kwargs
    #             if 'tool_calls' in message.additional_kwargs:
    #                 tool_call = message.additional_kwargs['tool_calls'][0]
    #                 events.append(cls(
    #                     event_type="TOOL_USE",
    #                     data={
    #                         "name": tool_call['function'].get('name', 'unknown_tool'),
    #                         "id": tool_call.get('id', ''),
    #                         "input": json.loads(tool_call['function'].get('arguments', '{}')),
    #                         "provider": "OPENAI",
    #                         "original_tool_calls": message.additional_kwargs['tool_calls']
    #                     }
    #                 ))
    #             # 3b. Gemini-style function call
    #             elif 'function_call' in message.additional_kwargs:
    #                 function_call = message.additional_kwargs['function_call']
    #                 events.append(cls(
    #                     event_type="TOOL_USE",
    #                     data={
    #                         "name": function_call.get('name', 'unknown_tool'),
    #                         "id": '', # Gemini doesn't provide this
    #                         "input": json.loads(function_call.get('arguments', '{}')),
    #                         "provider": "GOOGLE"
    #                     }
    #                 ))
    #             # 3c. Standard LangChain tool_calls
    #             elif hasattr(message, 'tool_calls') and message.tool_calls:
    #                 tool_call = message.tool_calls[0]
    #                 events.append(cls(
    #                     event_type="TOOL_USE",
    #                     data={
    #                         "name": tool_call.get('name', 'unknown_tool'),
    #                         "id": tool_call.get('id', ''),
    #                         "input": tool_call.get('args', {}),
    #                         "provider": "UNKNOWN",
    #                         "original_tool_calls": message.tool_calls
    #                     }
    #                 ))
    
    #         # 4. Handle remaining content
    #         if isinstance(message.content, str) and message.content.strip():
    #             events.append(cls(
    #                 event_type="AGENT_MESSAGE",
    #                 data={"content": message.content}
    #             ))
    
    #     # Handle Tool Messages
    #     elif isinstance(message, ToolMessage):
    #         events.append(cls(
    #             event_type="TOOL_RESULT",
    #             data={
    #                 "output": message.content,
    #                 "tool_call_id": message.tool_call_id or ""
    #             }
    #         ))
    
    #     # Handle Unknown Messages
    #     else:
    #         events.append(cls(
    #             event_type="UNKNOWN",
    #             data={"content": str(message)}
    #         ))
    
    #     return events if events else [cls(event_type="UNKNOWN", data={"content": "Empty message"})]

# from typing import Dict, Any, Optional, List, Union
# from pydantic import BaseModel, Field
# from langchain_core.messages import (
#     BaseMessage, 
#     SystemMessage, 
#     HumanMessage, 
#     AIMessage, 
#     ToolMessage
# )
# import json

# class HeavenEvent(BaseModel):
#     """Standard event format for HEAVEN system.
    
#     This class provides a standardized event format that can be used across
#     different agent frameworks (LangChain, ADK, etc.).
#     """
#     event_type: str
#     data: Dict[str, Any] = Field(default_factory=dict)
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary format."""
#         return {
#             "event_type": self.event_type,
#             "data": self.data
#         }
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'HeavenEvent':
#         """Create a HeavenEvent from a dictionary."""
#         return cls(
#             event_type=data.get("event_type", "UNKNOWN"),
#             data=data.get("data", {})
#         )
    
#     @classmethod
#     def from_langchain_message(cls, message: BaseMessage) -> 'HeavenEvent':
#         """Convert a LangChain message to a HeavenEvent."""
#         if isinstance(message, SystemMessage):
#             return cls(
#                 event_type="SYSTEM_MESSAGE",
#                 data={"content": message.content}
#             )
#         elif isinstance(message, HumanMessage):
#             return cls(
#                 event_type="USER_MESSAGE",
#                 data={"content": message.content}
#             )
#         elif isinstance(message, AIMessage):
#             return cls(
#                 event_type="AGENT_MESSAGE",
#                 data={"content": message.content}
#             )
#         elif isinstance(message, ToolMessage):
#             return cls(
#                 event_type="TOOL_RESULT",
#                 data={
#                     "output": message.content,
#                     "tool_call_id": message.tool_call_id or ""
#                 }
#             )
#         else:
#             return cls(
#                 event_type="UNKNOWN",
#                 data={"content": str(message)}
#             )
    
#     def to_langchain_message(self) -> Optional[BaseMessage]:
#         """Convert a HeavenEvent to a LangChain message."""
#         if self.event_type == "SYSTEM_MESSAGE":
#             return SystemMessage(content=self.data.get("content", ""))
#         elif self.event_type == "USER_MESSAGE":
#             return HumanMessage(content=self.data.get("content", ""))
#         elif self.event_type == "AGENT_MESSAGE":
#             return AIMessage(content=self.data.get("content", ""))
#         elif self.event_type == "TOOL_RESULT":
#             return ToolMessage(
#                 content=self.data.get("output", ""),
#                 tool_call_id=self.data.get("tool_call_id", "")
#             )
#         elif self.event_type == "TOOL_USE":
#             # Tool use events don't map directly to LangChain messages
#             # They're typically part of AIMessage.additional_kwargs
#             return None
#         else:
#             return None

# # Helper functions
# def is_heaven_event_dict(obj: Any) -> bool:
#     """Check if an object is a HEAVEN event dictionary."""
#     return (
#         isinstance(obj, dict) and
#         "event_type" in obj and
#         "data" in obj
#     )

# def convert_langchain_to_heaven(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
#     """Convert a list of LangChain messages to HEAVEN events."""
#     return [HeavenEvent.from_langchain_message(msg).to_dict() for msg in messages]

# def convert_heaven_to_langchain(events: List[Dict[str, Any]]) -> List[BaseMessage]:
#     """Convert a list of HEAVEN events to LangChain messages."""
#     messages = []
#     for event in events:
#         if is_heaven_event_dict(event):
#             heaven_event = HeavenEvent.from_dict(event)
#             message = heaven_event.to_langchain_message()
#             if message:
#                 messages.append(message)
#     return messages
