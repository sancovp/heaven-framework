"""
HEAVEN Callback Functions for LangGraph Integration

Provides different callback types for capturing and processing HEAVEN events
in workflows. These callbacks work with the existing heaven_main_callback
parameter in agent.run().
"""

import json
from typing import Any, List, Dict, Optional, Callable
from ...memory.heaven_event import HeavenEvent


class BackgroundEventCapture:
    """
    Callback that captures HEAVEN events for workflow state management.
    Events are stored and can be retrieved for LangGraph state updates.
    """
    
    def __init__(self):
        self.captured_events: List[Dict[str, Any]] = []
        self.event_count = 0
    
    def __call__(self, raw_langchain_message: Any):
        """Process raw LangChain message and capture as HEAVEN events"""
        try:
            # Convert to HEAVEN events
            heaven_events = HeavenEvent.from_langchain_message(raw_langchain_message)
            
            # Store as dictionaries for workflow state
            for event in heaven_events:
                event_dict = event.to_dict()
                event_dict["capture_index"] = self.event_count
                self.captured_events.append(event_dict)
                self.event_count += 1
                
        except Exception as e:
            print(f"ERROR in BackgroundEventCapture: {e}")
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all captured events"""
        return self.captured_events.copy()
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get events of specific type"""
        return [e for e in self.captured_events if e.get("event_type") == event_type]
    
    def clear(self):
        """Clear captured events"""
        self.captured_events.clear()
        self.event_count = 0


class PrintEventLogger:
    """
    Callback that prints HEAVEN events for debugging.
    Provides formatted output for different event types.
    """
    
    def __init__(self, verbose: bool = False, prefix: str = "🔍"):
        self.verbose = verbose
        self.prefix = prefix
        self.event_count = 0
    
    def __call__(self, raw_langchain_message: Any):
        """Process and print HEAVEN events"""
        try:
            heaven_events = HeavenEvent.from_langchain_message(raw_langchain_message)
            
            for event in heaven_events:
                self.event_count += 1
                self._print_event(event)
                
        except Exception as e:
            print(f"ERROR in PrintEventLogger: {e}")
    
    def _print_event(self, event: HeavenEvent):
        """Format and print a single event"""
        if event.event_type == "THINKING":
            content = event.data.get("content", "")[:100] + "..." if len(event.data.get("content", "")) > 100 else event.data.get("content", "")
            print(f"{self.prefix} [{event.event_type}] {content}")
            
        elif event.event_type == "TOOL_USE":
            tool_name = event.data.get("name", "unknown")
            print(f"{self.prefix} [{event.event_type}] {tool_name}")
            if self.verbose:
                print(f"    Input: {event.data.get('input', {})}")
                
        elif event.event_type == "TOOL_RESULT":
            output = str(event.data.get("output", ""))[:100] + "..." if len(str(event.data.get("output", ""))) > 100 else str(event.data.get("output", ""))
            print(f"{self.prefix} [{event.event_type}] {output}")
            
        elif event.event_type == "AGENT_MESSAGE":
            content = event.data.get("content", "")[:100] + "..." if len(event.data.get("content", "")) > 100 else event.data.get("content", "")
            print(f"{self.prefix} [{event.event_type}] {content}")
            
        else:
            print(f"{self.prefix} [{event.event_type}] {event.data}")


class HTTPEventStreamer:
    """
    Callback that formats events for HTTP/SSE streaming.
    Used for cross-container communication.
    """
    
    def __init__(self, event_queue: Optional[List] = None):
        self.event_queue = event_queue or []
        self.event_count = 0
    
    def __call__(self, raw_langchain_message: Any):
        """Process and queue events for HTTP streaming"""
        try:
            heaven_events = HeavenEvent.from_langchain_message(raw_langchain_message)
            
            for event in heaven_events:
                sse_event = {
                    "event": "heaven_event",
                    "data": json.dumps({
                        **event.to_dict(),
                        "stream_index": self.event_count
                    })
                }
                self.event_queue.append(sse_event)
                self.event_count += 1
                
        except Exception as e:
            error_event = {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            self.event_queue.append(error_event)


class CompositeCallback:
    """
    Callback that delegates to multiple other callbacks.
    Useful for combining different callback behaviors.
    """
    
    def __init__(self, callbacks: List[Callable]):
        self.callbacks = callbacks
    
    def __call__(self, raw_langchain_message: Any):
        """Call all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(raw_langchain_message)
            except Exception as e:
                print(f"ERROR in CompositeCallback delegate: {e}")


# Callback Factory Functions

def create_callback_from_config(callback_config: Dict[str, Any], **kwargs) -> Optional[Callable]:
    """
    Factory function to create callbacks from configuration.
    
    Args:
        callback_config: Configuration dictionary
        **kwargs: Additional arguments (e.g., main_callback, event_queue)
    
    Returns:
        Configured callback function or None
    """
    callback_type = callback_config.get("type", "none")
    
    if callback_type == "background":
        return BackgroundEventCapture()
        
    elif callback_type == "print":
        verbose = callback_config.get("verbose", False)
        prefix = callback_config.get("prefix", "🔍")
        return PrintEventLogger(verbose=verbose, prefix=prefix)
        
    elif callback_type == "http":
        event_queue = kwargs.get("event_queue", [])
        return HTTPEventStreamer(event_queue=event_queue)
        
    elif callback_type == "main":
        # Return the main callback passed from frontend
        return kwargs.get("main_callback")
        
    elif callback_type == "composite":
        # Create multiple callbacks
        sub_configs = callback_config.get("callbacks", [])
        callbacks = []
        for sub_config in sub_configs:
            sub_callback = create_callback_from_config(sub_config, **kwargs)
            if sub_callback:
                callbacks.append(sub_callback)
        return CompositeCallback(callbacks) if callbacks else None
        
    else:
        return None


def create_workflow_callback(callback_type: str = "background", **kwargs) -> Optional[Callable]:
    """
    Convenience function to create workflow callbacks.
    
    Args:
        callback_type: Type of callback to create
        **kwargs: Additional configuration
    
    Returns:
        Configured callback function
    """
    config = {"type": callback_type, **kwargs}
    return create_callback_from_config(config, **kwargs)


# Example Usage and Testing

def demo_callback_usage():
    """Demonstrate different callback types"""
    
    # Background capture for workflow state
    background = BackgroundEventCapture()
    
    # Print logger for debugging
    print_logger = PrintEventLogger(verbose=True, prefix="🧪")
    
    # Composite callback (both capture and print)
    composite = CompositeCallback([background, print_logger])
    
    # Simulate some events (this would normally come from agent.run)
    from langchain_core.messages import AIMessage, HumanMessage
    
    test_messages = [
        HumanMessage(content="Test user message"),
        AIMessage(content="Test AI response"),
        AIMessage(content=[
            {"type": "thinking", "thinking": "Let me think about this..."},
            {"type": "text", "text": "Here's my response"}
        ])
    ]
    
    print("=== Testing Composite Callback ===")
    for msg in test_messages:
        composite(msg)
    
    print(f"\n=== Captured {len(background.get_events())} events ===")
    for event in background.get_events():
        print(f"  {event['event_type']}: {list(event['data'].keys())}")


if __name__ == "__main__":
    demo_callback_usage()