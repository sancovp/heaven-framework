import json
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from enum import Enum
from .utils.get_env_value import EnvConfigUtil, DynamicString
import requests

def get_uni_api_url():
    """Load uni-api URL from config file with automatic container IP discovery"""
    
    def discover_uni_api_ip():
        """Automatically discover uni-api container IP"""
        try:
            import subprocess
            result = subprocess.run(
                ['docker', 'inspect', 'uni-api', '--format', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],
                capture_output=True, text=True, check=True
            )
            container_ip = result.stdout.strip()
            if container_ip:
                return f'http://{container_ip}:8000/v1/chat/completions'
        except Exception as e:
            print(f"Warning: Could not discover uni-api container IP: {e}")
        return None
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'uni_api_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        configured_url = config.get('uni_api_url')
        
        # If config exists, try it first, but fall back to auto-discovery if it fails
        if configured_url:
            return configured_url
            
    except Exception as e:
        print(f"Warning: Could not load uni-api config: {e}")
    
    # Try automatic discovery
    discovered_url = discover_uni_api_ip()
    if discovered_url:
        print(f"Auto-discovered uni-api URL: {discovered_url}")
        return discovered_url
    
    # Final fallback
    print("Using fallback uni-api URL")
    return 'http://host.docker.internal:8002/v1/chat/completions'

# Define the Provider Enum 
class ProviderEnum(Enum):
    ANTHROPIC = 'anthropic'
    OPENAI = 'openai'
    GOOGLE = 'google'
    GROQ = 'groq'
    DEEPSEEK = 'deepseek'

# Default model mapping
# DEFAULT_MODELS = {
#     ProviderEnum.ANTHROPIC: "claude-3-5-sonnet-20241022",
#    ProviderEnum.OPENAI: "gpt-4o",
#    ProviderEnum.GOOGLE: "gemini-2.0-pro-exp-02-05",
#    ProviderEnum.GROQ: "deepseek-r1-distill-llama-70b-specdec",
#    ProviderEnum.DEEPSEEK: "deepseek-reasoner"
# }

class UnifiedChat:
    PROVIDERS = {
        ProviderEnum.ANTHROPIC: ChatAnthropic,
        ProviderEnum.OPENAI: ChatOpenAI,
        ProviderEnum.GOOGLE: ChatGoogleGenerativeAI,
        ProviderEnum.GROQ: ChatGroq,
        ProviderEnum.DEEPSEEK: ChatDeepSeek
    }

    @classmethod
    def create(
        cls, 
        provider: ProviderEnum = ProviderEnum.ANTHROPIC, 
        model: str = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a unified chat model instance from any supported provider.
        
        Args:
            provider (ProviderEnum): The provider Enum (anthropic, openai, google, groq).
            model (str, optional): Specific model name.
            **kwargs: Additional configuration for the model.
        
        Returns:
            BaseChatModel: Instantiated chat model.
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(cls.PROVIDERS.keys())}")
        
        ModelClass = cls.PROVIDERS[provider]
        
        # Load provider-specific settings ENV values, for now it is only the api key
        kwargs["api_key"] = DynamicString(EnvConfigUtil.get_env_value, f"{provider.name}_API_KEY")
        # If a model name is specified, add it to kwargs.
                
        # Use default model if none is specified
        if not model:
            model =  DynamicString(EnvConfigUtil.get_env_value, f"{provider.name}_MODEL")
        
        # this is an Anthropic specific value and should be removed and remapped in the kwargs
        thinking_budget = kwargs.get('thinking_budget', None)
        # kwargs.pop("thinking_budget", None) # dont pop them, langchain needs it

        # Add OpenAI-specific settings
        if provider == ProviderEnum.OPENAI:
            # kwargs.update({
            #     'model': model,
            #     'callbacks': None,
            #     'verbose': True
            # })
            kwargs.pop("thinking_budget", None) 
            kwargs.update({
                'model': model,
                'callbacks': None,
                'verbose': True,
                'use_responses_api': True,
                'model_kwargs': {
                    "reasoning": {
                        "effort": None,
                        "summary": "auto",
                    }
                }
            })
            # remove temperature from o series model params
            if "o3" in model or "o1" in model or "o4" in model:
                kwargs.pop("temperature", None)
                reasoning_effort = None
                # enable thinking if thinking should be enabled (budget is not none)
                if thinking_budget is not None:
                    # TODO we might need to make a dedicated chatgpt ui for setting reasoning-effort
                    if thinking_budget > 1024 * 6:
                        reasoning_effort = "high"
                    elif thinking_budget > 1024 * 3:
                        reasoning_effort = "medium"
                    else:
                        reasoning_effort = "low"
                
                if reasoning_effort:
                    kwargs.setdefault("model_kwargs", {})
                    kwargs["model_kwargs"]["reasoning"] = {
                        "effort": reasoning_effort,
                        "summary": "auto",
                    }
                    
            else: # for 4o or 4o-mini model temperature
                if 'temperature' not in kwargs:
                    kwargs['temperature'] = float(EnvConfigUtil.get_env_value("MODEL_DEFAULT_TEMP", 0.7))
        
        elif provider == ProviderEnum.GOOGLE:

            kwargs["model"] = model

            supports_thinking = "2.5" in model or "3." in model  # Gemini ≥2.5


            if thinking_budget is not None and supports_thinking:

                kwargs["thinking_budget"] = thinking_budget

                kwargs["include_thoughts"] = True

            else:

                # Explicitly disable any thought tracing
                # remove both keys completely

                kwargs.pop("include_thoughts", None)

                kwargs.pop("thinking_budget", None)
        # OLD unsafe
        # elif provider == ProviderEnum.GOOGLE:
        #     kwargs["model"] = model           # Gemini model name

        #     # 1) forward the budget (0 disables thoughts per Google docs)
        #     if thinking_budget is not None:
        #         kwargs["thinking_budget"] = thinking_budget     # int
        #         kwargs["include_thoughts"] = True               # turn trace on
        #     else:
        #         # expose thoughts unless the caller explicitly opted out
        #         kwargs.setdefault("include_thoughts", True)

        elif provider == ProviderEnum.ANTHROPIC:
            kwargs['model'] = model
            if extract_model_number(model) > 3.6 and thinking_budget is not None: # assuming only claude model 3.7 or higher allow thinking
                kwargs['thinking'] = {"type": "enabled", "budget_tokens": thinking_budget}
                kwargs['temperature'] = 0.7
        else:
            kwargs['model'] = model
        
        # print(f"\nCreating {provider} model: {model}")
        # print(f"With kwargs: {kwargs}\n")
        print(f">>> DEBUG: UnifiedChat: FINAL KWARGS: {kwargs}")
        return ModelClass(**kwargs)

    # @classmethod
    # def invoke_uni_api(
    #     cls,
    #     model: str,
    #     uni_messages,
    #     uni_api_url: str = None,
    #     **kwargs
    # ):
    #     """Invoke uni-api directly"""
    #     import requests
        
    #     # Use config URL if none provided
    #     if uni_api_url is None:
    #         uni_api_url = get_uni_api_url()
        
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": "Bearer sk-heaven-uni-api-test-12345"
    #     }
        
    #     payload = {
    #         "model": model,
    #         "messages": uni_messages,
    #         **kwargs
    #     }
        
    #     response = requests.post(uni_api_url, headers=headers, json=payload, timeout=120)
        
    #     # DEBUG: Print full request/response details on error
    #     if response.status_code != 200:
    #         print(f"🚨 400 ERROR DEBUG 🚨")
    #         print(f"Request URL: {uni_api_url}")
    #         print(f"Response status: {response.status_code}")
    #         print(f"Response body: {response.text}")
            
    #         # Print EXACT JSON of each message
    #         messages = payload.get("messages", [])
    #         print(f"\n📨 EXACT JSON CONVERSATION HISTORY ({len(messages)} messages):")
            
    #         for i, msg in enumerate(messages):
    #             print(f"Message {i}: {json.dumps(msg, indent=2)}")
    #             print("---")
            
    #         print(f"🚨 END DEBUG 🚨")
        
    #     if response.status_code != 200:
    #         print(f"🚨 uni-api ERROR {response.status_code}: {response.text}")
    #     response.raise_for_status()
    #     return response.json()

    # ---------------------------- helper: cleanse -------------------------- #
    @staticmethod
    def _cleanse_messages_for_uni_api(messages: list[dict]) -> list[dict]:
        """Enforce *one* tool_call per assistant message and fix null content."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("tool_calls"):
                # keep only the first request
                if len(msg["tool_calls"]) > 1:
                    msg["tool_calls"] = [msg["tool_calls"][0]]
                # OpenAI requires content to be ""
                if msg.get("content") is None:
                    msg["content"] = ""
        return messages

    # --------------------------- uni-api direct call ----------------------- #
    @classmethod
    def invoke_uni_api(
        cls,
        model: str,
        uni_messages: list[dict],
        uni_api_url: str | None = None,
        **kwargs,
    ):
        """Low-level POST to uni-api with built-in single-tool guard."""
        if uni_api_url is None:
            uni_api_url = get_uni_api_url()

        # --- cleansing guard here ----------------------------------------- #
        uni_messages = cls._cleanse_messages_for_uni_api(uni_messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-heaven-uni-api-test-12345",
        }
        payload = {"model": model, "messages": uni_messages, **kwargs}

        response = requests.post(uni_api_url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            print(f"🚨 uni-api ERROR {response.status_code}")
            print(response.text)
            # dump conversation for fast debugging
            for i, m in enumerate(payload["messages"]):
                print(f"[{i}] {json.dumps(m, indent=2)}")
        response.raise_for_status()
        return response.json()



def extract_model_number(model: str) -> float:
    """
    Given a string of the form 'claude-3-7-something-20250219' 
    or 'claude-4-something-20250219', extract the numeric parts 
    and combine into a decimal (e.g., 3.7 or 4.0).
    
    Rules:
      - The first numeric part we encounter is the integer portion.
      - The next numeric part (if present) is the decimal portion.
      - If no second numeric part is found, use 0 as the decimal portion.
    """
    parts = model.split("-")

    integer_part = None
    decimal_part = "0"  # default if we don't find a second numeric part
    
    for i, part in enumerate(parts):
        # Check if the current part is strictly numeric
        if part.isdigit():
            integer_part = part
            # If next part exists and is numeric, treat that as the decimal part
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                decimal_part = parts[i + 1]
            break  # we found what we need, so stop
    
    # If no numeric part was found at all (very edge case), return 0.0 or raise an error
    if integer_part is None:
        return 0.0
    return float(f"{integer_part}.{decimal_part}")


# Example usage:
# if __name__ == "__main__":
#     # Create a chat model instance using UnifiedChat.
#     chat = UnifiedChat.create(
#         provider=ProviderEnum.GROQ, 
#         model='deepseek-r1-distill-llama-70b',
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2
#     )
#     response = chat.invoke([
#         (
#             "system",
#             "You are a helpful assistant that translates English to French. Translate the user sentence.",
#         ),
#         ("human", "I love programming."),
#     ])

#     print(response.content)
