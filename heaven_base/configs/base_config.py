# base_config.py
from pydantic import BaseModel
from typing import Dict, Any, Union, Callable
import json

# BaseFunctionConfig is a Pydantic model that holds the configuration for a function.
class BaseFunctionConfig(BaseModel):
    func_name: str                      # The name of the function (for identification) | This should be Callable
    args_template: Dict[str, Any]       # The template for arguments; keys must match function parameter names
    
    @classmethod
    def load_from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "BaseFunctionConfig":
        """
        Create a BaseFunctionConfig instance from a JSON string or a dictionary.

        Args:
            json_data: A JSON string or a dict containing BaseFunctionConfig fields.

        Returns:
            BaseFunctionConfig: An instance populated with the provided data.
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        return cls(**data)

